"""Supervised execution of a prepared notebook via a fresh local kernel."""
from __future__ import annotations

import collections
import os
import queue
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from jupyter_client.blocking import BlockingKernelClient
from jupyter_client.connect import write_connection_file

from .messages import CellMessages, MessageCollector
from .notebook import PreparedNotebook, write_notebook_atomic

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CELL_TIMEOUT = 30.0
STARTUP_TIMEOUT = 20.0
TERMINAL_GRACE = 5.0


class NotebookExecutionError(RuntimeError):
    """A notebook failed with an uncaught exception."""


class NotebookTimeoutError(TimeoutError):
    """A notebook cell exceeded its timeout."""


class KernelStartupError(RuntimeError):
    """The private labflow_kernel process did not become ready."""


@dataclass(frozen=True)
class NotebookExecutionResult:
    """Outcome and location of an atomically written result notebook."""

    status: str  # "ok" | "error" | "timeout"
    output_path: Path
    executed_cells: int
    error: str | None = None
    ename: str | None = None
    evalue: str | None = None


class _KernelProcess:
    """Own one kernel subprocess, connection file and client sockets."""

    def __init__(self) -> None:
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self.connection_path: str | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self.client: BlockingKernelClient | None = None
        self._stderr_tail = collections.deque(maxlen=50)
        self._stderr_thread: threading.Thread | None = None

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("kernel process can only be started once")
        self._tempdir = tempfile.TemporaryDirectory(prefix="runbook-notebook-")
        self.connection_path = os.path.join(self._tempdir.name, "connection.json")
        write_connection_file(
            self.connection_path,
            ip="127.0.0.1",
            key=secrets.token_urlsafe(32).encode("ascii"),
        )

        self.process = subprocess.Popen(
            [sys.executable, "-m", "labflow_kernel", "-f", self.connection_path],
            cwd=os.getcwd(),
            env=self._environment(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._stderr_thread = threading.Thread(target=self._drain_stderr, name="notebook-kernel-stderr", daemon=True)
        self._stderr_thread.start()

        self.client = BlockingKernelClient()
        self.client.load_connection_file(self.connection_path)
        self.client.start_channels()
        try:
            self.client.wait_for_ready(timeout=STARTUP_TIMEOUT)
        except Exception as exc:
            stderr = self.stderr_text()
            self._close_resources()
            raise KernelStartupError(f"kernel did not become ready: {exc}\n{stderr}") from exc

    def interrupt(self) -> None:
        if self.client is not None:
            self.client.control_channel.send(
                self.client.session.msg("interrupt_request", {})
            )

    @property
    def exited(self) -> bool:
        return self.process is not None and self.process.poll() is not None

    def close(self) -> None:
        self._close_resources()

    def _close_resources(self) -> None:
        process = self.process
        client = self.client
        tempdir = self._tempdir
        stderr_thread = self._stderr_thread
        self.client = None
        self.process = None
        self._tempdir = None
        self._stderr_thread = None

        if process is not None and process.poll() is None and client is not None:
            try:
                mid = client.shutdown()
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline and process.poll() is None:
                    if client.control_channel.poll(50):
                        reply = client.get_control_msg(timeout=0)
                        if reply["parent_header"].get("msg_id") == mid:
                            break
            except Exception:
                pass

        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)

        if client is not None:
            try:
                client.stop_channels()
            finally:
                if getattr(client, "_created_context", False):
                    client.context.destroy(linger=0)

        if stderr_thread is not None:
            stderr_thread.join(timeout=1.0)
        if tempdir is not None:
            tempdir.cleanup()

    def stderr_text(self) -> str:
        return "".join(self._stderr_tail)

    def _environment(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def _drain_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        for raw in process.stderr:
            self._stderr_tail.append(raw.decode("utf-8", errors="replace"))


def _apply_cell_messages(cell: dict[str, Any], messages: CellMessages) -> None:
    cell["outputs"] = list(messages.outputs)
    cell["execution_count"] = messages.execution_count


def _timeout_output(messages: CellMessages) -> dict[str, Any]:
    if messages.error is not None:
        return messages.error
    return {
        "output_type": "error",
        "ename": "TimeoutError",
        "evalue": "cell execution timed out",
        "traceback": ["TimeoutError: cell execution timed out"],
    }


def _kernel_died_output() -> dict[str, Any]:
    return {
        "output_type": "error",
        "ename": "KernelDied",
        "evalue": "the notebook kernel process exited",
        "traceback": ["KernelDied: the notebook kernel process exited"],
    }


def execute_prepared(
    prepared: PreparedNotebook,
    output_path: str | os.PathLike[str],
    cell_timeout: float = DEFAULT_CELL_TIMEOUT,
    continue_on_error: bool = False,
) -> NotebookExecutionResult:
    """Run a prepared notebook and always atomically write a terminal result."""
    if not isinstance(cell_timeout, (int, float)) or isinstance(cell_timeout, bool):
        raise TypeError("cell_timeout must be a number")
    timeout = float(cell_timeout)
    if timeout <= 0 or timeout != timeout or timeout == float("inf"):
        raise ValueError("cell_timeout must be a finite positive number of seconds")

    output = Path(output_path)
    kernel = _KernelProcess()
    collector: MessageCollector | None = None
    status = "ok"
    executed = 0
    error_name: str | None = None
    error_value: str | None = None

    try:
        kernel.start()
        assert kernel.client is not None
        collector = MessageCollector(kernel.client)
        collector.start()

        for cell_index, code in prepared.code_cells:
            cell = prepared.notebook["cells"][cell_index]
            # Register before sending: an early execute_input/busy message must
            # already have a bucket for this parent message ID.
            request = kernel.client.session.msg(
                "execute_request",
                {
                    "code": code,
                    "silent": False,
                    "store_history": True,
                    "user_expressions": {},
                    "allow_stdin": False,
                    "stop_on_error": False,
                },
            )
            parent_id = request["header"]["msg_id"]
            collector.begin(parent_id)
            kernel.client.shell_channel.send(request)
            deadline = time.monotonic() + timeout

            while time.monotonic() < deadline:
                try:
                    message = kernel.client.get_shell_msg(timeout=min(0.1, deadline - time.monotonic()))
                except queue.Empty:
                    message = None
                if (
                    message is not None
                    and message.get("msg_type") == "execute_reply"
                    and message.get("parent_header", {}).get("msg_id") == parent_id
                ):
                    collector.set_reply(parent_id, message)

                current = collector.snapshot(parent_id)
                if current.reply is not None and current.idle:
                    _apply_cell_messages(cell, current)
                    executed += 1
                    reply_status = current.reply.get("content", {}).get("status", "ok")
                    if reply_status == "error":
                        status = "error"
                        error_name = current.reply.get("content", {}).get("ename")
                        error_value = current.reply.get("content", {}).get("evalue")
                        if not continue_on_error:
                            result = NotebookExecutionResult(
                                status, output, executed,
                                "cell execution raised an exception", error_name, error_value,
                            )
                            write_notebook_atomic(prepared.notebook, output)
                            return result
                    break
                if kernel.exited:
                    _apply_cell_messages(cell, current)
                    if not any(item.get("output_type") == "error" for item in cell["outputs"]):
                        cell["outputs"].append(_kernel_died_output())
                    result = NotebookExecutionResult(
                        "error",
                        output,
                        executed + 1,
                        "the notebook kernel process exited",
                        cell["outputs"][-1].get("ename"),
                        cell["outputs"][-1].get("evalue"),
                    )
                    write_notebook_atomic(prepared.notebook, output)
                    return result
            else:
                # Timeout: interrupt this exact request, retain what arrived,
                # and give the kernel a short terminal grace before ending.
                kernel.interrupt()
                grace = time.monotonic() + TERMINAL_GRACE
                current = collector.snapshot(parent_id)
                while time.monotonic() < grace:
                    if current.reply is None:
                        try:
                            message = kernel.client.get_shell_msg(timeout=0.1)
                        except queue.Empty:
                            message = None
                        if (
                            message is not None
                            and message.get("msg_type") == "execute_reply"
                            and message.get("parent_header", {}).get("msg_id") == parent_id
                        ):
                            collector.set_reply(parent_id, message)
                    else:
                        # The shell reply is terminal.  Wait only for its
                        # corresponding idle, which the collector thread tracks.
                        collector.wait_idle(parent_id, timeout=0.1)
                    current = collector.snapshot(parent_id)
                    if current.reply is not None and current.idle:
                        break
                _apply_cell_messages(cell, current)
                if not any(output.get("output_type") == "error" for output in cell["outputs"]):
                    timeout_error = _timeout_output(current)
                    cell["outputs"].append(timeout_error)
                status = "timeout"
                error_name = cell["outputs"][-1].get("ename")
                error_value = cell["outputs"][-1].get("evalue")
                executed += 1
                result = NotebookExecutionResult(
                    status, output, executed, "cell execution timed out",
                    error_name, error_value,
                )
                write_notebook_atomic(prepared.notebook, output)
                return result

        result = NotebookExecutionResult(status, output, executed, None, error_name, error_value)
        write_notebook_atomic(prepared.notebook, output)
        return result
    finally:
        if collector is not None:
            collector.close()
        kernel.close()


def execute_notebook(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    parameters: Mapping[str, Any] | str | None = None,
    cell_timeout: float = DEFAULT_CELL_TIMEOUT,
    continue_on_error: bool = False,
) -> NotebookExecutionResult:
    """Load an input notebook, execute it privately, and write a result copy.

    The input document is never modified.  Success, handled exception and
    timeout outcomes all commit the result notebook atomically before cleanup.
    """
    from .notebook import load_notebook, prepare_notebook

    source = Path(input_path).resolve()
    destination = Path(output_path).resolve()
    try:
        if (
            source == destination
            or (source.exists() and destination.exists() and os.path.samefile(source, destination))
        ):
            raise ValueError("output_path must not be the same file as input_path")
    except FileNotFoundError:
        pass

    original = load_notebook(source)
    prepared = prepare_notebook(original, parameters)
    return execute_prepared(
        prepared,
        destination,
        cell_timeout=cell_timeout,
        continue_on_error=continue_on_error,
    )
