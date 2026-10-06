"""IOPub collection keyed by the originating execute request."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CellMessages:
    """Notebook outputs and terminal information collected for one parent ID."""

    outputs: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None
    execution_count: int | None = None
    reply: dict[str, Any] | None = None
    idle: bool = False


class MessageCollector:
    """Continuously drain IOPub and retain messages in per-parent buckets.

    A bucket is registered before its execute request is sent.  Consequently,
    a delayed stream/error/idle message remains associated with that parent ID
    and can never be attributed to the next cell.
    """

    def __init__(self, client) -> None:
        self._client = client
        self._condition = threading.Condition()
        self._cells: dict[str, CellMessages] = {}
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._run, name="notebook-iopub", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def begin(self, parent_id: str) -> CellMessages:
        with self._condition:
            bucket = CellMessages()
            self._cells[parent_id] = bucket
            return bucket

    def set_reply(self, parent_id: str, reply: dict[str, Any]) -> None:
        with self._condition:
            bucket = self._cells[parent_id]
            bucket.reply = reply
            count = reply.get("content", {}).get("execution_count")
            if isinstance(count, int):
                bucket.execution_count = count
            self._condition.notify_all()

    def snapshot(self, parent_id: str) -> CellMessages:
        with self._condition:
            source = self._cells[parent_id]
            return CellMessages(
                outputs=list(source.outputs),
                error=None if source.error is None else dict(source.error),
                execution_count=source.execution_count,
                reply=None if source.reply is None else dict(source.reply),
                idle=source.idle,
            )

    def wait_idle(self, parent_id: str, timeout: float | None = None) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: self._cells[parent_id].idle, timeout=timeout
            )

    def close(self, timeout: float = 1.0) -> None:
        self._stopping.set()
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                message = self._client.get_iopub_msg(timeout=0.1)
            except queue.Empty:
                continue
            except Exception:
                if self._stopping.is_set():
                    return
                continue
            self._handle(message)

    def _handle(self, message: dict[str, Any]) -> None:
        parent_id = message.get("parent_header", {}).get("msg_id")
        if not parent_id:
            return
        message_type = message.get("msg_type")
        content = message.get("content", {})
        with self._condition:
            bucket = self._cells.get(parent_id)
            if bucket is None:
                return

            if message_type == "status" and content.get("execution_state") == "idle":
                bucket.idle = True
            elif message_type == "execute_input":
                count = content.get("execution_count")
                if bucket.execution_count is None and isinstance(count, int):
                    bucket.execution_count = count
            elif message_type == "stream":
                name = content.get("name")
                text = content.get("text", "")
                if name in ("stdout", "stderr") and isinstance(text, str):
                    bucket.outputs.append(
                        {"output_type": "stream", "name": name, "text": text}
                    )
            elif message_type in ("execute_result", "display_data"):
                data = content.get("data")
                metadata = content.get("metadata", {})
                if isinstance(data, dict) and isinstance(metadata, dict):
                    output = {
                        "output_type": message_type,
                        "data": data,
                        "metadata": metadata,
                    }
                    if message_type == "execute_result":
                        count = content.get("execution_count")
                        output["execution_count"] = count if isinstance(count, int) else None
                    bucket.outputs.append(output)
            elif message_type == "error":
                error = {
                    "output_type": "error",
                    "ename": str(content.get("ename", "Error")),
                    "evalue": str(content.get("evalue", "")),
                    "traceback": list(content.get("traceback", [])),
                }
                bucket.error = error
                bucket.outputs.append(error)
            self._condition.notify_all()
