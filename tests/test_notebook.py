from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from labflow_kernel import MessageCollector, NotebookValidationError, execute_notebook


def write_nb(path: Path, cells: list[dict]) -> Path:
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook), encoding="utf-8")
    return path


def read_nb(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def code(source, *, cell_id=None, tags=None, outputs=("stale",), count=99):
    return {
        "cell_type": "code",
        "id": cell_id or f"code-{len(source)}",
        "metadata": {"tags": tags or []},
        "execution_count": count,
        "outputs": list(outputs),
        "source": source,
    }


def outputs_by_type(cell: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for output in cell["outputs"]:
        grouped.setdefault(output["output_type"], []).append(output)
    return grouped


def test_execute_notebook_preserves_structure_outputs_and_source_form(tmp_path):
    source_path = write_nb(
        tmp_path / "input.ipynb",
        [
            {
                "cell_type": "markdown",
                "id": "md-1",
                "metadata": {"custom": "kept"},
                "source": ["# Title\n", "second line"],
                "attachments": {"pic.png": {"image/png": "ZmFrZQ=="}},
            },
            code(["import sys\n", "print('out')\n", "print('err', file=sys.stderr)\n", "6 * 7"], cell_id="c-1"),
            {"cell_type": "raw", "id": "raw-1", "metadata": {}, "source": ["raw text\n", "more"], "attachments": {}},
        ],
    )
    original = read_nb(source_path)
    output_path = tmp_path / "result.ipynb"

    result = execute_notebook(source_path, output_path, cell_timeout=10)

    assert result.status == "ok"
    assert result.executed_cells == 1
    assert output_path.exists()
    assert read_nb(source_path) == original

    output = read_nb(output_path)
    assert output["metadata"] == original["metadata"]
    assert output["cells"][0] == original["cells"][0]
    assert output["cells"][2] == original["cells"][2]
    executed = output["cells"][1]
    assert executed["id"] == "c-1"
    assert executed["source"] == ["import sys\n", "print('out')\n", "print('err', file=sys.stderr)\n", "6 * 7"]
    assert executed["execution_count"] == 1
    grouped = outputs_by_type(executed)
    assert "".join(item["text"] for item in grouped["stream"] if item["name"] == "stdout") == "out\n"
    assert "".join(item["text"] for item in grouped["stream"] if item["name"] == "stderr") == "err\n"
    assert grouped["execute_result"][0]["data"]["text/plain"] == "42"


def test_parameter_defaults_run_before_literal_overrides(tmp_path):
    source_path = write_nb(
        tmp_path / "parameters.ipynb",
        [
            code("alpha = 1\nbeta = 'default'\n", cell_id="params", tags=["parameters"]),
            code("print(alpha, beta, type(beta).__name__)\n(alpha, beta)", cell_id="use"),
        ],
    )
    result = execute_notebook(
        source_path,
        tmp_path / "out.ipynb",
        parameters={"alpha": 7, "beta": "override"},
        cell_timeout=10,
    )

    assert result.status == "ok"
    notebook = read_nb(tmp_path / "out.ipynb")
    cells = notebook["cells"]
    assert cells[0]["id"] == "params"
    assert cells[0]["execution_count"] == 1
    injected = cells[1]
    assert injected["cell_type"] == "code"
    assert injected["metadata"]["tags"] == ["injected-parameters"]
    assert injected["source"] == ["alpha = 7\n", "beta = 'override'"]
    assert injected["execution_count"] == 2
    use_cell = cells[2]
    assert use_cell["execution_count"] == 3
    grouped = outputs_by_type(use_cell)
    assert grouped["stream"][0]["text"] == "7 override str\n"
    assert grouped["execute_result"][0]["data"]["text/plain"] == "(7, 'override')"


def test_parameters_accept_json_string_and_nested_finite_values(tmp_path):
    source_path = write_nb(
        tmp_path / "p.ipynb",
        [code("payload = None\n", cell_id="p", tags=["parameters"]), code("payload", cell_id="use")],
    )
    result = execute_notebook(
        source_path,
        tmp_path / "out.ipynb",
        parameters='{"payload": {"values": [1, true, null, "x"]}}',
        cell_timeout=10,
    )

    assert result.status == "ok"
    notebook = read_nb(tmp_path / "out.ipynb")
    value_cell = notebook["cells"][-1]
    result_output = outputs_by_type(value_cell)["execute_result"][0]
    assert result_output["data"]["text/plain"] == "{'values': [1, True, None, 'x']}"


def test_parameter_validation_is_literal_safe():
    dangerous = {"x": "__import__('os').system('echo unsafe')"}
    encoded = json.loads(json.dumps(dangerous))
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [code("x = 1\n", tags=["parameters"])],
    }
    from labflow_kernel import prepare_notebook

    prepared = prepare_notebook(notebook, encoded)
    injected = prepared.notebook["cells"][1]
    assert injected["source"] == ["x = \"__import__('os').system('echo unsafe')\""]

    for bad_name in ["class", "1abc", "has-dash"]:
        with pytest.raises(NotebookValidationError):
            prepare_notebook(copy.deepcopy(notebook), {bad_name: 1})
    with pytest.raises(NotebookValidationError):
        prepare_notebook(copy.deepcopy(notebook), {"x": float("nan")})
    with pytest.raises(NotebookValidationError):
        prepare_notebook(
            {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}, {"x": 1}
        )


def test_exception_stops_by_default_and_continues_when_requested(tmp_path):
    stopped = write_nb(
        tmp_path / "stop.ipynb",
        [code("1 / 0", cell_id="bad"), code("print('after')", cell_id="after")],
    )
    result = execute_notebook(stopped, tmp_path / "stopped.ipynb", cell_timeout=10)
    notebook = read_nb(tmp_path / "stopped.ipynb")

    assert result.status == "error"
    assert result.ename == "ZeroDivisionError"
    assert result.executed_cells == 1
    error = outputs_by_type(notebook["cells"][0])["error"][0]
    assert error["ename"] == "ZeroDivisionError"
    assert notebook["cells"][1]["outputs"] == []
    assert notebook["cells"][1]["execution_count"] is None

    continued_input = write_nb(
        tmp_path / "continue.ipynb",
        [code("1 / 0", cell_id="bad"), code("print('after')", cell_id="after")],
    )
    result = execute_notebook(
        continued_input, tmp_path / "continued.ipynb", continue_on_error=True, cell_timeout=10
    )
    notebook = read_nb(tmp_path / "continued.ipynb")

    assert result.status == "error"
    assert result.executed_cells == 2
    assert outputs_by_type(notebook["cells"][0])["error"][0]["ename"] == "ZeroDivisionError"
    assert outputs_by_type(notebook["cells"][1])["stream"][0]["text"] == "after\n"


def test_timeout_interrupts_writes_partial_outputs_and_leaves_later_cells_empty(tmp_path):
    source_path = write_nb(
        tmp_path / "slow.ipynb",
        [
            code("import time\nprint('before-wait')\ntime.sleep(10)\nprint('after-wait')", cell_id="slow"),
            code("print('never')", cell_id="never"),
        ],
    )
    result = execute_notebook(source_path, tmp_path / "timed-out.ipynb", cell_timeout=0.5)
    notebook = read_nb(tmp_path / "timed-out.ipynb")

    assert result.status == "timeout"
    assert result.executed_cells == 1
    slow_cell = notebook["cells"][0]
    grouped = outputs_by_type(slow_cell)
    assert grouped["stream"][0]["text"] == "before-wait\n"
    assert grouped["error"][0]["ename"] in {"KeyboardInterrupt", "TimeoutError"}
    assert notebook["cells"][1]["outputs"] == []
    assert notebook["cells"][1]["execution_count"] is None


def test_each_api_call_starts_isolated_kernel(tmp_path):
    source_path = write_nb(
        tmp_path / "isolation.ipynb",
        [code("value = globals().get('value', 0) + 1\nvalue", cell_id="v")],
    )
    first = execute_notebook(source_path, tmp_path / "first.ipynb", cell_timeout=10)
    second = execute_notebook(source_path, tmp_path / "second.ipynb", cell_timeout=10)

    assert first.status == second.status == "ok"
    first_nb = read_nb(tmp_path / "first.ipynb")
    second_nb = read_nb(tmp_path / "second.ipynb")
    first_result = outputs_by_type(first_nb["cells"][0])["execute_result"][0]
    second_result = outputs_by_type(second_nb["cells"][0])["execute_result"][0]
    assert first_result["data"]["text/plain"] == "1"
    assert second_result["data"]["text/plain"] == "1"


@pytest.mark.parametrize(
    ("notebook", "message"),
    [
        ({"nbformat": 5, "nbformat_minor": 0, "metadata": {}, "cells": []}, "nbformat"),
        (
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [code("x = 1", tags=["parameters"]), code("y = 2", tags=["parameters"])],
            },
            "only one",
        ),
        (
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [
                    {"cell_type": "markdown", "metadata": {"tags": ["parameters"]}, "source": ""}
                ],
            },
            "only mark a code cell",
        ),
        (
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [{"cell_type": "code", "metadata": {}, "source": [1]}],
            },
            "list of strings",
        ),
    ],
)
def test_invalid_notebooks(notebook, message):
    from labflow_kernel import prepare_notebook

    with pytest.raises(NotebookValidationError, match=message):
        prepare_notebook(notebook)


def test_late_messages_remain_bucketed_by_parent_id():
    collector = MessageCollector(object())
    collector.begin("cell-a")
    collector.begin("cell-b")

    collector._handle(
        {
            "parent_header": {"msg_id": "cell-a"},
            "msg_type": "stream",
            "content": {"name": "stdout", "text": "late-a"},
        }
    )
    collector._handle(
        {
            "parent_header": {"msg_id": "cell-b"},
            "msg_type": "status",
            "content": {"execution_state": "idle"},
        }
    )
    collector.set_reply(
        "cell-b",
        {"parent_header": {"msg_id": "cell-b"}, "content": {"status": "ok", "execution_count": 2}},
    )

    old = collector.snapshot("cell-a")
    new = collector.snapshot("cell-b")
    assert old.outputs == [{"output_type": "stream", "name": "stdout", "text": "late-a"}]
    assert old.idle is False
    assert new.outputs == []
    assert new.idle is True
    assert new.reply["content"]["status"] == "ok"
