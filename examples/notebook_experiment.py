"""Demonstrate the notebook experiment API without nbclient or papermill."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from labflow_kernel import execute_notebook


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="runbook-notebook-demo-"))
    input_path = workdir / "experiment.ipynb"
    output_path = workdir / "experiment-result.ipynb"

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [
            {
                "cell_type": "markdown",
                "id": "intro",
                "metadata": {},
                "source": ["# Parameter sweep\n", "Defaults are in the tagged code cell."],
            },
            {
                "cell_type": "code",
                "id": "parameters",
                "metadata": {"tags": ["parameters"]},
                "execution_count": None,
                "outputs": [],
                "source": ["learning_rate = 0.1\n", "layers = [8]\n", "tag = 'default'"],
            },
            {
                "cell_type": "code",
                "id": "train",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": ["config = {'lr': learning_rate, 'layers': layers, 'tag': tag}\n", "print(config)\n", "config"],
            },
        ],
    }
    input_path.write_text(json.dumps(notebook), encoding="utf-8")

    result = execute_notebook(
        input_path,
        output_path,
        parameters={"learning_rate": 0.01, "layers": [64, 32], "tag": "baseline"},
        cell_timeout=10,
    )

    produced = json.loads(output_path.read_text(encoding="utf-8"))
    final_cell = produced["cells"][-1]
    print(f"input : {input_path}")
    print(f"output: {output_path}")
    print(f"result: status={result.status} executed_cells={result.executed_cells}")
    print("injected source:", produced["cells"][2]["source"])
    print("final outputs:", json.dumps(final_cell["outputs"], ensure_ascii=False))


if __name__ == "__main__":
    main()
