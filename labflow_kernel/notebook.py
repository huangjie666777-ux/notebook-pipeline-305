"""Notebook planning and safe parameter materialisation.

The notebook reader deliberately depends only on the Python standard library.
Notebooks are treated as nbformat 4 JSON documents; source may be either a
single string or the standard list of source fragments.
"""
from __future__ import annotations

import copy
import json
import keyword
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PARAMETERS_TAG = "parameters"
INJECTED_PARAMETERS_TAG = "injected-parameters"


class NotebookValidationError(ValueError):
    """The input notebook or parameter payload is invalid."""


@dataclass(frozen=True)
class PreparedNotebook:
    """A deep-copied notebook and the code cells selected for execution."""

    notebook: dict[str, Any]
    code_cells: tuple[tuple[int, str], ...]
    parameter_cell_index: int | None


def source_to_text(source: Any) -> str:
    """Return nbformat source as executable text without changing its JSON form."""
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(part, str) for part in source):
        # In nbformat, list elements already include the line terminators.
        return "".join(source)
    raise NotebookValidationError("cell source must be a string or a list of strings")


def _tags(cell: Mapping[str, Any]) -> list[str] | None:
    metadata = cell.get("metadata", {})
    if not isinstance(metadata, dict):
        raise NotebookValidationError("cell metadata must be an object")
    tags = metadata.get("tags")
    if tags is None:
        return []
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise NotebookValidationError("cell metadata.tags must be a list of strings")
    return tags


def _clean_code_cell(cell: dict[str, Any]) -> None:
    source_to_text(cell.get("source", ""))
    cell.setdefault("source", "")
    cell["outputs"] = []
    cell["execution_count"] = None


def _validate_non_code_cell(cell: Mapping[str, Any]) -> None:
    source_to_text(cell.get("source", ""))
    attachments = cell.get("attachments")
    if attachments is not None and not isinstance(attachments, dict):
        raise NotebookValidationError("cell attachments must be an object")


def normalize_parameters(parameters: Mapping[str, Any] | str | None) -> dict[str, str]:
    """Validate JSON parameters and return JSON literals safe for assignment.

    Names are validated as non-keyword Python identifiers.  ``json.dumps``
    performs the value-side encoding with non-finite numbers rejected, so a
    string such as ``"__import__('os')"`` remains a quoted string literal.
    """
    if parameters is None:
        return {}
    if isinstance(parameters, str):
        try:
            parameters = json.loads(parameters)
        except json.JSONDecodeError as exc:
            raise NotebookValidationError(f"parameters must be valid JSON: {exc}") from exc
    if not isinstance(parameters, dict):
        raise NotebookValidationError("parameters must be a JSON object")
    if not parameters:
        return {}

    materialized: dict[str, str] = {}
    for name, value in parameters.items():
        if not isinstance(name, str):
            raise NotebookValidationError("parameter names must be strings")
        if not name.isidentifier() or keyword.iskeyword(name):
            raise NotebookValidationError(f"{name!r} is not a valid non-keyword Python identifier")
        try:
            # Round-trip through JSON first, then render the resulting finite
            # JSON value as a Python literal.  String contents remain quoted.
            parsed = json.loads(json.dumps(value, allow_nan=False))
            materialized[name] = _python_literal(parsed)
        except (TypeError, ValueError) as exc:
            raise NotebookValidationError(f"parameter {name!r} must contain finite JSON data") from exc
    return materialized


def _python_literal(value: Any) -> str:
    """Render a parsed finite JSON value as an executable Python literal."""
    if value is None:
        return "None"
    if value is True:
        return "True"
    if value is False:
        return "False"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_python_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                f"{repr(key)}: {_python_literal(item)}" for key, item in value.items()
            )
            + "}"
        )
    # The JSON round-trip above should make this unreachable.
    raise NotebookValidationError("parameter values must be finite JSON data")


def _injected_parameter_cell(assignments: Mapping[str, str]) -> dict[str, Any]:
    lines = [f"{name} = {literal}\n" for name, literal in assignments.items()]
    # Avoid a final newline difference affecting the final source fragment only.
    if lines:
        lines[-1] = lines[-1].removesuffix("\n")
    cell_id = "injected-parameters-" + uuid.uuid4().hex[:12]
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {"tags": [INJECTED_PARAMETERS_TAG]},
        "execution_count": None,
        "outputs": [],
        "source": lines,
    }


def prepare_notebook(
    notebook: Mapping[str, Any], parameters: Mapping[str, Any] | str | None = None
) -> PreparedNotebook:
    """Validate, copy and plan execution for an nbformat 4 notebook."""
    if not isinstance(notebook, dict):
        raise NotebookValidationError("notebook must be a JSON object")
    if notebook.get("nbformat") != 4:
        raise NotebookValidationError("only nbformat 4 notebooks are supported")
    minor = notebook.get("nbformat_minor", 0)
    if not isinstance(minor, int) or isinstance(minor, bool) or minor < 0:
        raise NotebookValidationError("nbformat_minor must be a non-negative integer")
    metadata = notebook.get("metadata", {})
    if not isinstance(metadata, dict):
        raise NotebookValidationError("notebook metadata must be an object")
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise NotebookValidationError("notebook.cells must be a list")

    prepared = copy.deepcopy(notebook)
    prepared_cells = prepared["cells"]
    parameter_indices: list[int] = []

    for index, cell in enumerate(prepared_cells):
        if not isinstance(cell, dict):
            raise NotebookValidationError("each cell must be an object")
        cell_type = cell.get("cell_type")
        tags = _tags(cell) or []
        if PARAMETERS_TAG in tags:
            parameter_indices.append(index)
            if cell_type != "code":
                raise NotebookValidationError("the parameters tag may only mark a code cell")
        if cell_type == "code":
            _clean_code_cell(cell)
        elif cell_type in ("markdown", "raw"):
            _validate_non_code_cell(cell)
            cell.setdefault("source", "")
        else:
            raise NotebookValidationError(f"unsupported cell type: {cell_type!r}")

    if len(parameter_indices) > 1:
        raise NotebookValidationError("only one code cell may be tagged 'parameters'")
    parameter_index = parameter_indices[0] if parameter_indices else None

    assignments = normalize_parameters(parameters)
    if assignments and parameter_index is None:
        raise NotebookValidationError("non-empty parameters require a code cell tagged 'parameters'")

    if assignments:
        injected = _injected_parameter_cell(assignments)
        prepared_cells.insert(parameter_index + 1, injected)

    code_cells = tuple(
        (index, source_to_text(cell.get("source", "")))
        for index, cell in enumerate(prepared_cells)
        if cell.get("cell_type") == "code"
    )
    return PreparedNotebook(prepared, code_cells, parameter_index)


def load_notebook(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load an nbformat JSON document from disk."""
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            notebook = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise NotebookValidationError(f"cannot read notebook {path}: {exc}") from exc
    return notebook


def write_notebook_atomic(notebook: Mapping[str, Any], path: str | os.PathLike[str]) -> None:
    """Write a notebook with a temp-file-and-rename commit."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(notebook, handle, ensure_ascii=False, indent=1)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        raise
