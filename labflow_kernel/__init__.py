"""labflow_kernel — 纯 Python 实现的 Jupyter 协议 5.3 后端内核。

仅依赖 pyzmq 与 jupyter_client(复用其 Session 编解码与 HMAC 校验),
不启动、不代理 ipykernel。
"""

__version__ = "0.1.0"

from .messages import CellMessages, MessageCollector
from .notebook import (
    INJECTED_PARAMETERS_TAG,
    PARAMETERS_TAG,
    NotebookValidationError,
    PreparedNotebook,
    load_notebook,
    normalize_parameters,
    prepare_notebook,
    write_notebook_atomic,
)
from .runner import (
    DEFAULT_CELL_TIMEOUT,
    KernelStartupError,
    NotebookExecutionResult,
    NotebookExecutionError,
    NotebookTimeoutError,
    execute_notebook,
    execute_prepared,
)

__all__ = [
    "DEFAULT_CELL_TIMEOUT",
    "CellMessages",
    "INJECTED_PARAMETERS_TAG",
    "KernelStartupError",
    "MessageCollector",
    "NotebookExecutionError",
    "NotebookExecutionResult",
    "NotebookTimeoutError",
    "NotebookValidationError",
    "PARAMETERS_TAG",
    "PreparedNotebook",
    "execute_notebook",
    "execute_prepared",
    "load_notebook",
    "normalize_parameters",
    "prepare_notebook",
    "write_notebook_atomic",
]
