"""Thread-local warning filters used by threaded sampling workflows."""

from __future__ import annotations

from contextlib import contextmanager
import re
import threading
from typing import Any, Iterator
import warnings

_LOGM_WARNING_RE = re.compile(
    r"logm result may be inaccurate,\s*approximate err\s*=\s*([0-9.eE+-]+)"
)
_WARNING_THREAD_LOCAL = threading.local()
_WARNING_HOOK_LOCK = threading.Lock()
_DEFAULT_SHOWWARNING = warnings.showwarning
_ORIGINAL_SHOWWARNING: Any | None = None


def _logm_warning_error(message: str) -> float | None:
    match = _LOGM_WARNING_RE.search(message)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def should_suppress_logm_warning(
    message: object,
    category: type[Warning],
    threshold: float | None,
) -> bool:
    """Return true when a SciPy logm warning is below the configured tolerance."""
    if threshold is None or not issubclass(category, RuntimeWarning):
        return False
    err = _logm_warning_error(str(message))
    return err is not None and err <= threshold


def _showwarning_with_thread_filters(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: Any | None = None,
    line: str | None = None,
) -> None:
    if getattr(_WARNING_THREAD_LOCAL, "suppress_future_warnings", False) and issubclass(category, FutureWarning):
        return

    threshold = getattr(_WARNING_THREAD_LOCAL, "logm_warning_tol", None)
    if should_suppress_logm_warning(message, category, threshold):
        return

    showwarning = _ORIGINAL_SHOWWARNING or _DEFAULT_SHOWWARNING
    showwarning(message, category, filename, lineno, file=file, line=line)


def install_thread_warning_filters() -> None:
    """Install the process warning hook used by thread-local filters."""
    global _ORIGINAL_SHOWWARNING
    with _WARNING_HOOK_LOCK:
        if warnings.showwarning is not _showwarning_with_thread_filters:
            _ORIGINAL_SHOWWARNING = warnings.showwarning
            warnings.showwarning = _showwarning_with_thread_filters


@contextmanager
def warning_filter_scope(
    *,
    suppress_future_warnings: bool | None = None,
    logm_warning_tol: float | None = None,
) -> Iterator[None]:
    """Apply warning filters only to warnings emitted by the current thread."""
    install_thread_warning_filters()
    previous_suppress_future = getattr(_WARNING_THREAD_LOCAL, "suppress_future_warnings", None)
    previous_logm_tol = getattr(_WARNING_THREAD_LOCAL, "logm_warning_tol", None)

    if suppress_future_warnings is not None:
        _WARNING_THREAD_LOCAL.suppress_future_warnings = suppress_future_warnings
    if logm_warning_tol is not None:
        _WARNING_THREAD_LOCAL.logm_warning_tol = logm_warning_tol

    try:
        yield
    finally:
        if previous_suppress_future is None:
            if hasattr(_WARNING_THREAD_LOCAL, "suppress_future_warnings"):
                delattr(_WARNING_THREAD_LOCAL, "suppress_future_warnings")
        else:
            _WARNING_THREAD_LOCAL.suppress_future_warnings = previous_suppress_future

        if previous_logm_tol is None:
            if hasattr(_WARNING_THREAD_LOCAL, "logm_warning_tol"):
                delattr(_WARNING_THREAD_LOCAL, "logm_warning_tol")
        else:
            _WARNING_THREAD_LOCAL.logm_warning_tol = previous_logm_tol
