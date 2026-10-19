"""Helpers for Python settings-file CLIs."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def load_settings_file(settings_path: str | Path) -> ModuleType:
    """Load a user-editable Python settings file by filesystem path.

    The file is compiled from source on every call, without reading or writing
    ``__pycache__`` bytecode. Python validates cached bytecode by the source's
    size and modification time in whole seconds, so an edit that keeps the
    size (e.g. ``progress_updates = 4`` -> ``0``) within the same second
    would otherwise load the old settings.
    """
    settings_path = Path(settings_path).expanduser().resolve()
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")
    if settings_path.suffix != ".py":
        raise ValueError(f"Settings file must be a Python file ending in .py: {settings_path}")

    module_name = f"_disorderly_cli_settings_{abs(hash(settings_path))}"
    module = ModuleType(module_name)
    module.__file__ = str(settings_path)
    code = compile(settings_path.read_bytes(), str(settings_path), "exec", dont_inherit=True)
    # Registered while it runs so that, e.g., dataclasses defined in the file work.
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def setting(settings: ModuleType, name: str, default: Any = None, *, required: bool = False) -> Any:
    """Read a named setting, optionally requiring it to be present."""
    if hasattr(settings, name):
        return getattr(settings, name)
    if required:
        raise AttributeError(f"Settings file must define {name!r}.")
    return default
