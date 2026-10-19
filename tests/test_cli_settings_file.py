import os
import sys

from disorderly.cli._settings_file import load_settings_file


def test_load_settings_file_sees_same_size_edit_within_one_second(tmp_path, monkeypatch):
    # Regression: settings were loaded through the import system, which reuses
    # __pycache__ bytecode when the source's size and whole-second mtime are
    # unchanged, so this edit was silently ignored.
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    path = tmp_path / "settings.py"
    path.write_text("progress_updates = 4\n")
    os.utime(path, (1_700_000_000, 1_700_000_000))
    assert load_settings_file(path).progress_updates == 4

    path.write_text("progress_updates = 0\n")
    os.utime(path, (1_700_000_000, 1_700_000_000))
    assert load_settings_file(path).progress_updates == 0
    assert not (tmp_path / "__pycache__").exists()


def test_load_settings_file_sets_file_and_supports_dataclasses(tmp_path):
    path = tmp_path / "settings.py"
    path.write_text(
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class Options:\n"
        "    n: int = 3\n\n"
        "options = Options()\n"
    )
    settings = load_settings_file(path)
    assert settings.__file__ == str(path.resolve())
    assert settings.options.n == 3
