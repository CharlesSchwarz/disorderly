"""Tests for package-level metadata."""

from importlib.metadata import version

import disorder_mc


def test_version_matches_installed_metadata():
    assert disorder_mc.__version__ == version("disorder-mc")
