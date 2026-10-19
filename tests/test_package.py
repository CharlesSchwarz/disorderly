"""Tests for package-level metadata."""

from importlib.metadata import version

import disorderly


def test_version_matches_installed_metadata():
    assert disorderly.__version__ == version("disorderly")
