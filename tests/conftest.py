"""Shared pytest fixtures."""

from __future__ import annotations

import os

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def repo_root() -> str:
    return REPO_ROOT


@pytest.fixture(scope="session")
def toy_reference_pdb(repo_root: str) -> str:
    """Clean 1ZNF reference (protein chain A + zinc) shipped in ``data/toy``."""
    path = os.path.join(repo_root, "data", "toy", "1ZNF_clean_ref.pdb")
    if not os.path.exists(path):
        pytest.skip("toy reference structure not available")
    return path
