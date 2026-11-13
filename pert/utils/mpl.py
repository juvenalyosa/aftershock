"""
Single entry point for matplotlib.

Import ``plt`` from here instead of from matplotlib directly so that the cache
directory is configured before matplotlib is first imported (important on
clusters where ``$HOME`` is read-only or slow).
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "pert-matplotlib"))

import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["plt"]
