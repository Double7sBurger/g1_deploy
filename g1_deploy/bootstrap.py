# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Make ``unitree_sdk2py`` importable before anything touches CycloneDDS.

The cyclonedds Python bindings ``dlopen`` ``libddsc`` at import time, and the wheel in this
environment was built against a local CycloneDDS install that is not on the default loader path.
``LD_LIBRARY_PATH`` is read by the loader only at process start, so it cannot be fixed after the
fact -- the process has to re-exec itself.

Call :func:`ensure_cyclonedds` as the first statement of any entry point that imports
``unitree_sdk2py``, before that import.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

CYCLONEDDS_LIB = Path.home() / "workspace/cyclonedds/install/lib"
"""Local CycloneDDS install the ``cyclonedds`` wheel was built against."""


def ensure_cyclonedds() -> None:
    """Re-exec this process with ``LD_LIBRARY_PATH`` set, if it is not already.

    Does nothing when the directory is absent (a system-wide CycloneDDS is then assumed) or when the
    path is already present, so it is safe to call unconditionally and idempotent across the re-exec.
    """
    if not CYCLONEDDS_LIB.is_dir():
        return
    if str(CYCLONEDDS_LIB) in os.environ.get("LD_LIBRARY_PATH", ""):
        return
    os.environ["LD_LIBRARY_PATH"] = f"{CYCLONEDDS_LIB}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.execv(sys.executable, [sys.executable] + sys.argv)
