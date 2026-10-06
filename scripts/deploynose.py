# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open the local deploynose app; no server or robot connection is required."""

import shutil
import subprocess
import webbrowser
from pathlib import Path


def main():
    path = Path(__file__).resolve().parents[1] / "g1_deploy" / "deploynose" / "index.html"
    print(f"deploynose: {path}")
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome:
        subprocess.Popen([chrome, "--app=" + path.as_uri()], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
