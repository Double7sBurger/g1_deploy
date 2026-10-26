# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analyze a deployment recording and open it in deploynose, already loaded.

The app reads a ``.deploynose`` bundle the browser hands it, and a ``file://`` page may not fetch
its own neighbours. So this serves the repository over loopback and passes the bundle as ``?file=``.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import threading
import webbrowser
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = "g1_deploy/deploynose/index.html"


def newest_recording(root: Path) -> Path:
    """The most recently written recording directory under ``root``."""
    candidates = [path.parent for path in root.glob("*/recording.sqlite3")]
    if not candidates:
        raise SystemExit(f"no recordings under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def serve(port: int) -> int:
    """Start a background HTTP server rooted at the repository and return its port."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(REPO))
    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.server_address[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording", nargs="?", default=None,
                        help="Recording directory; defaults to the newest under --root.")
    parser.add_argument("--root", default=str(REPO / "recordings"), help="Where to look for the newest run.")
    parser.add_argument("--port", type=int, default=0, help="Loopback port; 0 picks a free one.")
    parser.add_argument("--no_plots", action="store_true", help="Skip the static PNGs.")
    parser.add_argument("--reuse", action="store_true", help="Open an existing bundle without re-analyzing.")
    parser.add_argument("--window", choices=("policy", "all"), default="policy",
                        help="Export only what the policy was running for, or the whole recording.")
    args = parser.parse_args()

    recording = Path(args.recording).resolve() if args.recording else newest_recording(Path(args.root))
    bundle = recording / "analysis" / (recording.name + ".deploynose")
    if args.reuse and bundle.exists():
        print(f"[reuse] {bundle}")
    else:
        from g1_deploy.bootstrap import ensure_cyclonedds

        ensure_cyclonedds()
        from g1_deploy.analysis import save_report

        _, summary = save_report(str(recording), None, plots=not args.no_plots, window=args.window)
        print(f"[data] {summary['samples']} state samples, {summary['recorded_seconds']:.1f}s;"
              f" clean shutdown={summary['clean_shutdown']}, dropped={summary['dropped']}")
    if not bundle.exists():
        raise SystemExit(f"no bundle at {bundle}")

    port = serve(args.port)
    url = f"http://127.0.0.1:{port}/{APP}?file=/{bundle.relative_to(REPO).as_posix()}"
    print(f"[open] {url}")
    webbrowser.open(url)
    print("[..] serving; Ctrl-C to stop")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\n[ok] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
