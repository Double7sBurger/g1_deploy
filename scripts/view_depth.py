# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Watch the depth stream the policy is reading, as text. Works over SSH, needs no display.

Points at the same UDP port the control loop listens on, so it shows exactly what the policy sees --
whether the frames come from ``run_sim_loop.py --depth`` or from ``depth_publisher.py`` on the
robot's PC2. The renderer's viewer shows the robot; this shows the *observation*, which is the thing
that can be silently wrong.

Run it beside the control loop rather than instead of it: two UDP sockets on the same port both
receive, because the publisher sends to one address and the OS delivers to every bound socket only
when they ask to share the port. They do not by default, so start this **before** the control loop,
or give it its own port with a second publisher. The ``--port`` default matches the control loop's.

What to look for:

* a gradient from far at the top to near at the bottom -- that is the ground under a camera pitched
  down. A flat image means the camera is aimed at a wall, or not pitched at all.
* the near/far numbers against the geometry: a camera 1.26 m up and pitched 47.6 degrees sees the
  image centre at about 1.7 m on flat ground.
* the invalid fraction. A real D435i indoors reads a few percent; tens of percent means the scene is
  out of range, too reflective, or the sensor is blinded.

Usage::

    python scripts/view_depth.py --port 5601
    python scripts/view_depth.py --port 5601 --once     # one frame, then exit
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from g1_deploy.depth_link import DEFAULT_PORT, DepthReceiver

RAMP = " .:-=+*#%@"
"""Near is bright. A blank cell is a pixel with no return."""


def render(frame: np.ndarray, max_range: float, rows: int = 19) -> str:
    """Draw one depth frame as text.

    Args:
        frame: Depth in metres, invalid as ``<= 0``.
        max_range: Range the policy clips at; sets the brightness scale so the picture matches what
            the network is normalised against rather than the frame's own extremes.
        rows: Output rows; the frame is subsampled to fit.

    Returns:
        A newline-joined block.
    """
    step = max(1, frame.shape[0] // rows)
    out = []
    for row in frame[::step]:
        out.append("".join(
            " " if v <= 0 else RAMP[int(np.clip(1.0 - v / max_range, 0.0, 0.999) * len(RAMP))]
            for v in row
        ))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--width", type=int, default=64, help="Frame width the publisher sends.")
    ap.add_argument("--height", type=int, default=38, help="Frame height the publisher sends.")
    ap.add_argument("--max_range", type=float, default=3.0, help="Clip range; contract depth_max_range_m.")
    ap.add_argument("--hz", type=float, default=4.0, help="Redraw rate.")
    ap.add_argument("--once", action="store_true", help="Print one frame and exit.")
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()

    rx = DepthReceiver((args.height, args.width), port=args.port)
    print(f"[..] listening on UDP {args.port} for {args.width}x{args.height} frames ...")
    try:
        rx.wait_for_frame(timeout=args.timeout)
    except TimeoutError as exc:
        print(f"[FAIL] {exc}")
        print("       Is the publisher running, and pointed at this machine? If the control loop is")
        print("       already bound to this port, start this first or use a separate port.")
        rx.close()
        return 1

    try:
        while True:
            frame, age = rx.latest()
            valid = frame[frame > 0]
            head = (
                f"seq-received {rx.received}  dropped {rx.dropped}  age {age * 1000:5.1f} ms"
                f"  |  valid {100.0 * valid.size / frame.size:5.1f}%"
            )
            if valid.size:
                head += (f"  near {valid.min():.2f}  far {valid.max():.2f}  mean {valid.mean():.2f} m"
                         f"  |  centre row {np.mean(frame[frame.shape[0] // 2][frame[frame.shape[0] // 2] > 0]):.2f} m")
            print("\033[2J\033[H" + head + "\n" + render(frame, args.max_range), flush=True)
            if args.once:
                break
            time.sleep(1.0 / max(0.5, args.hz))
    except KeyboardInterrupt:
        print("\n[..] interrupted")
    finally:
        rx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
