# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish the G1's chest depth camera as the 64x38 frames the policy expects. Runs on PC2.

The camera is on PC2's USB bus and ``pyrealsense2`` has no macOS build, so this is the half that
must live on the robot regardless of where the control loop runs. It does the whole camera-side
transform -- crop to the trained field of view, downsample, drop to metres -- and sends 9.7 kB per
frame, because the crop needs the camera's own intrinsics and those are only queryable here.

Measured on this robot: D435i, USB 3.2, native 848x480, real field of view **89.6 x 58.7 deg**
against the 87.0 x 58.8 the policy trained on. :func:`~g1_deploy.depth_link.crop_to_fov` removes the
2.6 degrees of excess width, leaving 86.97 x 58.70 -- a residual of 0.03 deg, against 3% of silent
horizontal compression if the frame were merely resized.

**Run the camera faster than the control loop.** At the driver's default 30 Hz a 50 Hz control loop
sees the same frame twice roughly two steps in five, and the depth history then spans more wall time
than the 3 x 20 ms it did in training. At 90 Hz every control step gets a frame no more than 11 ms
old and the history spacing is the control period, which is what training had.

Usage, on PC2::

    python3 depth_publisher.py --host 192.168.123.222        # the laptop
    python3 depth_publisher.py --host 192.168.123.222 --preview   # plus an ASCII view
"""

from __future__ import annotations

import argparse
import math
import socket
import sys
import time

import numpy as np

# Import the wire format from wherever it is: the repo layout when run from a checkout, or a plain
# sibling file when only these two files were copied to the robot. One definition either way -- a
# second copy of the packing code on the robot is a copy that will drift from the receiver's.
_HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))
try:
    from g1_deploy.depth_link import DEFAULT_PORT, crop_to_fov, downsample_masked, pack_frame
except ImportError:
    from depth_link import DEFAULT_PORT, crop_to_fov, downsample_masked, pack_frame  # noqa: F401

TRAINED_HFOV = 87.0
"""Horizontal field of view the policy trained on [deg]; ``contract.json`` camera block."""

TRAINED_VFOV = 58.8
"""Vertical field of view the policy trained on [deg]."""


def ascii_preview(frame: np.ndarray, max_range: float) -> str:
    """Render a depth frame as text, for checking aim over SSH without a display."""
    ramp = " .:-=+*#%@"
    rows = []
    for row in frame[:: max(1, frame.shape[0] // 16)]:
        cells = []
        for value in row[:: max(1, frame.shape[1] // 48)]:
            if value <= 0.0:
                cells.append(" ")
            else:
                idx = int(np.clip(1.0 - value / max_range, 0, 0.999) * len(ramp))
                cells.append(ramp[idx])
        rows.append("".join(cells))
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", required=True, help="Where the control loop runs, e.g. 192.168.123.222.")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--out_width", type=int, default=64, help="Policy input width; contract depth_shape[2].")
    ap.add_argument("--out_height", type=int, default=38, help="Policy input height; contract depth_shape[1].")
    ap.add_argument("--width", type=int, default=848, help="Native depth width to request.")
    ap.add_argument("--height", type=int, default=480, help="Native depth height to request.")
    ap.add_argument(
        "--fps",
        type=int,
        default=90,
        help="Native frame rate. Must exceed the 50 Hz control rate or the policy sees repeats;"
        " 848x480 supports 90 on USB3.",
    )
    ap.add_argument(
        "--no_crop",
        action="store_true",
        help="Skip the field-of-view crop. Only for measuring what the crop is worth -- leaving it"
        " off ships a frame whose contents are 3%% narrower than the policy believes.",
    )
    ap.add_argument("--preview", action="store_true", help="Print an ASCII depth view once a second.")
    ap.add_argument("--report_s", type=float, default=5.0, help="Seconds between throughput lines.")
    args = ap.parse_args()

    import pyrealsense2 as rs

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipe.start(cfg)
    sensor = profile.get_device().first_depth_sensor()
    scale = sensor.get_depth_scale()

    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    hfov = math.degrees(2 * math.atan2(intr.width, 2 * intr.fx))
    vfov = math.degrees(2 * math.atan2(intr.height, 2 * intr.fy))
    if args.no_crop:
        x0, y0, cw, ch = 0, 0, intr.width, intr.height
    else:
        x0, y0, cw, ch = crop_to_fov(intr.width, intr.height, hfov, vfov, TRAINED_HFOV, TRAINED_VFOV)
    kept_h = math.degrees(2 * math.atan2(cw, 2 * intr.fx))
    kept_v = math.degrees(2 * math.atan2(ch, 2 * intr.fy))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
    dest = (args.host, args.port)

    print(
        f"[..] {intr.width}x{intr.height} @ {args.fps} Hz, fov {hfov:.1f} x {vfov:.1f} deg\n"
        f"[..] crop -> {cw}x{ch} at ({x0},{y0}), fov {kept_h:.2f} x {kept_v:.2f} deg"
        f"  (trained {TRAINED_HFOV} x {TRAINED_VFOV})\n"
        f"[..] downsample -> {args.out_width}x{args.out_height}, sending to {args.host}:{args.port}"
    )
    if abs(kept_h - TRAINED_HFOV) > 1.0 or abs(kept_v - TRAINED_VFOV) > 1.0:
        print("[!!] field of view still differs by more than a degree after cropping; the policy")
        print("     will read terrain at the wrong scale and nothing downstream can detect it")

    seq, sent, t_report = 0, 0, time.monotonic()
    t_preview = t_report
    try:
        while True:
            frames = pipe.wait_for_frames(timeout_ms=2000)
            depth = frames.get_depth_frame()
            if not depth:
                continue
            stamp_ns = int(depth.get_timestamp() * 1e6)  # RealSense reports milliseconds
            raw = np.asanyarray(depth.get_data())
            metres = raw[y0 : y0 + ch, x0 : x0 + cw].astype(np.float32) * scale
            small = downsample_masked(metres, args.out_height, args.out_width)
            sock.sendto(pack_frame(seq, small, stamp_ns), dest)
            seq += 1
            sent += 1

            now = time.monotonic()
            if args.preview and now - t_preview >= 1.0:
                t_preview = now
                valid = small[small > 0]
                print(f"\n[preview] {100.0 * valid.size / small.size:.0f}% valid"
                      f"  {valid.min():.2f}-{valid.max():.2f} m" if valid.size else "\n[preview] all invalid")
                print(ascii_preview(small, 3.0))
            if now - t_report >= args.report_s:
                print(f"[ok] {sent / (now - t_report):.0f} Hz out, {seq} frames total")
                sent, t_report = 0, now
    except KeyboardInterrupt:
        print("\n[..] interrupted")
    finally:
        pipe.stop()
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
