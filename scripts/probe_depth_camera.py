# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only probe: is there a depth camera on this machine, and does it match the contract?

Runs on the robot's PC2 (the Jetson the camera is wired to), not on the laptop. Answers, without
touching the robot's motors:

* is a RealSense present on USB at all
* can it actually stream depth, and at what native resolution and rate
* what is its real field of view, from the stream's own intrinsics
* how does that compare with what the policy was trained against

The last one is the point. ``policies/*/contract.json`` records the camera Isaac Lab rendered with:
87.0 x 58.8 degrees at 64x38, which is the D435's depth FOV downsampled. A camera with a different
FOV produces a correctly-shaped frame whose *contents* mean something else, and nothing downstream
can tell -- the array is 38x64 either way.

Usage, on PC2::

    python3 probe_depth_camera.py --contract contract.json   # if you copied the contract over
    python3 probe_depth_camera.py                            # bare probe, no contract needed
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys


def usb_scan() -> None:
    """List USB devices that look like a depth camera. Needs nothing installed."""
    print("=== USB ===")
    try:
        out = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=10).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  lsusb unavailable")
        return
    hits = [ln for ln in out.splitlines() if any(k in ln.lower() for k in ("intel", "realsense", "8086"))]
    if hits:
        for ln in hits:
            print(f"  {ln}")
    else:
        print("  no Intel/RealSense device on USB")
        print("  (a D435i shows up as 'Intel Corp. RealSense', vendor 8086)")


def stream_probe(seconds: float) -> dict | None:
    """Open the depth stream and report what it actually delivers.

    Returns:
        Measured properties, or ``None`` if pyrealsense2 is missing or no device streams.
    """
    print("\n=== depth stream ===")
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("  pyrealsense2 not installed")
        print("  on Jetson/Ubuntu:  pip3 install pyrealsense2")
        return None

    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        print("  pyrealsense2 is installed but sees no device")
        return None
    for dev in devices:
        print(f"  device: {dev.get_info(rs.camera_info.name)}"
              f"  serial {dev.get_info(rs.camera_info.serial_number)}"
              f"  fw {dev.get_info(rs.camera_info.firmware_version)}")

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth)  # let the driver pick its default depth profile
    try:
        profile = pipe.start(cfg)
    except RuntimeError as exc:
        print(f"  failed to start the depth stream: {exc}")
        return None

    try:
        vs = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intr = vs.get_intrinsics()
        hfov = math.degrees(2 * math.atan2(intr.width, 2 * intr.fx))
        vfov = math.degrees(2 * math.atan2(intr.height, 2 * intr.fy))
        scale = profile.get_device().first_depth_sensor().get_depth_scale()

        import time

        n, t0 = 0, time.monotonic()
        sample = None
        while time.monotonic() - t0 < seconds:
            frames = pipe.wait_for_frames(timeout_ms=2000)
            d = frames.get_depth_frame()
            if not d:
                continue
            n += 1
            if sample is None:
                import numpy as np

                a = np.asanyarray(d.get_data()).astype("float32") * scale
                valid = a[(a > 0) & np.isfinite(a)]
                sample = (float(valid.min()), float(valid.max()), float(valid.mean()),
                          100.0 * valid.size / a.size)
        hz = n / (time.monotonic() - t0)
    finally:
        pipe.stop()

    print(f"  native      : {intr.width} x {intr.height} @ {hz:.0f} Hz measured")
    print(f"  fov         : {hfov:.1f} x {vfov:.1f} deg  (from the stream's own intrinsics)")
    print(f"  depth scale : {scale} m per unit")
    if sample:
        lo, hi, mean, pct = sample
        print(f"  first frame : {pct:.0f}% valid pixels, range {lo:.2f}-{hi:.2f} m, mean {mean:.2f} m")
    return {"width": intr.width, "height": intr.height, "hfov": hfov, "vfov": vfov, "hz": hz}


def compare(measured: dict, contract_path: str) -> None:
    """Hold the measured camera up against what the policy was trained on."""
    with open(contract_path) as handle:
        c = json.load(handle)
    cam = c["camera"]
    t_h = math.degrees(2 * math.atan(cam["horizontal_aperture_mm"] / (2 * cam["focal_length_mm"])))
    t_v = math.degrees(2 * math.atan(
        cam["horizontal_aperture_mm"] * cam["height"] / cam["width"] / (2 * cam["focal_length_mm"])))
    print("\n=== trained vs measured ===")
    print(f"  {'':<12s} {'trained':>18s} {'measured':>18s}")
    print(f"  {'fov h':<12s} {t_h:>17.1f}° {measured['hfov']:>17.1f}°")
    print(f"  {'fov v':<12s} {t_v:>17.1f}° {measured['vfov']:>17.1f}°")
    trained_res = f"{cam['width']}x{cam['height']}"
    measured_res = f"{measured['width']}x{measured['height']}"
    print(f"  {'resolution':<12s} {trained_res:>18s} {measured_res:>18s}")
    dh, dv = abs(t_h - measured["hfov"]), abs(t_v - measured["vfov"])
    if dh < 5 and dv < 5:
        print(f"\n  [ok] field of view agrees to {dh:.1f}/{dv:.1f} deg -- downsampling is enough")
    else:
        print(f"\n  [!!] field of view differs by {dh:.1f}/{dv:.1f} deg.")
        print("       A frame of the right SHAPE but the wrong FOV is the failure that does not")
        print("       raise: every array is 38x64 either way, and the policy silently reads a")
        print("       different world than the one it was trained on. Crop before downsampling,")
        print("       or retrain. Do not just resize.")
    pitch = math.degrees(2 * math.acos(cam["offset_rot_wxyz"][0]))
    print(f"\n  mounting (trained): {cam['offset_pos_m']} m on torso_link,"
          f" pitched {pitch:.0f} deg about Y, {cam['convention']} convention")
    print("  Check the physical camera sits there and points the same way -- the policy reads the")
    print("  ground at a fixed angle and cannot tell a tilted camera from sloped terrain.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--contract", default=None, help="contract.json to compare against, if you copied it over.")
    ap.add_argument("--seconds", type=float, default=3.0, help="How long to measure the frame rate.")
    args = ap.parse_args()

    usb_scan()
    measured = stream_probe(args.seconds)
    if measured is None:
        print("\n[FAIL] no depth stream. Nothing downstream can work until this does.")
        return 1
    if args.contract:
        compare(measured, args.contract)
    print("\n[ok] probe complete; nothing was commanded and no motor was touched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
