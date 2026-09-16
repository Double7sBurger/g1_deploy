# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure where the depth camera actually points, from the frames it sends.

``contract.json`` records the camera Isaac Lab *rendered* with. Whether the physical D435i is
mounted the same way is a separate question, and one that nothing downstream can answer: a camera
off by a few degrees produces a perfectly ordinary depth image, and the policy reads the ground at
the wrong distance without any array changing shape.

On flat ground the geometry is fully determined. For a camera at height ``h`` pitched ``theta``
below horizontal with vertical field of view ``phi``, image row ``r`` -- taken as a fraction of the
frame measured from the centre, positive downward -- looks along a ray ``theta + phi * r`` below
horizontal, and the image-plane depth it reports is::

    d(r) = h / sin(theta + phi * r) * cos(phi * r)

The ``cos`` factor is what makes this image-plane depth rather than ray distance, which is the
quantity both MuJoCo and Isaac Lab's ``distance_to_image_plane`` return. Fitting ``h``, ``theta``
and ``phi`` to a measured row profile therefore recovers the real mounting, independently of what
any configuration file claims.

.. attention::
    **The ground must be flat and clear.** The fit assumes every row sees the same plane, so a
    person standing in frame, a wall within 3 m, or a slope all bias it. The reported residual is
    the check: a good fit on flat ground lands under about 2 cm, and anything larger means the fit
    is describing the room rather than the camera.

Usage::

    python scripts/fit_camera_pose.py --port 5601                  # live, from the publisher
    python scripts/fit_camera_pose.py --npz /tmp/real_frames.npz   # from recorded frames
    python scripts/fit_camera_pose.py --selftest                   # check the fit against MuJoCo
"""

from __future__ import annotations

import argparse
import math

import numpy as np

CONTRACT_PITCH_DEG = 47.6
"""Downward pitch the deployment renders with. Unitree's own drawing gives 42.4 degrees from
vertical, which is the same angle."""

CONTRACT_VFOV_DEG = 58.8
"""Vertical field of view from the contract's focal length and aperture. Unitree's drawing says
55.2, and the D435i's own intrinsics measured 58.7 -- worth knowing which one the robot agrees
with."""


def predict(rows: np.ndarray, height: float, pitch: float, vfov: float) -> np.ndarray:
    """Image-plane depth of flat ground for each normalised row.

    Args:
        rows: Row position as a fraction of the frame from the centre, positive downward.
        height: Camera height above the ground [m].
        pitch: Downward pitch of the optical axis [rad].
        vfov: Vertical field of view [rad].

    Returns:
        Predicted depth [m]; ``inf`` where the ray points at or above the horizon.
    """
    angle = pitch + vfov * rows
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(angle > 1e-3, height / np.sin(np.clip(angle, 1e-3, None)) * np.cos(vfov * rows),
                       np.inf)
    return out


def fit(profile: np.ndarray, max_range: float, vfov_deg: float) -> dict:
    """Fit camera height and pitch to a measured row profile, with the field of view held fixed.

    .. attention::
        **The field of view cannot be fitted along with the other two.** In the small-angle limit
        ``h / sin(theta + phi * r)`` is unchanged by halving ``h``, ``theta`` and ``phi`` together,
        so the three-parameter problem is degenerate -- run against MuJoCo's own camera, a free fit
        returned 0.63 m / 21.5 deg / 22.4 deg with a 0.2 cm residual against a ground truth of
        1.26 m / 47.6 deg / 58.8 deg. It fit beautifully and every number was wrong.

        The field of view is the one parameter that is independently known: it comes from the
        camera's own intrinsics, which ``scripts/probe_depth_camera.py`` reads off the live stream.
        Fixing it makes the shape of the profile determine the pitch and its scale determine the
        height, and the problem is then well posed.

    Args:
        profile: Mean depth per image row [m], top row first.
        max_range: Clip range; rows at or beyond it carry no information and are dropped.
        vfov_deg: Vertical field of view, held fixed.

    Returns:
        ``height``, ``pitch_deg`` and the RMS residual [m].
    """
    n = len(profile)
    rows = (np.arange(n) - (n - 1) / 2.0) / (n - 1)
    usable = np.isfinite(profile) & (profile > 0.1) & (profile < max_range * 0.98)
    if usable.sum() < 8:
        raise SystemExit(
            f"only {int(usable.sum())} of {n} rows are usable -- every other row is clipped at"
            f" {max_range} m or invalid. Point the camera at open flat ground."
        )
    r, d = rows[usable], profile[usable]
    v = math.radians(vfov_deg)

    best = None
    for h in np.arange(0.6, 2.0, 0.005):
        for p in np.radians(np.arange(15.0, 80.0, 0.1)):
            res = float(np.sqrt(np.mean((predict(r, h, p, v) - d) ** 2)))
            if best is None or res < best[0]:
                best = (res, h, p)
    res, h, p = best
    return {"height": h, "pitch_deg": math.degrees(p), "vfov_deg": vfov_deg, "residual": res}


def torso_pitch_deg(domain_id: int, interface: str, timeout: float = 10.0) -> float | None:
    """Forward pitch of the robot's trunk right now, from the IMU [deg]; positive is leaning forward.

    The fit measures the camera's angle **to the ground**, which is the mounting pitch plus whatever
    the trunk is doing. A robot hanging in a harness or standing in a crouch tilts the camera with
    it, and without this the two are indistinguishable -- a 13 degree lean reads as a 13 degree
    mounting error.

    Args:
        domain_id: DDS domain; 0 on a real G1.
        interface: Network interface on the robot's subnet.
        timeout: Give up after this long [s].

    Returns:
        Trunk pitch [deg], or ``None`` if no ``rt/lowstate`` arrived.
    """
    import time

    from g1_deploy import core
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    latest: dict = {}
    try:
        ChannelFactoryInitialize(domain_id, interface)
    except Exception:
        return None
    ChannelSubscriber("rt/lowstate", LowState_).Init(lambda m: latest.__setitem__("m", m), 10)
    deadline = time.monotonic() + timeout
    while "m" not in latest:
        if time.monotonic() > deadline:
            return None
        time.sleep(0.05)
    quat = np.asarray(latest["m"].imu_state.quaternion[:4], dtype=np.float32)
    # Gravity expressed in the trunk frame. Leaning forward tips the trunk's +Z toward its +X, which
    # puts a *positive* x component on the gravity vector -- verified against constructed rotations
    # of +-15 and +30 degrees rather than reasoned about, because the first version had this
    # backwards and turned a 10.6 degree forward lean into a 10.6 degree lean back, which then
    # doubled into the mounting estimate instead of cancelling.
    gravity = core.quat_apply_inverse_wxyz(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))
    return float(np.degrees(np.arcsin(np.clip(gravity[0], -1.0, 1.0))))


def report(profile: np.ndarray, max_range: float, vfov_deg: float, trunk_deg: float | None = None) -> None:
    """Fit and print the comparison against the contract."""
    out = fit(profile, max_range, vfov_deg)
    print(f"\n  camera height   {out['height']:.3f} m")
    print(f"  downward pitch  {out['pitch_deg']:.1f} deg     contract {CONTRACT_PITCH_DEG}")
    print(f"  vertical fov    {out['vfov_deg']:.1f} deg     (held fixed, not fitted)")
    print(f"  rms residual    {out['residual'] * 100:.1f} cm"
          + ("   (good fit)" if out["residual"] < 0.02 else "   (poor -- is the ground flat and clear?)"))
    measured = out["pitch_deg"]
    if trunk_deg is None:
        print("\n  [..] trunk pitch unknown, so this is the camera's angle TO THE GROUND, which is")
        print("       the mounting pitch plus whatever the robot is leaning. Pass --domain_id and")
        print("       --interface to read the IMU and separate them.")
        mounting = measured
    else:
        mounting = measured - trunk_deg
        lean = "forward" if trunk_deg > 0 else "back"
        print(f"\n  trunk pitch     {abs(trunk_deg):.1f} deg {lean} (IMU)")
        print(f"  -> mounting     {mounting:.1f} deg     contract {CONTRACT_PITCH_DEG}")
    dp = mounting - CONTRACT_PITCH_DEG
    if abs(dp) <= 3.0:
        print(f"\n  [ok] mounting agrees with the deployment to {abs(dp):.1f} deg")
    elif out["residual"] < 0.05:
        print(f"\n  [!!] the camera is mounted {abs(dp):.1f} deg {'lower' if dp > 0 else 'higher'}"
              " than the deployment renders. The policy would read ground at the wrong distance,")
        print("       and nothing downstream can detect it -- every frame is still 38x64.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=None, help="Listen live on this UDP port.")
    ap.add_argument("--npz", default=None, help="Read frames from a .npz written by a recorder.")
    ap.add_argument("--selftest", action="store_true", help="Fit MuJoCo's own camera, where the answer is known.")
    ap.add_argument("--frames", type=int, default=60, help="Frames to average before fitting.")
    ap.add_argument("--width", type=int, default=64, help="Frame width on --port.")
    ap.add_argument("--height", type=int, default=38,
                    help="Frame height on --port. The publisher's --raw_port stream carries the same"
                         " cropped field of view at a higher resolution, so pointing this at it"
                         " gives the fit more rows to work with and leaves the policy's port free"
                         " for the control loop.")
    ap.add_argument("--max_range", type=float, default=3.0)
    ap.add_argument("--domain_id", type=int, default=None,
                    help="Read the trunk's pitch off rt/lowstate so it can be subtracted. 0 on a"
                         " real G1. Without it the fit reports camera-to-ground, not mounting.")
    ap.add_argument("--interface", default="en6", help="Interface on the robot's subnet.")
    ap.add_argument("--vfov", type=float, default=CONTRACT_VFOV_DEG,
                    help="Vertical field of view to hold fixed [deg]. Take it from"
                         " scripts/probe_depth_camera.py against the real camera; the default is the"
                         " contract's. It cannot be fitted -- see fit().")
    args = ap.parse_args()

    if args.selftest:
        import mujoco

        from g1_deploy import core
        from g1_deploy.depth import load_contract
        from g1_deploy.sim.depth_camera import DepthRenderer, build_model_with_camera, contract_camera
        from g1_deploy.sim.env import DEFAULT_XML
        from scripts.benchmark_depth_mujoco import place_on_ground

        core.set_policy_backend("g1_29dof")
        spec = contract_camera(load_contract("policies/depth_student_w100"))
        model, cam = build_model_with_camera(str(DEFAULT_XML), spec, foot_plate=True)
        data = mujoco.MjData(model)
        data.qpos[7 : 7 + core.NUM_ROBOT_MOTORS] = np.asarray(core.build_default_pose(), np.float32)
        place_on_ground(model, data)
        renderer = DepthRenderer(model, spec, cam)
        frame = renderer.render(data)
        truth_h = float(data.cam_xpos[model.camera(cam).id][2])
        renderer.close()
        print(f"[selftest] MuJoCo ground truth: height {truth_h:.3f} m, pitch"
              f" {spec['pitch_deg']:.1f} deg, vfov {spec['fovy_deg']:.1f} deg")
        report(np.array([r[r > 0].mean() if (r > 0).any() else np.nan for r in frame]),
               args.max_range, spec['fovy_deg'])
        return 0

    if args.npz:
        frames = np.load(args.npz)["frames"]
    elif args.port:
        import time

        from g1_deploy.depth_link import DepthReceiver

        rx = DepthReceiver((args.height, args.width), port=args.port)
        print(f"[..] collecting {args.frames} frames on UDP {args.port} -- stand the robot on flat,")
        print("     clear ground, at least 3 m of it in front")
        rx.wait_for_frame(timeout=30.0)
        seen, out = set(), []
        while len(out) < args.frames:
            f, age = rx.latest()
            key = f.tobytes()[:64]
            if age < 0.1 and key not in seen:
                seen.add(key)
                out.append(f.copy())
            time.sleep(0.005)
        rx.close()
        frames = np.asarray(out)
    else:
        raise SystemExit("give one of --port, --npz or --selftest")

    mean = frames.mean(axis=0)
    print(f"[..] {len(frames)} frames, {mean.shape[0]} rows")
    trunk = None
    if args.domain_id is not None:
        trunk = torso_pitch_deg(args.domain_id, args.interface)
        if trunk is None:
            print("[warn] no rt/lowstate; reporting camera-to-ground rather than mounting")
    report(np.array([r[r > 0].mean() if (r > 0).any() else np.nan for r in mean]),
           args.max_range, args.vfov, trunk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
