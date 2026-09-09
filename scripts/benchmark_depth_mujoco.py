# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-loop benchmark for a depth student: MuJoCo renders what the policy sees.

``benchmark_mujoco.py`` scores a proprioception-only policy. This does the same for a vision
student, with the camera in the loop -- MuJoCo renders a 64x38 depth frame from the chest camera
every control step, :mod:`g1_deploy.depth` preprocesses it exactly as deployment will, and the
policy reads both halves of its observation. It is the only way to answer "does this distilled
student actually walk" without putting it on the robot.

Single process on purpose. ``benchmark_mujoco.py`` forks workers, but each would need its own GL
context, and on macOS an offscreen context off the main thread is a reliable way to get a crash
rather than a number.

Frames are optionally dumped so the same rollout can be compared against real camera frames --
see ``scripts/compare_depth.py``.

Usage::

    python scripts/benchmark_depth_mujoco.py --export_dir policies/depth_student_w100 \\
        --episodes 5 --dump /tmp/sim_frames.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from g1_deploy import benchmark as bp
from g1_deploy import core
from g1_deploy.depth import G1DepthPolicyRunner, load_contract, load_depth_policy
from g1_deploy.sim.depth_camera import DepthRenderer, build_model_with_camera, contract_camera
from g1_deploy.sim.mjcf import ground_height
from g1_deploy.sim.env import DEFAULT_XML


def place_on_ground(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Drop the free root until the lowest robot geom rests on the ground; return the root height."""
    geoms = [
        i for i in range(model.ngeom)
        if model.geom_bodyid[i] != 0 and model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE
    ]
    surface = ground_height(model, float(data.qpos[0]), float(data.qpos[1]))
    data.qpos[2] = surface + 1.0
    mujoco.mj_forward(model, data)
    data.qpos[2] = surface + 1.0 - (float(data.geom_xpos[geoms, 2].min()) - surface) + 0.002
    mujoco.mj_forward(model, data)
    return float(data.qpos[2])


def run_episode(model, renderer, runner, index: int, suite: str, dump: list | None) -> dict:
    """One scored episode with the camera in the loop."""
    episode = bp.Episode(suite, index)
    data = mujoco.MjData(model)

    mapped = core.POLICY_TO_ROBOT >= 0
    robot_names = [core.POLICY_JOINT_NAMES[i] for i in np.flatnonzero(mapped)]
    jitter = np.zeros(core.NUM_ROBOT_MOTORS, dtype=np.float32)
    jitter[core.POLICY_TO_ROBOT[mapped]] = episode.joint_jitter(robot_names)
    default = np.zeros(core.NUM_ROBOT_MOTORS, dtype=np.float32)
    default[core.POLICY_TO_ROBOT[mapped]] = core.DEFAULT_JOINT_POS[mapped]

    n_motors = core.NUM_ROBOT_MOTORS
    data.qpos[7 : 7 + n_motors] = default + jitter
    place_on_ground(model, data)

    kp, kd = core.control_gains()
    steps_per_ctrl = int(round(bp.CONTROL_DT / model.opt.timestep))
    runner.reset()

    n_steps = int(round(bp.EPISODE_S / bp.CONTROL_DT))
    rec = {k: [] for k in ("cmd", "lin_vel_b", "ang_vel_b", "gravity_z", "root_z")}
    fell_at = None
    for k in range(n_steps):
        quat = np.array(data.qpos[3:7], dtype=np.float32)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, data.qpos[3:7])
        rot = rot.reshape(3, 3)
        grav_z = float(core.quat_apply_inverse_wxyz(quat, np.array([0.0, 0.0, -1.0], np.float32))[2])
        if grav_z > bp.FALL_GRAVITY_Z:
            fell_at = k * bp.CONTROL_DT
            break

        vx, vy, heading, standing = episode.planar_command(k)
        cmd = (np.zeros(3, np.float32) if standing
               else np.array([vx, vy, core.yaw_rate_from_heading(heading, quat)], np.float32))
        rec["cmd"].append(cmd)
        rec["lin_vel_b"].append(rot.T @ data.qvel[0:3])
        rec["ang_vel_b"].append(np.array(data.qvel[3:6]))
        rec["gravity_z"].append(grav_z)
        rec["root_z"].append(float(data.qpos[2]))

        depth = renderer.render(data)
        if dump is not None and k % 25 == 0:
            dump.append(depth.copy())

        target, _action = runner.step(
            data.qpos[7 : 7 + n_motors].copy().astype(np.float32),
            data.qvel[6 : 6 + n_motors].copy().astype(np.float32),
            quat,
            data.qvel[3:6].copy().astype(np.float32),
            cmd,
            depth,
        )
        target[core.UNMAPPED_ROBOT_MOTORS] = 0.0
        for _ in range(steps_per_ctrl):
            q = data.qpos[7 : 7 + n_motors]
            dq = data.qvel[6 : 6 + n_motors]
            data.ctrl[:] = kp * (target - q) + kd * (-dq)
            mujoco.mj_step(model, data)

    survived = fell_at if fell_at is not None else bp.EPISODE_S
    out = {k: np.asarray(v, dtype=np.float32) for k, v in rec.items()}
    out["survived"] = np.float32(survived)
    out["fell"] = np.float32(fell_at is not None)
    out["distance"] = np.float32(np.hypot(data.qpos[0], data.qpos[1]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export_dir", default="policies/depth_student_w100")
    ap.add_argument("--xml", default=str(DEFAULT_XML))
    ap.add_argument("--suite", choices=("hold", "sequence"), default="hold")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--contact_timeconst", type=float, default=0.005)
    ap.add_argument("--dump", default=None, help="Write sampled depth frames here as .npz.")
    ap.add_argument("--foot_plate", action="store_true",
                    help="Replace each foot's four contact spheres with the solid plate training"
                         " uses. MuJoCo cannot load the training USD, so this reproduces the one"
                         " override that governs lateral stability.")
    ap.add_argument(
        "--blind",
        choices=("off", "far", "flat", "frozen"),
        default="off",
        help="Replace the rendered depth with a constant frame, to measure what the camera is"
        " worth. 'far' is max range everywhere -- simple, but an all-far frame never occurred in"
        " training, so it tests an out-of-distribution input as much as it tests blindness. 'flat'"
        " holds the frame the camera sees standing on level ground: in distribution, and carrying no"
        " terrain information, which is the fair control. 'frozen' holds whatever the first frame of"
        " the episode was.",
    )
    args = ap.parse_args()

    torch.set_num_threads(1)
    core.set_policy_backend("g1_29dof")
    contract = load_contract(args.export_dir)
    if list(contract["joint_names"]) != list(core.POLICY_JOINT_NAMES):
        raise SystemExit("contract joint order differs from the g1_29dof profile")

    spec = contract_camera(contract)
    model, camera = build_model_with_camera(args.xml, spec, foot_plate=args.foot_plate)
    model.geom_solref[:, 0] = args.contact_timeconst
    renderer = DepthRenderer(model, spec, camera)

    policy = load_depth_policy(str(Path(args.export_dir) / "policy.pt"), contract)
    runner = G1DepthPolicyRunner(policy, contract)

    if args.blind == "far":
        constant = np.full((spec["height"], spec["width"]), spec["max_range"], np.float32)
        renderer.render = lambda _data: constant  # type: ignore[method-assign]
    elif args.blind == "flat":
        # Render once on a level scene, then hold it. The policy keeps a plausible ground plane in
        # view and loses only the terrain, which is the difference being measured.
        flat_model, flat_cam = build_model_with_camera(str(DEFAULT_XML), spec, foot_plate=args.foot_plate)
        flat_data = mujoco.MjData(flat_model)
        flat_data.qpos[7 : 7 + core.NUM_ROBOT_MOTORS] = np.asarray(core.build_default_pose(), np.float32)
        place_on_ground(flat_model, flat_data)
        flat_renderer = DepthRenderer(flat_model, spec, flat_cam)
        constant = flat_renderer.render(flat_data).copy()
        flat_renderer.close()
        renderer.render = lambda _data: constant  # type: ignore[method-assign]
    elif args.blind == "frozen":
        held: dict = {}
        real_render = renderer.render

        def _frozen(data):
            if "f" not in held:
                held["f"] = real_render(data).copy()
            return held["f"]

        renderer.render = _frozen  # type: ignore[method-assign]

    dump: list | None = [] if args.dump else None
    print(f"[..] {args.suite} suite, {args.episodes} episodes x {bp.EPISODE_S:.0f}s"
          f"  camera {spec['width']}x{spec['height']} fovy {spec['fovy_deg']:.1f} pitch"
          f" {spec['pitch_deg']:.1f}"
          + ("  [FOOT PLATE]" if args.foot_plate else "  [spheres]")
          + ("" if args.blind == "off" else f"  [BLIND {args.blind}]"))

    results = []
    t0 = time.monotonic()
    for i in range(args.episodes):
        r = run_episode(model, renderer, runner, i, args.suite, dump)
        results.append(r)
        # Report what was asked for beside what happened. "Survived" alone counts a robot that
        # stands still through a walk command as a success, which is how a policy that ignores the
        # command entirely can score 80%.
        cmd_v = float(np.linalg.norm(r["cmd"][:, :2], axis=1).mean()) if len(r["cmd"]) else float("nan")
        got_v = float(np.linalg.norm(r["lin_vel_b"][:, :2], axis=1).mean()) if len(r["cmd"]) else float("nan")
        err = (float(np.linalg.norm(r["cmd"][:, :2] - r["lin_vel_b"][:, :2], axis=1).mean())
               if len(r["cmd"]) else float("nan"))
        print(f"  ep {i:>2}: survived {float(r['survived']):5.2f}s  cmd {cmd_v:4.2f} -> got {got_v:4.2f} m/s"
              f"  err {err:.3f}  travelled {float(r['distance']):5.2f} m")
    renderer.close()

    survived = np.array([float(r["survived"]) for r in results])
    success = float((survived >= bp.EPISODE_S - 1e-6).mean())
    errs = [
        float(np.linalg.norm(r["cmd"][:, :2] - r["lin_vel_b"][:, :2], axis=1).mean())
        for r in results
        if len(r["cmd"])
    ]
    moving = [r for r in results if len(r["cmd"]) and np.linalg.norm(r["cmd"][:, :2], axis=1).mean() > 0.05]
    ratio = float(np.mean([
        np.linalg.norm(r["lin_vel_b"][:, :2], axis=1).mean() / np.linalg.norm(r["cmd"][:, :2], axis=1).mean()
        for r in moving])) if moving else float("nan")
    print(f"\n[ok] survived-to-the-end {100 * success:.1f}%   mean survived {survived.mean():.2f}s"
          f"   ({time.monotonic() - t0:.0f}s wall)")
    print(f"     tracking err {np.mean(errs):.3f} m/s   achieved/commanded speed {ratio:.2f}"
          f"   over {len(moving)} moving episodes")
    print("     'survived' is only 'did not fall'; the speed ratio is what says whether it walked.")

    if dump:
        np.savez_compressed(args.dump, frames=np.asarray(dump, dtype=np.float32))
        print(f"[ok] {len(dump)} depth frames -> {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
