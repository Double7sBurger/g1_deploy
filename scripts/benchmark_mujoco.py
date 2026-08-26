# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the sim-to-sim benchmark in MuJoCo and write one ``.npz`` of per-episode metrics.

The schedule, the initial perturbation, the fall rule and the scoring all come from
:mod:`benchmark_protocol`; this file only steps MuJoCo. See :mod:`benchmark_isaaclab` for the other
side and :mod:`benchmark_report` for the comparison.

Usage::

    uv run --no-project python deploy/benchmark_mujoco.py \\
        --policy logs/.../deploy_2999/policy.pt --policy_physics newton --suite hold
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

from g1_deploy import benchmark as bp
from g1_deploy import core

DEFAULT_XML = Path.home() / "workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"

_WORKER: dict = {}


def _init_worker(policy_path: str, backend: str, xml: str, timeconst: float) -> None:
    """Load the model and policy once per worker process."""
    core.set_policy_backend(backend)
    module = torch.jit.load(policy_path)
    module.eval()
    torch.set_num_threads(1)
    _WORKER.update(module=module, xml=xml, timeconst=timeconst)


def _policy(obs: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        return _WORKER["module"](torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32))).numpy()


def _place_on_ground(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Drop the free root until the lowest robot geom rests on the floor; return the root height."""
    robot_geoms = [
        i for i in range(model.ngeom) if model.geom_bodyid[i] != 0 and model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE
    ]
    data.qpos[2] = 1.0
    mujoco.mj_forward(model, data)
    data.qpos[2] = 1.0 - float(data.geom_xpos[robot_geoms, 2].min()) + 0.002
    mujoco.mj_forward(model, data)
    return float(data.qpos[2])


def run_episode(job: tuple[str, int]) -> dict:
    """Run one benchmark episode and return its scored metrics."""
    suite, index = job
    episode = bp.Episode(suite, index)
    model = mujoco.MjModel.from_xml_path(_WORKER["xml"])
    if _WORKER["timeconst"] > 0:
        model.geom_solref[:, 0] = _WORKER["timeconst"]
    data = mujoco.MjData(model)

    mapped = core.POLICY_TO_ROBOT >= 0
    robot_names = [core.POLICY_JOINT_NAMES[i] for i in np.flatnonzero(mapped)]
    jitter_robot = np.zeros(core.NUM_ROBOT_MOTORS, dtype=np.float32)
    jitter_robot[core.POLICY_TO_ROBOT[mapped]] = episode.joint_jitter(robot_names)
    default_robot = np.zeros(core.NUM_ROBOT_MOTORS, dtype=np.float32)
    default_robot[core.POLICY_TO_ROBOT[mapped]] = core.DEFAULT_JOINT_POS[mapped]

    data.qpos[7 : 7 + core.NUM_ROBOT_MOTORS] = default_robot + jitter_robot
    _place_on_ground(model, data)

    kp, kd = core.control_gains()
    steps_per_ctrl = int(round(bp.CONTROL_DT / model.opt.timestep))
    runner = core.G1PolicyRunner(_policy)
    runner.reset()
    target = default_robot.copy()

    def state():
        """Current (quat_wxyz, body linear velocity, body angular velocity, gravity z)."""
        quat = np.array(data.qpos[3:7], dtype=np.float32)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, data.qpos[3:7])
        # MuJoCo reports a free joint's linear velocity in the world frame and its angular velocity
        # in the body frame; the command lives in the body frame, so only the linear part rotates.
        lin_b = rot.reshape(3, 3).T @ data.qvel[0:3]
        return (
            quat,
            lin_b,
            np.array(data.qvel[3:6]),
            float(core.quat_apply_inverse_wxyz(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))[2]),
        )

    def command_at(step: int, quat: np.ndarray) -> np.ndarray:
        vx, vy, heading, standing = episode.planar_command(step)
        if standing:
            return np.zeros(3, dtype=np.float32)
        return np.array([vx, vy, core.yaw_rate_from_heading(heading, quat)], dtype=np.float32)

    # Settle: hold the default pose with zero action, but keep feeding the observation buffer so the
    # policy starts with the same five frames of history Isaac Lab's manager will have built.
    for _ in range(int(bp.SETTLE_S / bp.CONTROL_DT)):
        for _ in range(steps_per_ctrl):
            q = data.qpos[7 : 7 + core.NUM_ROBOT_MOTORS]
            dq = data.qvel[6 : 6 + core.NUM_ROBOT_MOTORS]
            data.ctrl[:] = kp * (default_robot - q) + kd * (-dq)
            mujoco.mj_step(model, data)
        quat, _, _, _ = state()
        runner.observe(
            data.qpos[7 : 7 + core.NUM_ROBOT_MOTORS].copy().astype(np.float32),
            data.qvel[6 : 6 + core.NUM_ROBOT_MOTORS].copy().astype(np.float32),
            quat,
            data.qvel[3:6].copy().astype(np.float32),
            command_at(0, quat),
        )

    n_steps = int(round(bp.EPISODE_S / bp.CONTROL_DT))
    rec = {k: [] for k in ("cmd", "lin_vel_b", "ang_vel_b", "gravity_z", "root_z")}
    for k in range(n_steps):
        quat, lin_b, ang_b, grav_z = state()
        cmd = command_at(k, quat)
        rec["cmd"].append(cmd)
        rec["lin_vel_b"].append(lin_b)
        rec["ang_vel_b"].append(ang_b)
        rec["gravity_z"].append(grav_z)
        rec["root_z"].append(float(data.qpos[2]))

        target, _ = runner.step(
            data.qpos[7 : 7 + core.NUM_ROBOT_MOTORS].copy().astype(np.float32),
            data.qvel[6 : 6 + core.NUM_ROBOT_MOTORS].copy().astype(np.float32),
            quat,
            data.qvel[3:6].copy().astype(np.float32),
            cmd,
        )
        target[core.UNMAPPED_ROBOT_MOTORS] = 0.0
        for _ in range(steps_per_ctrl):
            q = data.qpos[7 : 7 + core.NUM_ROBOT_MOTORS]
            dq = data.qvel[6 : 6 + core.NUM_ROBOT_MOTORS]
            data.ctrl[:] = kp * (target - q) + kd * (-dq)
            mujoco.mj_step(model, data)

    out = bp.score({k: np.array(v) for k, v in rec.items()})
    out["index"] = index
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", required=True)
    parser.add_argument("--policy_physics", choices=("physx", "newton", "g1_29dof"), default="newton")
    parser.add_argument("--suite", choices=("hold", "sequence"), default="hold")
    parser.add_argument("--repeats", type=int, default=3, help="Seeds per hold-grid entry, or episodes for sequence.")
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    parser.add_argument("--contact_timeconst", type=float, default=0.005)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="deploy/benchmark_mujoco.npz")
    args = parser.parse_args()

    n = bp.num_episodes(args.suite, args.repeats)
    jobs = [(args.suite, i) for i in range(n)]
    print(f"[..] mujoco  suite={args.suite}  {n} episodes x {bp.EPISODE_S:.0f}s  workers={args.workers}")

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(args.policy, args.policy_physics, args.xml, args.contact_timeconst),
    ) as pool:
        results = pool.map(run_episode, jobs)

    results.sort(key=lambda r: r["index"])
    keys = [k for k in results[0] if k != "index"]
    np.savez(
        args.out,
        suite=args.suite,
        policy=args.policy,
        side="mujoco",
        **{k: np.array([r[k] for r in results]) for k in keys},
    )
    print(
        f"[ok] {args.out}  success {np.mean([r['success'] for r in results]):.1%}"
        f"  lin err {np.nanmean([r['lin_vel_err'] for r in results]):.3f} m/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
