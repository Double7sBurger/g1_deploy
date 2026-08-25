# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay an Isaac Lab rollout into MuJoCo step by step and find which joint separates first.

Stage two of the lockstep comparison (:mod:`lockstep_record` is stage one). MuJoCo is initialized
from the recorded *whole* state -- root pose, root twist, every joint angle and rate -- so step 0 is
identical by construction and any later difference is dynamics.

Two modes, and the distinction matters:

``--mode teacher``
    Push the recorded joint targets in open loop. The policy never runs, so the two robots receive
    byte-identical commands and the only thing that can differ is how they respond. This is the mode
    that localizes a *dynamics* difference to a joint.
``--mode closed``
    Let MuJoCo run the policy itself from the same initial state. Shows the compounded difference,
    which is what the videos show, but cannot attribute it -- once the two states differ at all, the
    policies see different inputs and the comparison stops being controlled.

.. attention::
    Open-loop replay diverges eventually no matter what: two simulators cannot stay on one trajectory
    once any difference exists, and the gap then grows on its own. The late-time magnitude means
    nothing. What is diagnostic is **which** joint leaves first and **when**, because a joint that
    separates in the first few steps implicates the actuator or the joint model, while one that waits
    for ground contact implicates the contact model.

Usage::

    uv run --no-project python deploy/lockstep_compare.py --rollout /tmp/lab_rollout.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

from g1_deploy import core
from g1_deploy.sim.env import SIM_DT, G1SimEnv

DEFAULT_XML = Path.home() / "workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"

TRUNK_BODIES = ("pelvis", "waist_yaw_link", "waist_roll_link", "torso_link")


class _ReplayBridge:
    """Stands in for :class:`~dds_sim_bridge.G1SimBridge` without any DDS.

    The transport was already proven neutral -- 45/45 episodes bit-identical through DDS versus
    in-process -- so putting it in this loop would only cost wall-clock and add a way to be wrong.
    """

    num_motors = core.NUM_ROBOT_MOTORS

    def __init__(self, kp: np.ndarray, kd: np.ndarray):
        self._kp = np.asarray(kp, float)
        self._kd = np.asarray(kd, float)
        self._target = np.zeros(self.num_motors)
        self._have = False

    @property
    def cmd_received(self) -> bool:
        return self._have

    def set_target(self, target: np.ndarray) -> None:
        self._target = np.asarray(target, float)
        self._have = True

    def read_cmd(self):
        zeros = np.zeros(self.num_motors)
        return self._target, zeros, self._kp, self._kd, zeros

    def publish_low_state(self, obs: dict) -> None:
        """No-op: nothing subscribes in this harness."""


def load_rollout(path: str) -> dict:
    """Load a :mod:`lockstep_record` npz into plain arrays."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def init_mujoco_from(env: G1SimEnv, rec: dict, step: int = 0) -> None:
    """Put MuJoCo in exactly the state Isaac Lab was in at ``step``."""
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[0:3] = rec["root_pos"][step]
    env.data.qpos[3:7] = rec["root_quat_wxyz"][step]
    env.data.qpos[env.qpos_adr] = rec["q29"][step]
    env.data.qvel[0:3] = rec["root_lin_vel_w"][step]
    env.data.qvel[3:6] = rec["root_ang_vel_b"][step]
    env.data.qvel[env.qvel_adr] = rec["dq29"][step]
    mujoco.mj_forward(env.model, env.data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rollout", default="/tmp/lab_rollout.npz")
    parser.add_argument("--mode", choices=("teacher", "closed"), default="teacher")
    parser.add_argument("--policy", default=None, help="Required for --mode closed.")
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    parser.add_argument("--sim_dt", type=float, default=SIM_DT)
    parser.add_argument("--contact_timeconst", type=float, default=0.005)
    parser.add_argument("--trunk_mass_scale", type=float, default=1.0)
    parser.add_argument("--joint_frictionloss", type=float, default=None)
    parser.add_argument("--joint_damping", type=float, default=None)
    parser.add_argument("--align_legs_to_usd", default=None)
    parser.add_argument("--tol", type=float, default=0.05, help="Per-joint divergence threshold [rad].")
    parser.add_argument("--steps", type=int, default=None, help="Limit the number of control steps.")
    parser.add_argument(
        "--posture_steps",
        type=int,
        default=25,
        help="Control steps used for the posture statistics. Kept short on purpose: open-loop replay"
        " eventually topples, and averaging a fall into a posture number is meaningless.",
    )
    parser.add_argument("--trace", default="left_knee_joint", help="Joint to trace step by step.")
    parser.add_argument("--trace_steps", type=int, default=8)
    args = parser.parse_args()

    core.set_policy_backend("newton")
    rec = load_rollout(args.rollout)
    names = [str(n) for n in rec["joint_names"]]
    if names != core.POLICY_JOINT_NAMES:
        raise ValueError("rollout joint order does not match the active deployment table")

    n_steps = len(rec["q29"]) - 1
    if args.steps is not None:
        n_steps = min(n_steps, args.steps)

    kp, kd = core.control_gains()
    bridge = _ReplayBridge(kp, kd)
    mass_scale = {b: args.trunk_mass_scale for b in TRUNK_BODIES} if args.trunk_mass_scale != 1.0 else None
    env = G1SimEnv(
        args.xml,
        bridge,
        sim_dt=args.sim_dt,
        decimation=int(round(core.CONTROL_DT / args.sim_dt)),
        contact_timeconst=args.contact_timeconst,
        body_mass_scale=mass_scale,
        joint_frictionloss=args.joint_frictionloss,
        joint_damping=args.joint_damping,
        align_legs_to_usd=args.align_legs_to_usd,
    )
    init_mujoco_from(env, rec, 0)

    runner = None
    if args.mode == "closed":
        if args.policy is None:
            raise ValueError("--mode closed needs --policy")
        import torch

        module = torch.jit.load(args.policy)
        module.eval()

        def policy(obs):
            with torch.inference_mode():
                return module(torch.from_numpy(np.ascontiguousarray(obs, np.float32))).numpy()

        runner = core.G1PolicyRunner(policy)
        runner.reset()

    motors = [i for i in range(core.NUM_ROBOT_MOTORS) if i not in set(core.UNMAPPED_ROBOT_MOTORS.tolist())]
    motor_name = {v: k for k, v in core._ROBOT_INDEX_BY_POLICY_NAME.items()}
    q_mj = np.zeros((n_steps + 1, core.NUM_ROBOT_MOTORS))
    tau_mj = np.zeros((n_steps + 1, core.NUM_ROBOT_MOTORS))
    root_z_mj = np.zeros(n_steps + 1)
    q_mj[0] = env.data.qpos[env.qpos_adr]
    root_z_mj[0] = env.data.qpos[2]

    for k in range(n_steps):
        if args.mode == "teacher":
            target = rec["target29"][k]
        else:
            quat = np.asarray(env.data.qpos[3:7], np.float32)
            command = np.array(
                [rec["vx"], rec["vy"], core.yaw_rate_from_heading(float(rec["heading"]), quat)], np.float32
            )
            target, _ = runner.step(
                np.asarray(env.data.qpos[env.qpos_adr], np.float32),
                np.asarray(env.data.qvel[env.qvel_adr], np.float32),
                quat,
                np.asarray(env.data.qvel[3:6], np.float32),
                command,
            )
            target[core.UNMAPPED_ROBOT_MOTORS] = 0.0
        bridge.set_target(target)
        tau_mj[k] = env.compute_torques()
        env.step_control()
        q_mj[k + 1] = env.data.qpos[env.qpos_adr]
        root_z_mj[k + 1] = env.data.qpos[2]

    q_lab = rec["q29"][: n_steps + 1]
    tau_lab = rec["torque29"][: n_steps + 1]
    z_lab = rec["root_pos"][: n_steps + 1, 2]
    diff = np.abs(q_mj - q_lab)

    print("\n" + "=" * 84)
    print(f"LOCKSTEP {args.mode.upper()}  {n_steps} control steps, identical initial state")
    print("=" * 84)
    print(f"  step 0 |dq| max over joints = {diff[0].max():.2e} rad   (0 means the states really matched)")

    print(f"\n{'joint':26s} {'first |dq|>' + f'{args.tol:.2f}':>16s} {'|dq| @1s':>9s} {'lab q':>8s} {'mj q':>8s}")
    print("-" * 84)
    first = {}
    for m in motors:
        over = np.flatnonzero(diff[:, m] > args.tol)
        first[m] = int(over[0]) if len(over) else None
    order = sorted(motors, key=lambda m: (first[m] is None, first[m] if first[m] is not None else 0))
    at1s = min(n_steps, int(round(1.0 / core.CONTROL_DT)))
    for m in order[:14]:
        k = first[m]
        when = f"step {k:4d} ({k * core.CONTROL_DT:5.2f}s)" if k is not None else "never"
        print(f"{motor_name[m]:26s} {when:>16s} {diff[at1s, m]:9.3f} {q_lab[at1s, m]:+8.3f} {q_mj[at1s, m]:+8.3f}")

    print(f"\n{'=' * 84}\nPOSTURE: where the two robots sit relative to the SAME target")
    print("=" * 84)
    window = slice(0, min(args.posture_steps, n_steps) + 1)
    print(f"{'joint':26s} {'lab q-target':>13s} {'mj q-target':>12s} {'lab |tau|':>10s} {'mj |tau|':>9s}")
    print("-" * 84)
    for label in (
        "left_knee_joint",
        "right_knee_joint",
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
    ):
        m = core._ROBOT_INDEX_BY_POLICY_NAME[label]
        tgt = rec["target29"][window][:, m]
        print(
            f"{label:26s} {np.mean(q_lab[window][:, m] - tgt):+13.3f} {np.mean(q_mj[window][:, m] - tgt):+12.3f} "
            f"{np.mean(np.abs(tau_lab[window][:, m])):10.1f} {np.mean(np.abs(tau_mj[window][:, m])):9.1f}"
        )
    w = window
    print(
        f"\n  (statistics over the first {w.stop} steps = {(w.stop - 1) * core.CONTROL_DT:.2f}s;"
        f" open-loop replay topples later and averaging a fall into a posture number means nothing)"
    )
    print(
        f"  pelvis height   lab {z_lab[w].mean():.3f} m   mujoco {root_z_mj[w].mean():.3f} m"
        f"   (both start {z_lab[0]:.3f})"
    )
    knee = core._ROBOT_INDEX_BY_POLICY_NAME[args.trace]
    print(
        f"  {args.trace:24s} lab {q_lab[w][:, knee].mean():+.3f} rad   mujoco"
        f" {q_mj[w][:, knee].mean():+.3f} rad   commanded {rec['target29'][w][:, knee].mean():+.3f}"
    )

    print(f"\n{'=' * 84}\nSTEP BY STEP: {args.trace}   (state and target identical at step 0)")
    print("=" * 84)
    print(
        f"{'step':>5s} {'target':>8s} {'q lab':>8s} {'q mj':>8s} {'q mj-lab':>9s} "
        f"{'tau lab':>9s} {'tau mj':>8s} {'z lab':>7s} {'z mj':>7s}"
    )
    print("-" * 84)
    for k in range(min(args.trace_steps, n_steps) + 1):
        print(
            f"{k:5d} {rec['target29'][k, knee]:+8.3f} {q_lab[k, knee]:+8.3f} {q_mj[k, knee]:+8.3f} "
            f"{q_mj[k, knee] - q_lab[k, knee]:+9.3f} {tau_lab[k, knee]:+9.1f} {tau_mj[k, knee]:+8.1f} "
            f"{z_lab[k]:7.3f} {root_z_mj[k]:7.3f}"
        )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
