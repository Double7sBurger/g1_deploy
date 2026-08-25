# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare the two benchmark runs and print the report.

Takes the ``.npz`` files written by :mod:`benchmark_isaaclab` and :mod:`benchmark_mujoco` and prints
a per-suite summary, a per-command breakdown for the ``hold`` suite, and the episodes where the two
sides disagree most -- which is where to look next.

Usage::

    uv run --no-project python deploy/benchmark_report.py \\
        --isaaclab deploy/bench_lab_hold.npz --mujoco deploy/bench_mj_hold.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from g1_deploy import benchmark as bp

_METRICS = (
    ("success", "success rate", "{:.0%}", 1),
    ("survival_s", "survival [s]", "{:.2f}", 1),
    ("lin_vel_err", "lin vel err [m/s]", "{:.3f}", -1),
    ("ang_vel_err", "yaw rate err [rad/s]", "{:.3f}", -1),
    ("lin_track_score", "lin track score", "{:.3f}", 1),
    ("ang_track_score", "yaw track score", "{:.3f}", 1),
    ("root_z", "mean root height [m]", "{:.3f}", 0),
)


def _rule(char: str = "-", width: int = 78) -> str:
    return char * width


def _summary(lab: dict, mj: dict, suite: str) -> None:
    print(_rule("="))
    print(f"SUITE: {suite}    {len(lab['success'])} episodes x {bp.EPISODE_S:.0f}s per side")
    print(_rule("="))
    print(f"{'metric':24s} {'Isaac Lab':>12s} {'MuJoCo':>12s} {'delta':>12s}   {'':s}")
    print(_rule())
    for key, label, fmt, better in _METRICS:
        a, b = np.asarray(lab[key], float), np.asarray(mj[key], float)
        am, bm = np.nanmean(a), np.nanmean(b)
        delta = bm - am
        if better == 0:
            note = ""
        elif abs(delta) < 1e-9:
            note = "identical"
        else:
            note = "MuJoCo better" if np.sign(delta) == better else "MuJoCo worse"
        print(f"{label:24s} {fmt.format(am):>12s} {fmt.format(bm):>12s} {delta:+12.3f}   {note}")


def _per_command(lab: dict, mj: dict, repeats: int) -> None:
    print("\n" + _rule("="))
    print("PER-COMMAND BREAKDOWN (hold suite)")
    print(_rule("="))
    print(
        f"{'vx':>6s} {'vy':>6s} | {'success  lab / mj':>19s} | {'survival lab / mj':>19s} | {'lin err lab / mj':>19s}"
    )
    print(_rule())
    for g, (vx, vy) in enumerate(bp.HOLD_GRID):
        idx = [g + k * len(bp.HOLD_GRID) for k in range(repeats)]
        idx = [i for i in idx if i < len(lab["success"])]
        if not idx:
            continue
        sl, sm = np.mean(lab["success"][idx]), np.mean(mj["success"][idx])
        tl, tm = np.mean(lab["survival_s"][idx]), np.mean(mj["survival_s"][idx])
        el, em = np.nanmean(lab["lin_vel_err"][idx]), np.nanmean(mj["lin_vel_err"][idx])
        flag = "  <-- gap" if abs(sl - sm) >= 0.5 or abs(el - em) > 0.2 else ""
        print(
            f"{vx:6.2f} {vy:6.2f} |    {sl:5.0%} / {sm:5.0%}     |   {tl:5.2f} / {tm:5.2f}     "
            f"|   {el:5.3f} / {em:5.3f}{flag}"
        )


def _disagreements(lab: dict, mj: dict, suite: str, top: int = 6) -> None:
    print("\n" + _rule("="))
    print(f"LARGEST PER-EPISODE DISAGREEMENTS (top {top})")
    print(_rule("="))
    gap = np.abs(np.asarray(lab["survival_s"], float) - np.asarray(mj["survival_s"], float))
    order = np.argsort(-gap)[:top]
    print(f"{'episode':>8s} {'command':>22s} {'survival lab / mj':>21s} {'lin err lab / mj':>20s}")
    print(_rule())
    for i in order:
        ep = bp.Episode(suite, int(i))
        vx, vy, _, standing = ep.segments[0]
        label = "STAND" if standing else f"vx={vx:+.2f} vy={vy:+.2f}"
        print(
            f"{i:8d} {label:>22s}      {lab['survival_s'][i]:5.2f} / {mj['survival_s'][i]:5.2f}"
            f"        {lab['lin_vel_err'][i]:6.3f} / {mj['lin_vel_err'][i]:6.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--isaaclab", required=True)
    parser.add_argument("--mujoco", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    lab, mj = np.load(args.isaaclab, allow_pickle=True), np.load(args.mujoco, allow_pickle=True)
    suite = str(lab["suite"])
    if suite != str(mj["suite"]):
        raise ValueError(f"suite mismatch: {suite} vs {mj['suite']}")
    if len(lab["success"]) != len(mj["success"]):
        raise ValueError(f"episode-count mismatch: {len(lab['success'])} vs {len(mj['success'])}")

    print(f"\npolicy: {lab['policy']}")
    _summary(lab, mj, suite)
    if suite == "hold":
        _per_command(lab, mj, args.repeats)
    _disagreements(lab, mj, suite)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
