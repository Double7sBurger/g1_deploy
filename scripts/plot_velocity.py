# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot what the policy was asked for against what it achieved.

The two numbers live in different processes. The command is the control loop's -- it is what the
operator typed, or the heading controller derived -- and the achieved velocity exists only in the
simulator, because ``rt/lowstate`` carries orientation and angular rate but no linear velocity. So
each side records its own ``.npz`` stamped with the wall clock, and this aligns them.

**On hardware only the command half exists.** There is no base-velocity estimate on this robot, and
nothing here invents one: a hardware run plots the command alone, which is still worth seeing beside
a simulated run of the same checkpoint.

Usage::

    # terminal 1
    mjpython scripts/run_sim_loop.py --domain_id 41 --interface lo0 --hoist_s 4 --viz \\
        --xml terrain/g1_box.xml --depth policies/ymsd_v2 --foot_plate --record /tmp/sim.npz
    # terminal 2
    python scripts/run_policy_loop.py --real --skip_release_mode --teleop --no_strafe \\
        --depth policies/ymsd_v2 --policy_physics g1_29dof --vx 0.1 --duration inf \\
        --domain_id 41 --interface lo0 --record /tmp/cmd.npz
    # afterwards
    python scripts/plot_velocity.py --sim /tmp/sim.npz --cmd /tmp/cmd.npz --out /tmp/vel.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CHANNELS = (
    ("forward  vx", 0, "m/s"),
    ("lateral  vy", 1, "m/s"),
    ("yaw rate wz", 2, "rad/s"),
)
"""What to plot, as ``(label, index, unit)``. The command's third element is a yaw *rate*, derived
from the heading error by :func:`g1_deploy.core.yaw_rate_from_heading` -- so it is comparable with
the body-frame angular velocity, not with a heading."""


def load(path: str | None) -> dict | None:
    """Read one recording, or ``None`` if it was not given.

    Args:
        path: ``.npz`` written by ``--record``.

    Returns:
        The archive as a dict of arrays, or ``None``.

    Raises:
        SystemExit: If the file is missing, rather than plotting half the picture silently.
    """
    if path is None:
        return None
    if not Path(path).exists():
        raise SystemExit(f"{path} does not exist -- was the run given --record?")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def align(cmd: dict, sim: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample the simulator's velocity onto the command's timestamps.

    Both sides stamp with ``time.time()`` on the same machine, so the clocks agree; what differs is
    the rate and the phase. Interpolating the simulator onto the command's instants keeps every
    commanded value as recorded -- the command is a step function the operator set, and resampling
    *it* would invent intermediate values that were never asked for.

    The window is clipped to the overlap. A simulator started earlier, or left running after the
    control loop exits, contributes nothing to a comparison.

    Args:
        cmd: Archive from ``run_policy_loop.py --record``.
        sim: Archive from ``run_sim_loop.py --record``.

    Returns:
        ``(t, commanded, achieved)`` -- seconds from the first shared sample, and both ``(n, 3)``.

    Raises:
        SystemExit: If the recordings do not overlap in time, which means they are from different
            runs and averaging them would produce a plausible, meaningless plot.
    """
    lo = max(cmd["t"][0], sim["t"][0])
    hi = min(cmd["t"][-1], sim["t"][-1])
    if hi <= lo:
        raise SystemExit(
            f"the recordings do not overlap: command {cmd['t'][0]:.1f}-{cmd['t'][-1]:.1f},"
            f" simulator {sim['t'][0]:.1f}-{sim['t'][-1]:.1f} (wall clock). Different runs?"
        )
    keep = (cmd["t"] >= lo) & (cmd["t"] <= hi)
    t = cmd["t"][keep]
    achieved = np.stack([
        np.interp(t, sim["t"], sim["lin_vel_b"][:, 0]),
        np.interp(t, sim["t"], sim["lin_vel_b"][:, 1]),
        np.interp(t, sim["t"], sim["ang_vel_b"][:, 2]),
    ], axis=1)
    return t - t[0], cmd["cmd"][keep], achieved


def report(t: np.ndarray, commanded: np.ndarray, achieved: np.ndarray | None) -> None:
    """Print the numbers behind the plot.

    A figure is for seeing the shape of the error; these are for quoting. Tracking is summarised
    over the samples where something was actually commanded -- averaging in the standing intervals
    would report a small error for a policy that never moved.

    Args:
        t: Seconds from the start.
        commanded: ``(n, 3)``.
        achieved: ``(n, 3)``, or ``None`` on a hardware recording.
    """
    print(f"  {len(t)} samples over {t[-1]:.1f} s")
    for label, i, unit in CHANNELS:
        line = f"  {label:<12s} commanded {commanded[:, i].min():+.2f} to {commanded[:, i].max():+.2f} {unit}"
        if achieved is not None:
            moving = np.abs(commanded[:, i]) > 0.05
            err = np.abs(achieved[:, i] - commanded[:, i])
            line += f"   achieved {achieved[:, i].min():+.2f} to {achieved[:, i].max():+.2f}"
            if moving.any():
                ratio = achieved[moving, i].mean() / commanded[moving, i].mean()
                line += (f"   err {err[moving].mean():.3f} {unit}"
                         f"   ratio {ratio:+.2f} over {int(moving.sum())} moving samples")
            else:
                line += "   never commanded away from zero"
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cmd", required=True, help=".npz from run_policy_loop.py --record.")
    ap.add_argument("--sim", default=None,
                    help=".npz from run_sim_loop.py --record. Omit for a hardware run, where the"
                         " achieved velocity does not exist.")
    ap.add_argument("--out", default=None, help="Write the figure here instead of showing it.")
    args = ap.parse_args()

    cmd = load(args.cmd)
    sim = load(args.sim)
    if sim is None:
        t, commanded, achieved = cmd["t"] - cmd["t"][0], cmd["cmd"], None
        print("[..] command only; no simulator recording, so nothing to compare against")
    else:
        t, commanded, achieved = align(cmd, sim)
    report(t, commanded, achieved)

    import matplotlib

    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(10, 7), sharex=True)
    for ax, (label, i, unit) in zip(axes, CHANNELS):
        ax.plot(t, commanded[:, i], label="commanded", lw=1.6)
        if achieved is not None:
            ax.plot(t, achieved[:, i], label="achieved", lw=1.0, alpha=0.85)
        ax.set_ylabel(f"{label}\n[{unit}]")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    axes[0].set_title("commanded against achieved velocity, body frame")
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=130)
        print(f"[ok] {args.out}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
