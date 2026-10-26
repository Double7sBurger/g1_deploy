# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Overlay two motion-tracking recordings -- typically hardware against ``--sim sync``.

deploynose holds one recording at a time, and the two runs do not share a clock: synchronous
simulation advances faster than wall time. Both record the reference time they were following, so
that is the axis everything is plotted against and the two curves line up frame for frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

JOINT = "left_ankle_pitch"


def load(directory: str):
    """Reference time, tilt error, command-following error and joint angles for one recording."""
    from g1_deploy.analysis import analyze
    from g1_deploy.hardware import MOTOR_NAMES

    metadata, summary, data = analyze(directory)
    if "tracking_t" not in data:
        raise SystemExit(f"{directory} is not a motion-tracking run")
    reference = data["reference_time"]
    at = data["tracking_t"]
    q = np.stack([np.interp(at, data["t"], data["q"][:, j]) for j in range(len(MOTOR_NAMES))], axis=1)
    target = np.stack([np.interp(at, data["target_t"], data["target_q"][:, j])
                       for j in range(len(MOTOR_NAMES))], axis=1)
    return dict(name=Path(directory).resolve().name, reference=reference, q=q, target=target,
                tilt=data["tracking_tilt_error"], residual=np.abs(q - target).mean(axis=1),
                clip=metadata.get("tracking", {}).get("clip") or metadata.get("tracking"),
                seconds=summary["recorded_seconds"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("hardware", help="Recording directory, drawn second and in the foreground.")
    parser.add_argument("simulation", help="Recording directory to compare it against.")
    parser.add_argument("--joint", default=JOINT, help="Joint drawn in the bottom panel.")
    parser.add_argument("--out", default="recordings/tracking_comparison.png")
    parser.add_argument("--max_tilt_error", type=float, default=0.8)
    args = parser.parse_args()

    from g1_deploy.bootstrap import ensure_cyclonedds

    ensure_cyclonedds()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from g1_deploy.hardware import MOTOR_NAMES

    if args.joint not in MOTOR_NAMES:
        raise SystemExit(f"unknown joint {args.joint!r}")
    column = MOTOR_NAMES.index(args.joint)
    runs = [("simulation", load(args.simulation)), ("hardware", load(args.hardware))]

    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for label, run in runs:
        axes[0].plot(run["reference"], run["tilt"], lw=1.4, label=f"{label}  ({run['name']})")
        axes[1].plot(run["reference"], run["residual"], lw=1.2, label=label)
        axes[2].plot(run["reference"], run["q"][:, column], lw=1.2, label=f"{label} measured")
    axes[2].plot(runs[1][1]["reference"], runs[1][1]["target"][:, column], ls=":", lw=1,
                 c="k", label="commanded (hardware)")
    axes[0].axhline(args.max_tilt_error, ls="--", c="k", lw=1, label="tilt guard")
    axes[0].set_ylabel("reference tilt error [rad]")
    axes[1].set_ylabel("mean |q - commanded| [rad]")
    axes[2].set_ylabel(f"{args.joint} [rad]")
    axes[2].set_xlabel("reference time [s] -- the clip's own clock, not wall time")
    axes[0].set_title("motion tracking: hardware against simulation")
    for axis in axes:
        axis.legend(fontsize=8)
    figure.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=110)

    print(f"[plot] {args.out}")
    for label, run in runs:
        end = run["reference"][-1]
        print(f"[{label:10s}] followed {end:.2f}s of the clip   peak tilt {run['tilt'].max():.3f} rad"
              f"   mean |q - commanded| {run['residual'].mean():.3f} rad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
