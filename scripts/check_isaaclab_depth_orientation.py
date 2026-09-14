# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Settle which way up and which way round Isaac Lab's depth frames come out. Numbers, no pictures.

This runs **inside Isaac Lab**, not in this repo's environment. Copy it to the IsaacLab checkout and
call :func:`check` with a constructed env, or paste the body into ``play.py`` after the env exists.

The question it answers: the depth render disagrees with both the analytic flat-ground profile and
with MuJoCo, and ``rough_29dof_depth_distill_env_cfg.py`` records that differencing the frame as-is
gives RMS 1.125 m against 0.042 m row-flipped. Its own docstring then notes that a flat scene cannot
distinguish a row flip from a 180-degree roll about the optical axis. Stairs appearing on the wrong
side says it is the roll -- but that was read off a plot, and a plot has its own orientation. This
reads the tensor.

Two independent checks, neither of which involves drawing anything:

* **Vertical.** On any ground the camera pitches down onto, the top of the image looks further than
  the bottom. If row 0 is nearer than the last row, rows run bottom-up.
* **Horizontal.** Needs asymmetry, and the terrain provides it: sample the same frame over several
  environments, correlate the left-minus-right depth imbalance against the *actual* terrain height
  to the robot's left and right, taken from the height scanner. A positive correlation means column
  0 is on the robot's left; negative means the image is mirrored.

The horizontal check is the one worth doing carefully, because it is the one a symmetric scene
cannot answer and the one a viewer can fake.
"""

from __future__ import annotations

import numpy as np


def check(env, steps: int = 40, sensor: str = "depth_camera",
          data_type: str = "distance_to_image_plane") -> dict:
    """Report the orientation of the depth frames this env produces.

    Args:
        env: A constructed Isaac Lab env (``env.unwrapped`` is used).
        steps: Control steps to sample. More is better for the horizontal check, which relies on the
            robots wandering into asymmetric terrain.
        sensor: Scene key of the depth camera.
        data_type: Annotator key to read.

    Returns:
        ``rows_top_down`` and ``cols_left_right`` as booleans where determined, plus the statistics
        they were decided on.
    """
    import torch

    unwrapped = env.unwrapped
    camera = unwrapped.scene[sensor]
    scanner = unwrapped.scene.get("height_scanner") if hasattr(unwrapped.scene, "get") else None

    top, bottom, left, right, ground_bias = [], [], [], [], []
    actions = torch.zeros((unwrapped.num_envs, unwrapped.action_space.shape[-1]), device=unwrapped.device)
    for _ in range(steps):
        env.step(actions)
        frame = camera.data.output[data_type]
        # (envs, h, w) or (envs, h, w, 1)
        depth = frame[..., 0] if frame.ndim == 4 else frame
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        valid = depth > 0.05

        def band(sl_h, sl_w):
            part, mask = depth[:, sl_h, sl_w], valid[:, sl_h, sl_w]
            total = mask.sum(dim=(1, 2)).clamp(min=1)
            return ((part * mask).sum(dim=(1, 2)) / total).cpu().numpy()

        h, w = depth.shape[-2], depth.shape[-1]
        top.append(band(slice(0, max(1, h // 6)), slice(None)))
        bottom.append(band(slice(h - max(1, h // 6), h), slice(None)))
        left.append(band(slice(None), slice(0, max(1, w // 4))))
        right.append(band(slice(None), slice(w - max(1, w // 4), w)))

        if scanner is not None:
            # Ground height to the robot's left minus its right, from the privileged scanner. The
            # scan is a flat list of points; split it on the y coordinate of the pattern.
            pts = scanner.data.ray_hits_w[..., 2] - unwrapped.scene["robot"].data.root_pos_w[:, 2:3]
            half = pts.shape[1] // 2
            ground_bias.append((pts[:, :half].mean(dim=1) - pts[:, half:].mean(dim=1)).cpu().numpy())

    top_m, bottom_m = np.concatenate(top), np.concatenate(bottom)
    left_m, right_m = np.concatenate(left), np.concatenate(right)

    print(f"\n=== vertical, {len(top_m)} samples ===")
    print(f"  top sixth    {top_m.mean():.3f} m")
    print(f"  bottom sixth {bottom_m.mean():.3f} m")
    rows_top_down = bool(top_m.mean() > bottom_m.mean())
    print(f"  -> row 0 is the {'TOP' if rows_top_down else 'BOTTOM'} of the image"
          f"   ({'matches' if rows_top_down else 'FLIPPED against'} MuJoCo and the analytic profile)")

    print(f"\n=== horizontal ===")
    print(f"  left quarter  {left_m.mean():.3f} m")
    print(f"  right quarter {right_m.mean():.3f} m")
    cols_left_right = None
    if ground_bias:
        bias = np.concatenate(ground_bias)
        imbalance = left_m - right_m
        if bias.std() < 1e-3:
            print("  terrain is symmetric over these samples -- cannot decide. Run on rough terrain,")
            print("  or more steps, so the robots stand somewhere with a left-right height difference.")
        else:
            # Higher ground on a side means *nearer* depth on the side of the image that shows it.
            corr = float(np.corrcoef(bias, imbalance)[0, 1])
            print(f"  terrain left-minus-right height spread {bias.std():.3f} m")
            print(f"  correlation(ground bias, image left-minus-right depth) = {corr:+.3f}")
            if abs(corr) < 0.2:
                print("  -> too weak to call. More steps, or rougher terrain.")
            else:
                cols_left_right = corr < 0
                print(f"  -> column 0 is the robot's {'LEFT' if cols_left_right else 'RIGHT'}"
                      f"   ({'matches' if cols_left_right else 'MIRRORED against'} MuJoCo)")
    else:
        print("  no height_scanner in the scene, so the horizontal check cannot run.")
        print("  Use the teacher env, which has one, or place a known asymmetric obstacle.")

    print("\n=== verdict ===")
    if rows_top_down and cols_left_right is True:
        print("  frames agree with MuJoCo. Whatever flipped the picture is in the viewer.")
    elif not rows_top_down and cols_left_right is False:
        print("  both axes are flipped: the frame is rotated 180 degrees. Fix it at the readout,")
        print("  not in the camera pose -- no rotation of a pure-pitch mount can mirror an image.")
    elif not rows_top_down:
        print("  rows only are flipped. That is a row-order convention, not a roll.")
    elif cols_left_right is False:
        print("  columns only are mirrored. A rotation cannot do that; look at the annotator readout.")
    return {"rows_top_down": rows_top_down, "cols_left_right": cols_left_right,
            "top": float(top_m.mean()), "bottom": float(bottom_m.mean()),
            "left": float(left_m.mean()), "right": float(right_m.mean())}


if __name__ == "__main__":
    print(__doc__)
    print("Import this and call check(env); it needs a constructed Isaac Lab env.")
