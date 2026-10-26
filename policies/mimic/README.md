# G1 motion tracking

These actors were trained with Newton MJWarp and the official Isaac 6.2
`Robots/Unitree/G1/g1_29dof.usd`, with both hand variants set to `None` (29 body joints).
The action-rate penalty weight is -0.01. `kick` preserves the accepted spin-kick checkpoint;
its SHA-256 is `201c57fc8f5bba999d71832effed9c09bf3815f3c7b4c23ac5ee9e01811c0f84`.
`cr7` uses the corrected `g1_cr7_grounded` reference and the final 30,000-iteration run.

| Directory | Training checkpoint | Clip duration | Original Newton full-clip trials | Local DDS/MuJoCo playback |
|---|---|---:|---:|---|
| `kick` | `model_29999.pt` | 9.26 s | 32/32 | reached clip end |
| `walk` | `model_19999.pt` | 7.98 s | 32/32 | reached clip end |
| `run` | `model_19999.pt` | 7.96 s | 32/32 | reached clip end |
| `cartwheel` | `model_29999.pt` | 4.70 s | 32/32 | reached clip end |
| `double-kong` | `model_29999.pt` | 7.16 s | **0/32** | tilt guard at 2.34 s |
| `speed-vault` | `model_29999.pt` | 3.90 s | 32/32 | reached clip end |
| `cr7` | `model_29999.pt` | 5.92 s | 32/32 | reached clip end |

Newton counts are from each final training job's fixed evaluation, preserved in `evaluation.json`.
The local column is one deterministic play per clip on 2026-10-23 using the existing
`unitree_mujoco/unitree_robots/g1/scene_29dof.xml`, 2 ms physics, 50 Hz control and the 0.8 rad
reference-tilt guard. This MJCF is a different asset from the training USD. Reaching the clip end
does not establish tracking accuracy or hardware readiness. In particular, retain `double-kong`
as an unsuccessful experiment, not a learned full motion.

## Package contents

Each motion directory contains:

- `model_*.pt`: original RSL-RL checkpoint, including training state; not loaded directly by the loop.
- `policy.pt`: deterministic TorchScript actor; inference runs on the CPU.
- `contract.json`: exact joint names, observation order, default pose, per-joint action scales,
  nominal Kp/Kd, 50 Hz period, asset provenance and policy/reference hashes.
- `motion.npz` and `motion.json`: matching reference trajectory and its named-joint/body metadata.
- `evaluation.json`: original final Newton evaluation, including termination and joint-limit diagnostics.
- `validation.json`: observation/action/target comparison against the pinned training environment.
- `manifest.json`: source checkpoint/reference paths and SHA-256 hashes for the package files.

Export validation used 256 inputs per actor and found zero eager/TorchScript difference.
Deployment parity used five reference phases per motion, including perturbed joint states:
maximum absolute errors across all seven were 2.11e-7 for observations, 1.91e-6 for actions
and 1.08e-6 rad for joint targets. These checks test interface equivalence, not real-world dynamics.

## Play a clip

From the repository root, with the usual deployment environment installed:

```bash
# macOS viewer; Linux uses python and --interface lo
mjpython scripts/run_policy_loop.py --sim sync --viz \
  --tracking policies/mimic/walk --domain_id 81 --interface lo0
```

Use `--xml /path/to/scene_29dof.xml` if the MuJoCo asset is elsewhere. Each invocation starts a fresh
reference at frame zero and plays once; `--duration inf` still stops at the end unless
`--hold_at_end` is set. Close the viewer
to stop a macOS rehearsal early. Do not share this DDS domain with a second simulator.

The hardware entry point is the existing `--real` loop, for a compatible 29-DoF robot:

```bash
python scripts/run_policy_loop.py --real \
  --tracking policies/mimic/walk --stand_policy policies/mimic/kick \
  --interface <robot-network-interface> --domain_id 0
```

This interface has not been exercised on hardware. The loop requires `mode_machine=5`; it refuses
the locked-waist 27-DoF variant. With `--stand_policy`, `go` engages a live kick actor at fixed
reference frame zero; `go2` blends into the selected motion actor over one second before starting
the clip. The standing posture has bent knees, and both actors use the same nominal Kp/Kd.
This mode replaces the passive ramp/alignment hold. On completion or abort, it uses the existing
damp-down sequence; it does not switch to a locomotion policy. With `--hold_at_end`, the motion actor keeps tracking its final
frame until the duration limit or an operator/guard stop. Use `--stand_kp` only with the passive
pose-hold mode, without `--stand_policy`. `--imu_tilt`, `--action_limit` and `--clamp_targets` retain their
existing meanings, but adding action/target clipping changes the trained controller's outputs.

## Observation and playback contract

The actor consumes one 154-value frame in this order:

| Values | Source |
|---|---|
| Reference joint positions, 29 | `motion.npz`, absolute angles |
| Reference joint velocities, 29 | `motion.npz`, rad/s |
| Relative torso orientation, 6 | `R_robot_torso.T @ R_reference_torso`, first two columns flattened row-major |
| Pelvis angular velocity, 3 | LowState pelvis IMU gyro, rad/s |
| Joint positions, 29 | Encoder positions minus training default positions |
| Joint velocities, 29 | Encoder velocities minus training default velocities |
| Previous actions, 29 | Previous actor output, before joint scaling |

Torso orientation is computed from the calibrated pelvis IMU and the three waist encoders:
`R_pelvis @ Rz(waist_yaw) @ Rx(waist_roll) @ Ry(waist_pitch)`. A single heading offset aligns
reference frame zero to the robot; later reference turns are preserved. The reference quaternion
format is WXYZ. Targets are `q_default + action_scale * action`, then mapped to motors by name.
The starting reference pose is distinct from the training default used for observations and targets.

The reference clock uses simulation time in synchronous simulation and monotonic time on hardware;
an overdue cycle skips elapsed reference frames rather than slowing the motion. A reference with
N frames commands N-1 control intervals. No phase resampling or simulator-style reset occurs.
The actor has no observation history, base linear velocity, anchor position or critic inputs.

Tracking stops when the torso gravity direction differs from the reference by more than
`--tracking_max_tilt_error` (default 0.8 rad), and returns a nonzero status for that stop. This permits
intended inversions but does not recreate training's privileged body-position termination tests.

## Recording

Hardware runs record by default. Add `--log_dir recordings` for a local rehearsal. In addition to
DDS messages and actor inputs/outputs, the `tracking` stream records reference frame/time, torso
tilt error, action and joint target. Offline analysis preserves the 58-value reference command
separately from locomotion velocity commands and plots reference tilt error in the overview.
