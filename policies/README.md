# What is in here, and which of it runs

Four students are deployable; the three `teacher_*` directories are not and say so in their own
contracts (`deployable: false`). Every number below is measured on flat ground under a pinned
0.8 m/s straight command, except `success_rate`, which is the training metric on rough terrain.

| directory | teacher | camera | success | single-stance | flight | pelvis roll | action scale |
|---|---|---|---|---|---|---|---|
| `ymsd` | yms 0.994 | depth | **0.963** | 0.899 | 0.008 | -0.65 deg | per-joint |
| `student_yms` | yms 0.994 | none | 0.918 | 0.898 | 0.010 | +1.14 deg | per-joint |
| `student_cw` | cw 0.985 | none | 0.859 | 0.935 | 0.011 | -0.94 deg | 0.5 |
| `ysud` | ysu 0.988 | depth | 0.829 | 0.850 | 0.003 | -0.21 deg | 0.5 |
| `depth_student_cl` | cl 0.884 | depth | 0.765 | - | - | - | 0.5 |

`single-stance` is the fraction of time on one foot; a walk is 0.85-0.94 and a hop is near zero.
All five walk.

## Running one

Blind students take a directory, not a file -- the contract beside the policy carries the action
scale, and the `ms` line replaced the blanket 0.5 with a per-joint table (0.11 on the hip, 0.625 on
the ankle). Driving such a policy at 0.5 asks the hip for four and a half times the angle it was
trained to ask for, and nothing errors.

```bash
cd ~/workspace/g1_deploy && export PYTHONPATH=$PWD
# terminal 1
python scripts/run_sim_loop.py --domain_id 41 --hoist_s 4.0 --viz --policy_physics g1_29dof
# terminal 2 -- blind
python scripts/run_policy_loop.py --sim sync --teleop --policy policies/student_yms \
  --policy_physics g1_29dof --vx 0.4 --duration inf --domain_id 41 --interface lo
```

For a depth student the simulator has to publish frames, and `--depth` takes the same directory:

```bash
# terminal 1
python scripts/run_sim_loop.py --domain_id 41 --hoist_s 4.0 --viz \
  --policy_physics g1_29dof --depth policies/ymsd --foot_plate
# terminal 2
python scripts/run_policy_loop.py --sim sync --teleop --depth policies/ymsd \
  --policy_physics g1_29dof --vx 0.4 --duration inf --domain_id 41 --interface lo
```

`--policy_physics g1_29dof` is not optional: it selects the 43-joint order these were trained on.
Keyboard: `w/s` speed, `a/d` lateral, `q/e` heading, space stop, `x` quit.

## What the camera is currently worth: nothing

Measured by blanking the depth and rerunning the same episodes:

| | with camera | blanked |
|---|---|---|
| `ymsd`, mixed terrain | 0.863 | 0.874 |
| `ymsd`, stairs up, level 9 | 0.935 | 0.900 |
| `ymsd`, stairs down, level 9 | 0.743 | **0.798** |
| `ysud`, mixed terrain | 0.977 | 0.969 |

The camera itself is fine -- 88% of pixels inside the 3 m range, the optical axis 47.6 degrees below
horizontal, stair structure visible in the frames (`~/g1_flat_video/ymsd_stairs_pov.mp4` shows the
robot and its own view side by side). The policy does not use it: blanking the depth changes the leg
actions by 0.30%. Treat these as proprioceptive policies that happen to carry a camera, and expect
nothing from the depth path failing.

## Which commands they actually follow

The training task pins `lin_vel_y = (-0.0, 0.0)`, so **no student was ever asked to strafe** --
`a`/`d` on the keyboard is not a supported input. Yaw was trained, but through
`heading_command=True` with `rel_heading_envs=1.0`: the yaw-rate command the policy saw was
`clip(0.5 * (heading - yaw), -1, 1)`, a value that decays to zero as the robot turns to face its
heading. A sustained yaw rate from a joystick is therefore a mild out-of-distribution input, and the
policy tracks it with a gain well under one.

`ymsd`, 64 envs on flat ground, command pinned for 350 steps, first 50 discarded:

| command | vx cmd -> got | vy cmd -> got | wz cmd -> got |
|---|---|---|---|
| forward 0.4 | 0.40 -> 0.26 | - | - |
| forward 0.8 | 0.80 -> 0.65 | - | - |
| strafe +0.5 | - | +0.50 -> **-0.01** | - |
| strafe -0.5 | - | -0.50 -> **+0.13** | - |
| turn +0.5 | - | - | 0.50 -> 0.08 |
| turn -0.5 | - | - | -0.50 -> -0.06 |
| turn +1.0 | - | - | 1.00 -> 0.42 |
| fwd 0.5 + turn 0.5 | 0.50 -> 0.33 | - | 0.50 -> 0.37 |

Two things to drive it by:

- **Turn while walking, not in place.** At the same yaw command the tracked rate is 0.37 while
  walking forward versus 0.08 standing -- 74% of command versus 16%.
- **Ignore the lateral axis.** Strafing is either dead (+0.5 -> -0.01) or backwards (-0.5 -> +0.13);
  `depth_student_w100` behaved the same way. To move sideways, turn and then walk.

Measured with `scratchpad/cmd_response.py`.

## ymsd vs ysud: one number apart

Their contracts are byte-identical except `action_scale`, and their teachers are the same
environment (`AirTime100WaistWarmup` + the -10 pelvis-height penalty + left/right mirror
augmentation). `ysud` uses the blanket 0.5; `ymsd` uses mjlab's per-joint rule
`0.25 * effort_limit / stiffness`:

| joint group | ymsd | ysud | ymsd/ysud |
|---|---|---|---|
| hip pitch | 0.110 | 0.5 | 0.22x |
| hip roll / yaw | 0.147 | 0.5 | 0.29x |
| knee | 0.174 | 0.5 | 0.35x |
| **ankle pitch / roll** | **0.625** | 0.5 | **1.25x** |
| waist roll / pitch | 0.063 | 0.5 | 0.13x |
| shoulder / elbow | 0.156 | 0.5 | 0.31x |

The ankle carries the lowest stiffness on this robot (kp 20 against 200 at hip pitch and knee), so
mjlab's rule hands it the largest angular range. Inside `ymsd` the ankle has 5.7x the per-unit
angular authority of the hip; inside `ysud` the ratio is 1.0. `ymsd` therefore walks off its ankles
and `ysud` off its hips, and everything below follows from that.

Flat ground, pinned straight command, 64 envs x 30 s, `body_symmetry.py`:

| | single-stance | double-support | flight | pelvis pitch | pelvis height | falls |
|---|---|---|---|---|---|---|
| `ymsd` @ 0.1 m/s | **0.045** | 0.946 | 0.009 | +1.46 deg | 0.728 m | 1/65 |
| `ymsd` @ 0.5 m/s | 0.889 | 0.106 | 0.005 | -2.75 deg | 0.727 m | 0/64 |
| `ysud` @ 0.1 m/s | 0.601 | 0.396 | 0.002 | +1.68 deg | 0.788 m | 0/64 |
| `ysud` @ 0.5 m/s | 0.837 | 0.161 | 0.002 | +0.24 deg | 0.785 m | 0/64 |

At 0.1 m/s `ymsd` stops stepping -- 95% of the time on both feet -- which is why it goes over
backwards on hardware at low speed: with no swing leg there is no recovery step. `ysud` keeps
stepping at the same command. `ysud` also stands 6 cm taller at every speed and holds its pelvis
within a quarter degree of level at 0.5 m/s where `ymsd` sits 2.8 degrees off.

Two hardware behaviours are **not** reproduced in simulation and are therefore deploy-side:

- **`ysud`'s visible pelvis tilt.** Measured roll is 0.3-0.9 deg and pitch 0.2-1.7 deg in sim. A
  constant tilt on the robot that is not in sim points at the IMU-to-pelvis alignment: the policy's
  only attitude input is `projected_gravity`, so a mounting offset is held as a body tilt one for
  one. Check what `projected_gravity` reads with the robot held level against sim's `[0, 0, -1]`.
- **`ysud`'s yaw drift.** Sim drift under a straight command is -0.005 rad/s. Nothing in the
  observation set closes a loop on *heading* -- `base_ang_vel` is a rate, so any gyro bias
  integrates without correction. Both policies drift; only a heading estimate fixes it.

## Taking out an IMU mounting offset

The policy's only attitude input is `projected_gravity`, and on hardware that comes from the IMU's
own quaternion (`core.py`, `observe`). If the IMU's zero sits a few degrees off the pelvis frame the
policy trained in, then with the robot actually level the policy reads "tilted by theta" and answers
by holding the body at -theta. The error is constant, invisible in simulation, and reads as a robot
that walks permanently leaning. Nothing in the pipeline measured it until now -- `--align` is only a
hold-until-you-type-the-word gate, not a calibration.

Measure it, robot standing still and level in its default pose on level ground:

```bash
python scripts/check_robot.py --domain_id 0 --interface en6
```

It prints the full gravity vector and the implied angles; `gravity_z` alone cannot see this, since
five degrees off level still reads -0.996. If the angles are not ~0 it also prints the flag to paste:

```bash
python scripts/run_policy_loop.py --real ... --imu_tilt 3.00,-2.00
```

`set_imu_tilt` rotates both `projected_gravity` and the gyro into the pelvis frame, so the two
attitude terms stay in the same frame as each other. It is identity until called -- runs without the
flag behave exactly as before. Verified numerically: with a 3 deg pitch / -2 deg roll offset
injected, a level robot reads (+3.00, -2.00) uncorrected and (0.00, 0.00) corrected, while a robot
genuinely pitched +5 deg still reads +5.00 after correction.

## `mjd` -- the student of the MuJoCo-aligned environment

Same contract as `ysud` in every field except the task name, so it runs with the same command; only
`--depth policies/mjd` changes. Trained under `mj`: the hardware torque ceilings (88/139/50/25,
and an ankle of 50 rather than Isaac Lab's 20), 0.2 N*m of joint dry friction, 0.05 of passive
damping, a ground friction of 1.0, and MuJoCo's torso and waist inertials.

Flat ground, pinned command, 64 envs, `body_symmetry.py`:

| speed | single-stance | flight | mean \|pelvis roll\| | pelvis pitch | waist roll | falls |
|---|---|---|---|---|---|---|
| 0.8 m/s | **0.928** | 0.006 | 1.75 deg | -3.35 deg | +0.71 deg | 0 |
| 0.5 m/s | 0.778 | 0.005 | 1.01 deg | -4.39 deg | +0.77 deg | 0 |
| 0.1 m/s | **0.031** | 0.004 | 0.58 deg | -1.18 deg | +1.13 deg | 1 / 65 |

**At 0.8 m/s it is the best-walking student here** -- single-stance 0.928 against `ymsd`'s 0.899 and
`ysud`'s 0.850 -- and it holds a straight line (vy -0.056 m/s, wz +0.050 rad/s).

**Below roughly 0.3 m/s it stops stepping**, the same failure `ymsd` has on hardware: 96.5% double
support at 0.1 m/s, so a backward lean has no recovery step. `ysud` is still the only student that
keeps stepping there (0.601). Drive `mjd` at 0.5 m/s and up, or keep `ysud` for slow work.

It did not clear the video bar (training `success_rate` 0.897 against the 0.95 required), so there is
no all-terrain clip. `~/g1_flat_video/mjd_vs_ysud_waist.mp4` puts it beside `ysud` at 0.5 m/s, front
view, for the waist question.
