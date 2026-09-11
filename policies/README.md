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
