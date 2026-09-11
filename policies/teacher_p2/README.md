# teacher_p2 — NOT runnable on the robot

`p2` is the w100 gait with the leg power penalty at -5e-4: success 0.997, leg power 237 -> 131 W,
left/right airborne ratio 0.63 -> 1.04 (this is the arm that fixed the limp). It is here because it
is a good gait, not because it can be deployed.

**It reads 190 inputs the robot cannot produce.** The actor takes 328 values:

| term | dim | on the robot? |
|---|---|---|
| base_lin_vel | 3 | **no** — needs a base-velocity estimator |
| base_ang_vel | 3 | yes (IMU) |
| projected_gravity | 3 | yes (IMU) |
| velocity_commands | 3 | yes |
| joint_pos / joint_vel / actions | 43 each | yes |
| height_scan | 187 | **no** — a 1.6 x 1.0 m terrain map |

`g1_deploy/core.py` builds exactly the six observable terms (`OBS_DIM = history * (3+3+3+3N)`), so
loading this file raises on the shape, which is the correct behaviour.

Two ways to get this gait onto the robot, both of which cost a training run:

1. **Distil it into a proprioception-only student** — same DAgger setup as `depth_student_cl` with
   the depth group removed, 5 history frames, teacher = this checkpoint. Deployable, and blind: on
   rough terrain it has to feel the ground rather than see it.
2. **Retrain with a deployable actor** — drop `base_lin_vel` and `height_scan` from the policy
   observation group and leave them in the critic's. Asymmetric actor-critic, no teacher-student
   gap, one 6000-iteration run.

Regenerate with:

```bash
uv run python scripts/export_teacher_policy.py \
  --task Isaac-Velocity-Rough-G1-29Dof-AirTime100-Power2 \
  --checkpoint ~/g1_assets/sweeps/g1-power/p2/model_5999.pt \
  --out_dir <dir> physics=newton_mjwarp \
  env.scene.robot.spawn.usd_path=$HOME/g1_assets/ladder/g1_a1_feet.usda
```
