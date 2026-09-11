# teacher_cw — NOT runnable on the robot (a student is training)

`cw` = randomization + the stronger push + waist L2 -1.0 + the leg power penalty at -5e-4. It is the
arm that walks evenly: airborne-share ratio 1.02 against t1's 0.47, leg power 237 -> 132 W,
success 0.985. That is why it is wanted here.

**It reads 190 inputs the robot cannot produce** — the same 328-value actor as `teacher_p2`:
`height_scan` (187) is a 1.6 x 1.0 m terrain map and `base_lin_vel` (3) needs a base-velocity
estimator. `g1_deploy/core.py` builds only the six observable terms, so this file fails on the input
shape, which is the correct behaviour. `contract.json` records this as `deployable: false`.

**The deployable version is being distilled**: OSMO `g1-distill-cw-1`, 4000 iterations, teacher =
this checkpoint, student = the same six terms with five frames of history (690 values), no camera.
It lands as `policies/student_cw/` and is driven with:

```bash
python scripts/run_policy_loop.py --sim sync --teleop \
  --policy policies/student_cw/policy.pt --policy_physics g1_29dof \
  --vx 0.4 --duration inf --domain_id 41 --interface lo
```

One caveat to check on arrival: this student is **blind**. It has no camera and no height scan, so
on anything but near-flat ground it is feeling the terrain through five frames of proprioception.
Rehearse on the flat XML first.

Regenerate this export with:

```bash
uv run python scripts/export_teacher_policy.py \
  --task Isaac-Velocity-Rough-G1-29Dof-AirTime100-Waist1-Power2 \
  --checkpoint ~/g1_assets/sweeps/g1-combo2/cw/model_5999.pt \
  --out_dir <dir> physics=newton_mjwarp \
  env.scene.robot.spawn.usd_path=$HOME/g1_assets/ladder/g1_a1_feet.usda
```
