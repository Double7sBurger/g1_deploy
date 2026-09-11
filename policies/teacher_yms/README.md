# teacher_yms — NOT runnable on the robot (a student is training)

`yms` = the per-joint action scale (`0.25 * effort / stiffness`) plus left-right mirror
augmentation. It is the best gait this line has produced, measured on flat ground under a pinned
0.8 m/s straight command, three seeds:

| | success | pelvis roll | L/R airborne | worst joint asym |
|---|---|---|---|---|
| yms s42/s43/s44 | 0.989 / 0.994 / 0.987 | +0.12 / +0.52 / +0.31 deg | 1.029 / 1.041 / 1.065 | 0.68 / 0.70 / 0.90 deg |
| ms control | 0.996 | -2.79 deg | 0.713 | 3.0 to 5.6 deg |

**It reads 190 inputs the robot cannot produce.** The actor takes 328 values: `height_scan` (187) is
a 1.6 x 1.0 m terrain map and `base_lin_vel` (3) needs a base-velocity estimator. `g1_deploy/core.py`
builds only the six observable terms, so this file fails on the input shape, which is correct.
`contract.json` records it as `deployable: false`.

**The deployable version is being distilled**: OSMO `g1-distill-yms-1`, 4000 iterations, teacher =
this checkpoint, student = the six terms with five history frames (690 values), no camera. It lands
as `policies/student_yms/` and is driven with:

```bash
cd ~/workspace/g1_deploy && export PYTHONPATH=$PWD
# terminal 1
python scripts/run_sim_loop.py --domain_id 41 --hoist_s 4.0 --viz --policy_physics g1_29dof
# terminal 2
python scripts/run_policy_loop.py --sim sync --teleop \
  --policy policies/student_yms/policy.pt --policy_physics g1_29dof \
  --vx 0.4 --duration inf --domain_id 41 --interface lo
```

Two things to check on arrival, because neither transfers automatically:

* **Symmetry.** The mirror augmentation is a property of how the *teacher* was trained; the student
  is a plain DAgger fit and does not get it. Measure pelvis roll and the airborne share ratio on the
  student, not on the teacher's numbers above.
* **The actuator is implicit.** Both this line and the teacher drive the joints through the solver,
  where the hardware's controller computes `tau = kp(q*-q) - kd qdot` explicitly. `g1-explicit-1` is
  testing whether these gains survive that change; until it reports, treat any hardware oscillation
  as expected rather than as a bug in the policy.
