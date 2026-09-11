# depth_student_cl — fixed camera (g1-depth-distill-9, 2026-09-09)

Distilled from `cl` (= cw + hip-pitch L2 -0.15), 4000 iterations, `success_rate` 0.765 against the
teacher's 0.884. Camera: `convention="world"`, `rot=(0.4035, 0, -0.9150, 0)`, forehead mount
(0.0576, 0.0175, 0.4299) on `torso_link`, 64x38, 3 m range, 3-frame stack oldest-first.

## The camera barely matters, on this student and the one before it

Measured by evaluating the student twice per step on the same proprioception, once with the real
depth frame and once with "nothing in range":

| student | camera influence on the action | depth pixels in range |
|---|---|---|
| this one (fixed camera) | **0.70%** of action magnitude | 88.1% |
| `depth_student_cl_prefix_camera` | 0.38% | 88.7% |

Fixing the camera roughly doubled its influence and it is still under one percent. The terrain is
solvable blind, so the student learns to ignore the image. Do not treat this as a vision policy;
treat it as a proprioceptive policy that carries a camera. It also means the depth path failing --
no frames, a dead camera, all-invalid returns -- degrades it very little, which is convenient in
rehearsal and misleading in evaluation.

## Predecessor

`policies/depth_student_cl_prefix_camera/` is the export that was here until 2026-09-09 23:00. Its
weights are from the run *before* the camera convention was fixed, while its `contract.json` already
described the fixed mount -- so it advertised a camera geometry it had never trained under. Kept for
comparison, not for use.
