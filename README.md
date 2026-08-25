# g1-deploy

Run an Isaac Lab G1 locomotion policy on a real Unitree G1 — or on MuJoCo over the same DDS topics,
so the rehearsal and the real thing are the same code path.

The robot-facing half needs **numpy, torch and `unitree_sdk2py`**. No simulator, no Isaac Lab, no
GPU: the policy is a 208K-parameter MLP that runs in 0.016 ms on a single CPU thread, 0.1% of the
20 ms control period. A laptop or the G1's own computer is enough.

## Layout

```
g1_deploy/
  core.py         the deployment contract: joint order, observation layout, PD gains, default pose
  controller.py   reads rt/lowstate, writes rt/lowcmd — the only part that talks to the robot
  hardware.py     power-on to policy control, and putting the robot down safely
  bootstrap.py    CycloneDDS library path fix (Linux dev boxes only; a no-op elsewhere)
  benchmark.py    shared command schedule and scoring, used by every runner
  sim/            MuJoCo stand-in for the robot — optional, the only part needing mujoco
scripts/
  run_policy_loop.py   entry point, for both the robot and the simulator
  run_sim_loop.py      free-running simulator that answers on rt/lowstate / rt/lowcmd
  benchmark_*.py       quantitative comparison runners
  record_video.py      offscreen recording of a rehearsal
```

## Install

```bash
git clone <this repo> && cd g1_deploy
pip install -e .                 # robot only
pip install -e ".[sim,video]"    # plus MuJoCo rehearsal and recording

# unitree_sdk2py is not on PyPI
git clone https://github.com/unitreerobotics/unitree_sdk2_python
pip install -e unitree_sdk2_python
```

You also need an exported TorchScript policy (`policy.pt`) from `isaaclab play`.

## Rehearse before touching hardware

Two terminals. The simulator answers on the same DDS topics the robot does, so the controller cannot
tell them apart.

```bash
# terminal 1 — a stand-in robot, held up for 4 s like a gantry, with a viewer
python scripts/run_sim_loop.py --domain_id 41 --hoist_s 4.0 --viz

# terminal 2 — the exact hardware sequence, minus the factory-controller release
python scripts/run_policy_loop.py --real --skip_release_mode \
  --policy policy.pt --policy_physics newton --vx 0.5 --duration 20 --ramp_s 3.0 --domain_id 41
```

The simulator does **not** reset between runs. Restart terminal 1 for each attempt.

## On the robot

`--real` releases the factory motion controller, ramps from the robot's current pose to the policy's
start pose, refuses to engage unless that ends upright, and damps down on every exit path.

```bash
python scripts/run_policy_loop.py --real \
  --policy policy.pt --policy_physics newton \
  --vx 0.0 --duration 10 --ramp_s 3.0 \
  --domain_id 0 --interface eth0        # macOS: en0/en5/en7, find it with ifconfig
```

Use a wired link. A 50 Hz control loop over WiFi is not something to find out about on the robot.

**The robot must be supported** — hung or held — through the ramp. It cannot hold its own default
pose at these gains: MuJoCo drops it in 1.46 s and Isaac Lab in about 1.5 s, because ankle kp is 20
N·m/rad and that is not enough to balance statically. Lower it only once the policy is running.

There is exactly one operator prompt and it comes before anything moves. There is deliberately none
between the ramp and engaging the policy: nothing sends `rt/lowcmd` while a prompt is waiting, so the
motor watchdog times out and drops the robot.

## Reading the output

`N/M control steps would have been clamped` is expected and is **not** a reason to pass
`--clamp_targets`. The policy drives the soft ankles with deliberately far-away targets — ankle kp is
20, so a target 0.9 rad away is simply how it asks for 18 N·m — and every *achieved* ankle angle
stays inside the mechanical travel. Clamping cuts the same command to about 4 N·m. Turn it on only if
the joints are actually reaching their stops.

## Known limitation: the training asset is a superseded G1

Isaac Lab's shipped `g1.usd` and `g1_minimal.usd` (6.0 and 6.1) match
`g1_unitree_deprecated.urdf`, not Unitree's current one. Against the real robot:

| | old (training) | current (URDF / MJCF / hardware) |
|---|---|---|
| hip roll joint z | 0 | −0.0305 m |
| pelvis → ankle | 0.6865 m | 0.7429 m |
| total mass | 32.24 kg | 35.11 kg |

Same joint command, 6 cm taller stance. Measured on the 45-episode hold suite with one checkpoint:

| MuJoCo geometry | success | tracking error | standing |
|---|---|---|---|
| current (≈ hardware) | 75.6% | 0.364 m/s | 0% |
| old (≈ training) | 97.8% | 0.118 m/s | 100% |

So expect a policy trained on the shipped asset to walk but not stand, and to undershoot commanded
speed. `g1_deploy/sim/align.py` can rewrite MuJoCo's legs to the old geometry, which reproduces Isaac
Lab exactly (0.000 mm forward kinematics across six joint configurations, four held out) — useful for
explaining the gap, not for predicting hardware. The real fix is to retrain on a current asset;
NVIDIA's `i4h-asset-catalog` ships one (`Robots/UnitreeG1/g1_29dof_wholebody_dex3/`) whose leg joints
match the current URDF to 0.000 mm.
