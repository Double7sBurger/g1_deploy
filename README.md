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
  teleop.py       keyboard command source — terminal keys, and viewer keys in --sim sync
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

**Take `unitree_sdk2py`'s `cyclonedds==0.10.2` pin literally.** Installing it with `--no-deps` against
a newer CycloneDDS looks fine and is not: 0.11+ enforces XTypes type-consistency at match time, and
the SDK's Python IDL hashes differently from the robot's C++ IDL, so readers silently never match.
Measured against a real G1 on `rt/odommodestate`:

```
ours (Python)  [MINIMAL fce8a80b52eff039d4683fc3ad80]/[COMPLETE 4014d13a03d68f246f38f1449cb2]
robot (C++)    [MINIMAL 9fa6ccdc5cafdeaab8a762e31044]/[COMPLETE e59749630fff75cc170afb238783]
```

The failure mode is nasty because **sim-to-sim still passes**: both ends are the same Python
serialiser, so the hashes agree and everything works right up until the hardware. Symptom on the
robot is `subscription_matched.total_count == 0` on every complex type, while trivially-shaped
topics like `rt/wirelesscontroller` do match and make it look like the network is fine. There are
macOS arm64 wheels for 0.10.2, so no source build is needed.

DDS domain ids must satisfy `7400 + 250 * id < 65536`, i.e. stay under about 230; above that
CycloneDDS fails to open its discovery socket with a port "out of range".

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

On macOS pass `--interface lo0`; `lo` is the Linux name and CycloneDDS will not find it. `--viz` has
to be launched with `mjpython`, which MuJoCo requires for a viewer on macOS.

## Driving it from the keyboard

`--vx/--vy/--heading` are otherwise read once at startup. `--teleop` makes them live, in every mode
including `--real` on hardware:

```bash
python scripts/run_policy_loop.py --sim sync --teleop \
  --policy policy.pt --vx 0.5 --duration inf --domain_id 41 --interface lo0
```

```
w/s  forward +/- 0.1 m/s      q/e  turn the heading target +/- 15 deg
a/d  left/right +/- 0.1 m/s   space  stop   r  face the current heading   x  quit   ?  help
```

Arrow keys work too. Keys are read from the terminal in every mode; with `--sim sync --viz` the
viewer window accepts them as well, since the controller owns it.

`--duration inf` runs until you stop it. Prefer `x` or closing the viewer window over Ctrl-C: under
`mjpython` the script does not run on the main thread, so Python hands `KeyboardInterrupt` to
mjpython's event loop and the control loop never sees it.

**`--sim sync` cannot share a DDS domain with anything else.** It brings its own simulator, so a
`run_sim_loop.py` left open in another terminal puts two robots on the same `rt/lowstate`. Two 500 Hz
publishers overrun the SDK reader's queue, the reliable writer back-pressures, and the loop wedges
inside `dds_write` in C -- where Ctrl-C cannot reach it either. The run now probes the domain for a
second before starting and refuses rather than wedging, but give the two modes different
`--domain_id` values and the question does not come up.

The command is clamped to the ranges training sampled: `vx` 0 to 1.0 m/s, `vy` -0.5 to 0.5 m/s.
**There is no reverse gear** -- `UniformVelocityCommandCfg.ranges.lin_vel_x` starts at 0, so a
backward command is extrapolation, not a missing feature.

`q`/`e` move the *heading target*, not a yaw rate, because that is what the policy reads: this task
trains with `heading_command=True`, so the third command element is a proportional controller on
heading error recomputed every step. A teleoperated run therefore stays on the same distribution a
fixed `--heading` run is on. `r` adopts the heading the robot is currently at, which zeroes the error
and stops the turn.

Note that `space` is not a way to make the robot stand: at `vx = 0` this policy falls in about 2 s,
for the asset reason below.

## On the robot

### The stop, first

**The G1 has no hardware emergency stop.** Unitree's documented one is a remote combination that asks
the *factory* controller to damp — and once this program owns `rt/lowcmd`, relying on the factory
layer still listening is a bet, not a design. So the abort is handled here: every loop that commands
the robot parses the remote out of `rt/lowstate` and damps down on it. This is what Unitree's own
`unitree_rl_lab` does (`LT + B` → its Passive state, kp=0 kd=3).

`g1_deploy/remote.py` aborts on **`L2 + B`**, **`L1 + A`**, or **`select`**, during the ramp, the
hold and the policy loop alike.

> **The remote's mapping changed at Motion Control 8.2.0.0.** Damping is `L2 + B` on 8.2.0.0 and
> later, `L1 + A` before it — and on that older firmware `L1 + B` is *zero torque*, which drops a
> hanging robot with no damping at all. Check the sticker that came with your controller. All three
> abort combos are accepted here so that guessing the firmware wrong is not what decides whether the
> stop works.

The combos that must **not** abort — `L2 + A` (debug-mode confirm pose), `L2 + R2` (enter debug
mode), `start` — are tested to be ignored.

### Bring-up order

Composite of Unitree's [Quick Start](https://support.unitree.com/home/en/G1_developer/quick_start)
and their own RL recipe in `unitree_rl_gym/deploy/deploy_real/README.md`:

1. Hang the G1 on the protective rack, wheels locked. Every relevant Unitree page says to.
2. Power on. Wait ~1 min for initialisation (all joints go zero-torque), then another 30 s.
3. `L2 + B` → damping. LED solid **orange**.
4. `L2 + R2` → debug mode. LED solid **yellow**. Confirm with `L2 + A` (diagnostic pose) then
   `L2 + B` back to damping; repeat `L2 + R2` if `L2 + A` did nothing. In debug mode the joints are
   in the damping state — they sag under gravity but resist.
5. Wired ethernet. The robot's locomotion computer is **192.168.123.161**; set your machine static in
   `192.168.123.0/24` (Unitree's docs say 192.168.123.99), mask 255.255.255.0, and `ping
   192.168.123.161`. `ifconfig` gives the interface name to pass to `--interface`. **DDS domain is
   0.**
6. Read-only check first — confirm `mode_machine == 5` (29-DoF) and that `tick` advances, before
   anything writes `rt/lowcmd`.
7. `--dry_run`, hoisted, feet clear. See below.
8. Only then the policy, and lower the hoist so the feet take weight *before* engaging it.

Undocumented and worth knowing: **there is no published number for the motor watchdog** — how long
`rt/lowcmd` may stop arriving before the drives cut out. Unitree's docs do not state one and their
repos contain no such constant. `unitree_mujoco` has no equivalent either, so the rehearsal here does
not exercise it. That is why there is no operator prompt between the ramp and engaging the policy.

### Step 1: the dry run, hoisted

`--dry_run` stops after the ramp. It releases the factory controller, interpolates to the policy's
start pose, holds it while you look at the robot, reports per-joint tracking, and damps down. **The
policy is never engaged and the robot is never asked to balance**, so this is the step that proves
the joint order, the joint directions and the start pose while nothing can go wrong yet.

```bash
python scripts/run_policy_loop.py --real --dry_run \
  --policy policy.pt --policy_physics newton \
  --ramp_s 3.0 --hold_s 5.0 --domain_id 0 --interface eth0
```

The pose it ramps to is a slight crouch — hips -11.5°, knees +24.1°, ankles -13.2°, shoulders 20°
forward and 9° out, elbows 50°. Waist roll/pitch and the wrist pitch/yaw joints have no policy
counterpart and are held at 0 with real stiffness, not left limp.

What the tracking report is for: every motor answering and holding. One large row is a dead motor,
a joint against a stop, or a limit collision. Measured on hardware, hoisted with the feet clear, the
whole robot tracks to **0.036 rad (2°) worst, 0.012 rad mean** — the shoulders are the worst and the
unloaded ankles are fine. In the MuJoCo rehearsal the ankles instead sit ~0.17 rad short, because
there the hoist leaves them carrying load at kp 20; do not read the simulator's number as the target.

It does **not** check the joint ordering. `--policy_physics` orders the policy's own observation and
action vectors, while `build_default_pose()` is assembled by joint name — so the ramp target is
byte-identical under `physx` and `newton`, and a wrong backend is invisible here. What validates
that is the MuJoCo benchmark: a scrambled action order does not walk. And a small error means the
robot went where it was told, not that the pose is right — look at it.

### Step 2: the policy

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
