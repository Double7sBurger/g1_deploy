# Working in this repo

An Isaac Lab G1 locomotion policy running on a real Unitree G1 29-DoF over DDS, or on MuJoCo over the
same topics. `README.md` is the user-facing document and is accurate; this file is the things that
will cost you an hour if you learn them by hitting them.

## Environment

Conda env **`deploy`** (Python 3.10). Companion checkouts the code expects by absolute path:

| path | why |
|---|---|
| `~/workspace/unitree_mujoco` | `unitree_robots/g1/scene_29dof.xml`, the default `--xml` |
| `~/workspace/unitree_sdk2_python` | `unitree_sdk2py`, installed `-e` |
| `~/workspace/unitree_rl_gym` | reference only — source of the remote-controller bit layout |

**`cyclonedds` must be 0.10.2.** `unitree_sdk2py` pins it and the pin is real. A newer CycloneDDS
(11.x) enforces XTypes type-consistency at match time and the SDK's Python IDL hashes differently
from the robot's C++ IDL, so readers silently never match. This passes every sim-to-sim test — both
ends share the Python serialiser — and fails only against hardware. Never install past the pin with
`--no-deps`. Full write-up with the measured type hashes is in `README.md`.

macOS specifics: pass `--interface lo0` (`lo` is the Linux name and CycloneDDS rejects it), and
`--viz` requires launching with `mjpython`. DDS domain ids must satisfy `7400 + 250*id < 65536`, so
stay under ~230 — above that CycloneDDS fails to open its discovery socket.

## The robot

Wired on **`en6`**, static `192.168.123.222`. Locomotion computer `192.168.123.161` (all the robot's
DDS participants are there), PC2 `192.168.123.164` (`unitree`/`123`). **DDS domain 0.**

Bring-up order and the remote-controller combos are in `README.md`. Two things worth repeating here:

- **There is no hardware emergency stop.** `g1_deploy/remote.py` is the stop: every loop that
  commands the robot parses the remote out of `rt/lowstate` and damps down on `L2+B` / `L1+A` /
  `select`. The mapping changed at Motion Control 8.2.0.0, which is why all three are accepted.
- **In debug mode the motion switcher reports a stale mode name.** Measured:
  `CheckMode() -> {'form': '0', 'name': 'ai'}` while `rt/lowcmd` is provably silent. So
  `hardware.take_lowcmd()` measures the topic instead of trusting the name — believing the name
  makes `release_motion_mode()` spin until it times out and the run dies before it starts.

## Vision (depth students)

`policies/<name>/` holds `policy.pt` + `contract.json`. The contract is authoritative and was dumped
from a live env with domain randomization **off** — a randomized read returns one sample of each gain
band, not the nominal (hip kp 146.7 instead of 200).

Pipeline, all of it verified on this machine:

| piece | where |
|---|---|
| camera-side crop + downsample + send | `scripts/depth_publisher.py`, runs on PC2 |
| wire format, receiver, staleness | `g1_deploy/depth_link.py` |
| preprocessing + policy wrapper | `g1_deploy/depth.py` |
| MuJoCo depth rendering | `g1_deploy/sim/depth_camera.py` |
| closed loop without hardware | `scripts/benchmark_depth_mujoco.py` |
| watch the observation | `scripts/view_depth.py` |
| Isaac Lab terrain for MuJoCo | `scripts/make_terrain.py` |

- **Crop before downsampling.** The D435i here streams 89.6 x 58.7 deg; training was 87.0 x 58.8.
  Resizing the full frame is a 3% horizontal compression that nothing downstream can detect — the
  array is 38x64 either way.
- **Run the camera above 50 Hz.** The driver's default 848x480 profile is 30 Hz; 90 is available on
  USB3. At 30 the policy sees repeats and the depth history spans more wall time than the 3 x 20 ms
  it trained on.
- **uint16 millimetres on the wire, not float32 metres.** That is the sensor's native format, and
  macOS caps a UDP datagram at 9216 bytes (`net.inet.udp.maxdgram`) — 64x38 float32 is 9748 and fails
  to send.
- **`--viz` needs `mjpython`** and coexists with the offscreen depth renderer, at a cost: 2887 frames
  received without it, 1033 with.
- **`view_depth.py` and the control loop contend for the UDP port.** Run one at a time.

## Terrain

`make_terrain.py` calls Isaac Lab's sub-terrain generators from `~/workspace/IsaacLab` (needs
`lazy_loader trimesh scipy pyyaml pillow`; the orchestrator's `pxr` is not available and is not
needed). Height fields become a MuJoCo `hfield`, meshes a mesh — **MuJoCo collides meshes by convex
hull**, so rough ground as a mesh is one dome and scored 0% against 60% as an `hfield`.

Not yet trustworthy: contacts are untuned and an `hfield` episode was seen reaching 13.56 m/s, which
is the solver ejecting the robot. And whether the trained task uses stock `ROUGH_TERRAINS_CFG` is
unverified.

## Things that are easy to get wrong

- **This policy walks but cannot stand.** `--vx 0.0` falls in ~1.4 s, measured three times on the
  hardware code path. The asset mismatch behind it is in `README.md`. Never demo with `vx = 0`.
- **`--sim sync` cannot share a DDS domain** with `run_sim_loop.py` or a robot; it brings its own
  simulator. Two 500 Hz publishers on `rt/lowstate` overrun the SDK reader's queue and the loop
  wedges inside `dds_write` in C, where Ctrl-C cannot reach it. There is a startup probe that
  refuses rather than wedging.
- **Under `mjpython`, Ctrl-C does not stop the loop** — the script runs off the main thread, so
  Python delivers `KeyboardInterrupt` to mjpython's event loop. Use `x` or close the viewer window.
- **`N/M control steps would have been clamped` is expected**, and is not a reason to pass
  `--clamp_targets`. See `README.md`.
- **`--dry_run` does not validate `--policy_physics`.** `build_default_pose()` is keyed by joint
  name, so the ramp target is byte-identical under all three layouts, `g1_29dof` included. What
  validates the ordering is the MuJoCo benchmark — a scrambled action order does not walk. The
  *gains* do differ: `g1_29dof` drives the four wrist pitch/yaw motors instead of holding them, so
  their kd goes 1.0 → 10.0.
- **A shape mismatch loading a checkpoint means the wrong `--policy_physics`.** `physx`/`newton` are
  the superseded 37-joint USD (600-element observation); `g1_29dof` is the current robot description
  (43 joints, 690). The gains, default pose and joint order for the last come from
  `g1_deploy/data/g1_dr29_contract.json`, dumped with domain randomization off on purpose — a live
  read of a randomized env returns one *sample* of each gain band, not the nominal.

## Two claims in the source that are not established

Both are flagged in place, both reach the right conclusion for possibly the wrong reason. Do not
repeat them as fact:

- `hardware.py` says the factory motion controller writes `rt/lowcmd`. Measured on a G1 in debug
  mode: `rt/lowcmd` had **zero** traffic. Unitree's docs say the built-in program "periodically sends
  commands with a speed of 0" but never name the topic.
- `hardware.py` says the robot rejects commands that do not echo `mode_machine`. No Unitree page or
  source line asserts this. Echo it anyway — every official example does, and it costs nothing.

Also undocumented anywhere: **the motor watchdog timeout**. No Unitree page states one, no Unitree
repo contains one, and `unitree_mujoco` implements no equivalent — so the MuJoCo rehearsal does not
exercise it. That is why there is deliberately no operator prompt between the ramp and engaging the
policy.

## The depth student's lateral axis is broken

`depth_student_w100` tracks forward commands at 0.8-0.96 of commanded speed and fails systematically
the moment `vy != 0`: every `vy = -0.5` episode runs away at 2.1-2.4x, every `vy = +0.5` episode
stalls. The teacher `ckpt/policy_newasset.pt` tracks all of them correctly in the same environment,
on the same asset, through the same benchmark — so it is the distillation, not the plumbing.

**Command forward only** (`--vy 0.0`) until that is resolved on the training side.

Two things that were checked and are *not* the cause: the foot geometry (reproducing training's
plate override moved survival 80% -> 86.7% and left the pattern untouched) and the heading controller
(the hold suite's heading target is 0 for every episode).

Not checked, and worth ruling out on the training machine: the contract records `obs_1d_dim` but not
the observation *term order*. The 690-wide layout here is inherited from the old policy. Forward
walking working argues it is right, but a subtle misordering would look exactly like this.

## Verifying a change

Run all of these before claiming anything works. They are fast and they have caught real regressions.

```bash
conda activate deploy && cd ~/workspace/g1_deploy

# single-process deterministic
python scripts/run_policy_loop.py --sim sync --policy ckpt/policy.pt --vx 0.5 --duration 10 --domain_id 61 --interface lo0

# two-terminal DDS, the hardware sequence
python -u scripts/run_sim_loop.py --domain_id 62 --interface lo0 --hoist_s 30 &
echo go | python -u scripts/run_policy_loop.py --real --dry_run --policy ckpt/policy.pt --hold_s 3 --domain_id 62 --interface lo0

# quantitative — must stay at 66.7% / 0.380 m/s for this checkpoint at repeats=1
python scripts/benchmark_mujoco.py --policy ckpt/policy.pt --suite hold --repeats 1 --workers 4 --out /tmp/b.npz
```

```bash
# vision closed loop; --blind is the control that proves the camera is being used
python scripts/benchmark_depth_mujoco.py --episodes 15 --foot_plate
```

The benchmark number is the regression signal: any change that moves it has changed behaviour.
For the vision benchmark, read **achieved/commanded speed**, not the survival rate — "survived" only
means "did not fall", and a policy standing still through a walk command scores 100%.
`ruff` is configured (`line-length = 120`) but not installed in `deploy`; keep lines under 120.

Kill stray simulators between runs (`pkill -f run_sim_loop`) — they leak into the next test's domain.

## Outstanding

Bring-up is still two processes: `--dry_run` damps and exits, then the policy run ramps again, which
leaves an `rt/lowcmd` gap while the operator lowers the hoist. Unitree's own `unitree_rl_gym` recipe
is one process that keeps publishing throughout and gates each escalation on a button press. Folding
`hold_pose` into "hold and publish until a keypress" would close the gap and make the hold-and-look
free on every run. Agreed as the right shape; not built.
