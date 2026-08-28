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
  name, so the ramp target is byte-identical under `physx` and `newton`. What validates the ordering
  is the MuJoCo benchmark — a scrambled action order does not walk.

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

The benchmark number is the regression signal: any change that moves it has changed behaviour.
`ruff` is configured (`line-length = 120`) but not installed in `deploy`; keep lines under 120.

Kill stray simulators between runs (`pkill -f run_sim_loop`) — they leak into the next test's domain.

## Outstanding

Bring-up is still two processes: `--dry_run` damps and exits, then the policy run ramps again, which
leaves an `rt/lowcmd` gap while the operator lowers the hoist. Unitree's own `unitree_rl_gym` recipe
is one process that keeps publishing throughout and gates each escalation on a button press. Folding
`hold_pose` into "hold and publish until a keypress" would close the gap and make the hold-and-look
free on every run. Agreed as the right shape; not built.
