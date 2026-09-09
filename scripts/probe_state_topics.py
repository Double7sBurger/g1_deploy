# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Find out whether the G1 publishes a base pose we could feed to elevation mapping.

Route A (height map on hardware) needs a base pose at locomotion rate. ``rt/lowstate`` carries only
the IMU and the joint encoders, so the question is whether the factory stack publishes an estimate
on some other topic, and -- more importantly -- whether it keeps publishing once we take
``rt/lowcmd`` and the factory motion mode is released.

Run it in three passes and compare:

    python scripts/probe_state_topics.py --interface en6                 # robot idle, factory mode on
    python scripts/probe_state_topics.py --interface en6 --release       # after releasing factory mode
    python scripts/probe_state_topics.py --interface en6 --walk-first    # while a policy is driving

For every candidate topic it reports whether messages arrive, at what rate, and whether the pose
actually changes when the robot moves -- a topic that publishes a frozen zero is worse than one that
publishes nothing, because it looks alive.
"""

from __future__ import annotations

import argparse
import time

# (topic, module, message class name)
CANDIDATES = [
    ("rt/lf/sportmodestate", "unitree_go", "SportModeState_"),
    ("rt/sportmodestate", "unitree_go", "SportModeState_"),
    ("rt/odommodestate", "unitree_go", "SportModeState_"),
    ("rt/lf/lowstate", "unitree_hg", "LowState_"),
    ("rt/lowstate", "unitree_hg", "LowState_"),
]


def _resolve(module: str, name: str):
    import importlib

    mod = importlib.import_module(f"unitree_sdk2py.idl.{module}.msg.dds_")
    return getattr(mod, name)


def probe(topic: str, module: str, cls_name: str, seconds: float) -> dict:
    """Subscribe for ``seconds`` and report arrival rate and whether a pose field moves."""
    from unitree_sdk2py.core.channel import ChannelSubscriber

    try:
        msg_cls = _resolve(module, cls_name)
    except (ImportError, AttributeError) as exc:
        return {"topic": topic, "status": f"message type unavailable: {exc}"}

    received = []

    def _on_msg(msg):
        received.append((time.time(), msg))

    sub = ChannelSubscriber(topic, msg_cls)
    sub.Init(_on_msg, 10)
    time.sleep(seconds)

    if not received:
        return {"topic": topic, "status": "silent", "count": 0}

    rate = (len(received) - 1) / max(received[-1][0] - received[0][0], 1e-6)
    out = {"topic": topic, "status": "publishing", "count": len(received), "rate_hz": round(rate, 1)}

    first, last = received[0][1], received[-1][1]
    for field in ("position", "velocity", "body_height"):
        if not hasattr(first, field):
            continue
        a, b = getattr(first, field), getattr(last, field)
        a = list(a) if hasattr(a, "__iter__") else [a]
        b = list(b) if hasattr(b, "__iter__") else [b]
        out[field] = {
            "first": [round(v, 4) for v in a],
            "last": [round(v, 4) for v in b],
            "moved": any(abs(x - y) > 1e-6 for x, y in zip(a, b)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True, help="network interface facing the robot, e.g. en6")
    parser.add_argument("--domain_id", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=3.0, help="listen window per topic [s]")
    parser.add_argument("--release", action="store_true", help="release the factory motion mode first")
    args = parser.parse_args()

    import sys, pathlib  # noqa: E401

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "g1_deploy"))
    from dds_bootstrap import ensure_cyclonedds  # noqa: PLC0415

    ensure_cyclonedds()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: PLC0415

    ChannelFactoryInitialize(args.domain_id, args.interface)

    if args.release:
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient  # noqa: PLC0415

        client = LocoClient()
        client.SetTimeout(10.0)
        client.Init()
        print("[..] releasing factory motion mode")
        client.Damp()
        time.sleep(1.0)
        print("[ok] released")

    print(f"\nlistening {args.seconds}s per topic on {args.interface}, domain {args.domain_id}\n")
    for topic, module, cls_name in CANDIDATES:
        result = probe(topic, module, cls_name, args.seconds)
        print(f"--- {topic}")
        for key, value in result.items():
            if key != "topic":
                print(f"      {key}: {value}")

    print(
        "\nWhat to look for: a topic that is 'publishing' at >= 50 Hz AND whose 'position' has"
        "\n'moved': True while you push the robot around. Publishing a frozen pose is a failure,"
        "\nnot a success -- elevation mapping would silently smear the map."
    )


if __name__ == "__main__":
    main()
