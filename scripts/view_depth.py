# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Watch the depth stream the policy is reading, as text. Works over SSH, needs no display.

Points at the same UDP port the control loop listens on, so it shows exactly what the policy sees --
whether the frames come from ``run_sim_loop.py --depth`` or from ``depth_publisher.py`` on the
robot's PC2. The renderer's viewer shows the robot; this shows the *observation*, which is the thing
that can be silently wrong.

Run it beside the control loop rather than instead of it: two UDP sockets on the same port both
receive, because the publisher sends to one address and the OS delivers to every bound socket only
when they ask to share the port. They do not by default, so start this **before** the control loop,
or give it its own port with a second publisher. The ``--port`` default matches the control loop's.

What to look for:

* a gradient from far at the top to near at the bottom -- that is the ground under a camera pitched
  down. A flat image means the camera is aimed at a wall, or not pitched at all.
* the near/far numbers against the geometry: a camera 1.26 m up and pitched 47.6 degrees sees the
  image centre at about 1.7 m on flat ground.
* the invalid fraction. A real D435i indoors reads a few percent; tens of percent means the scene is
  out of range, too reflective, or the sensor is blinded.

Usage::

    python scripts/view_depth.py --port 5601
    python scripts/view_depth.py --port 5601 --once     # one frame, then exit
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from g1_deploy.depth_link import DEFAULT_PORT, DepthReceiver

RAMP = " .:-=+*#%@"
"""Near is bright. A blank cell is a pixel with no return."""


def render(frame: np.ndarray, max_range: float, rows: int = 19) -> str:
    """Draw one depth frame as text.

    Args:
        frame: Depth in metres, invalid as ``<= 0``.
        max_range: Range the policy clips at; sets the brightness scale so the picture matches what
            the network is normalised against rather than the frame's own extremes.
        rows: Output rows; the frame is subsampled to fit.

    Returns:
        A newline-joined block.
    """
    step = max(1, frame.shape[0] // rows)
    out = []
    for row in frame[::step]:
        out.append("".join(
            " " if v <= 0 else RAMP[int(np.clip(1.0 - v / max_range, 0.0, 0.999) * len(RAMP))]
            for v in row
        ))
    return "\n".join(out)


def colourise(frame: np.ndarray, max_range: float, gray: bool = False) -> np.ndarray:
    """Map depth to RGB: near is warm, far is cool, no return is black.

    The same clip the policy applies is used for the colour scale, so what the window shows is
    scaled exactly like the numbers the network sees rather than stretched to the frame's own
    extremes -- a picture that rescales itself every frame hides the thing worth noticing, which is
    ground getting closer.

    Args:
        frame: Depth in metres, invalid as ``<= 0``.
        max_range: Range the policy clips at.
        gray: Draw the single channel the policy actually reads, rather than false colour. The
            network sees one normalised number per pixel; the colours are a display choice, made
            because ten centimetres of height is a couple of grey levels and half a hue.

    Returns:
        ``(h, w, 3)`` uint8.
    """
    valid = frame > 0
    t = np.clip(np.where(valid, frame, max_range) / max_range, 0.0, 1.0)
    if gray:
        # Near bright, matching the text view's ramp, so the two read the same way round.
        level = ((1.0 - t) * 255).astype(np.uint8)
        out = np.stack([level] * 3, axis=-1)
        out[~valid] = 0
        return out
    # A coarse turbo: red -> yellow -> green -> cyan -> blue as depth grows.
    r = np.clip(1.5 - 3.0 * t, 0, 1)
    g = np.clip(1.5 - np.abs(3.0 * t - 1.5), 0, 1)
    b = np.clip(3.0 * t - 1.5, 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def run_window(rx, args) -> int:
    """Show the stream in a tkinter window until it is closed.

    Args:
        rx: A live :class:`~g1_deploy.depth_link.DepthReceiver`.
        args: Parsed CLI arguments.

    Returns:
        Process exit status.
    """
    import tkinter as tk

    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title(f"depth :{args.port}")
    label = tk.Label(root)
    label.pack()
    status = tk.Label(root, font=("Menlo", 11), anchor="w", justify="left")
    status.pack(fill="x")
    keep = {}

    def tick():
        frame, age = rx.latest()
        if frame is not None:
            img = Image.fromarray(colourise(frame, args.max_range, args.gray))
            img = img.resize((frame.shape[1] * args.zoom, frame.shape[0] * args.zoom), Image.NEAREST)
            keep["img"] = ImageTk.PhotoImage(img)
            label.configure(image=keep["img"])
            valid = frame[frame > 0]
            mid = frame[frame.shape[0] // 2]
            mid = mid[mid > 0]
            status.configure(text=(
                f"recv {rx.received}  dropped {rx.dropped}  age {age * 1000:5.1f} ms\n"
                f"valid {100.0 * valid.size / frame.size:5.1f}%  "
                + (f"near {valid.min():.2f}  far {valid.max():.2f}  centre row {mid.mean():.2f} m"
                   if valid.size and mid.size else "all invalid")
            ))
        root.after(int(1000 / max(1.0, args.hz)), tick)

    tick()
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        rx.close()
    return 0


def run_save(rx, args) -> int:
    """Collect frames to an ``.npz`` and report where the camera returns nothing.

    Saves a burst rather than one frame, because a single frame cannot separate a row that is always
    empty -- the robot's own body in the lower field of view, or a surface past the sensor's range --
    from one that dropped out on that frame. A row invalid in every frame of the burst is structural
    and the renderer should reproduce it; one invalid in a few is noise.

    Args:
        rx: A live :class:`~g1_deploy.depth_link.DepthReceiver`.
        args: Parsed CLI arguments.

    Returns:
        Process exit status.
    """
    frames, seen = [], set()
    deadline = time.monotonic() + max(10.0, args.save_n / 5.0)
    while len(frames) < args.save_n and time.monotonic() < deadline:
        frame, _age = rx.latest()
        key = frame.tobytes()
        if key not in seen:
            seen.add(key)
            frames.append(frame.copy())
        time.sleep(0.01)
    rx.close()
    if not frames:
        print("[FAIL] no frames collected")
        return 1

    stack = np.stack(frames)
    valid = stack > 0
    np.savez_compressed(args.save, frames=stack, max_range=args.max_range)
    print(f"[ok] {args.save}  {len(frames)} distinct frames of {stack.shape[1]}x{stack.shape[2]}")
    print(f"     valid {100.0 * valid.mean():.1f}% overall")
    per_row = 100.0 * valid.mean(axis=(0, 2))
    always_empty = np.flatnonzero(per_row == 0.0)
    print("     per-row valid %, top to bottom:")
    print("       " + " ".join(f"{v:3.0f}" for v in per_row))
    if always_empty.size:
        print(f"     rows {always_empty.min()}-{always_empty.max()} are empty in every frame"
              f" ({always_empty.size} of {stack.shape[1]}) -- structural, not noise")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--width", type=int, default=64, help="Frame width the publisher sends.")
    ap.add_argument("--height", type=int, default=38, help="Frame height the publisher sends.")
    ap.add_argument("--max_range", type=float, default=3.0, help="Clip range; contract depth_max_range_m.")
    ap.add_argument("--hz", type=float, default=4.0, help="Redraw rate.")
    ap.add_argument("--once", action="store_true", help="Print one frame and exit.")
    ap.add_argument("--window", action="store_true",
                    help="Open a colour window instead of drawing text. Needs tkinter and Pillow,"
                         " both of which ship with the environment.")
    ap.add_argument("--zoom", type=int, default=10, help="Window pixels per depth pixel.")
    ap.add_argument("--gray", action="store_true",
                    help="Draw the normalised single channel the policy reads instead of false"
                         " colour. Harder to judge small height differences by eye, which is why"
                         " colour is the default.")
    ap.add_argument("--save", default=None,
                    help="Write the frames seen to this .npz and exit after --save_n of them. The"
                         " point is comparing the real camera against the renderer row by row: the"
                         " arrays are the same shape either way, so a difference in which rows come"
                         " back empty is invisible in a picture and obvious in the numbers.")
    ap.add_argument("--save_n", type=int, default=30, help="Frames to collect for --save.")
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()

    rx = DepthReceiver((args.height, args.width), port=args.port)
    print(f"[..] listening on UDP {args.port} for {args.width}x{args.height} frames ...")
    try:
        rx.wait_for_frame(timeout=args.timeout)
    except TimeoutError as exc:
        print(f"[FAIL] {exc}")
        print("       Is the publisher running, and pointed at this machine? If the control loop is")
        print("       already bound to this port, start this first or use a separate port.")
        rx.close()
        return 1

    if args.save:
        return run_save(rx, args)
    if args.window:
        return run_window(rx, args)

    try:
        while True:
            frame, age = rx.latest()
            valid = frame[frame > 0]
            head = (
                f"seq-received {rx.received}  dropped {rx.dropped}  age {age * 1000:5.1f} ms"
                f"  |  valid {100.0 * valid.size / frame.size:5.1f}%"
            )
            if valid.size:
                head += (f"  near {valid.min():.2f}  far {valid.max():.2f}  mean {valid.mean():.2f} m"
                         f"  |  centre row {np.mean(frame[frame.shape[0] // 2][frame[frame.shape[0] // 2] > 0]):.2f} m")
            print("\033[2J\033[H" + head + "\n" + render(frame, args.max_range), flush=True)
            if args.once:
                break
            time.sleep(1.0 / max(0.5, args.hz))
    except KeyboardInterrupt:
        print("\n[..] interrupted")
    finally:
        rx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
