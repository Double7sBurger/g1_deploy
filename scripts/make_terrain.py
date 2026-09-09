# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the training task's terrain for MuJoCo, from Isaac Lab's own generators.

A vision policy is trained on rough ground and tested here on a flat plane, which is the one thing a
depth camera exists to notice. This closes that gap without reimplementing anything: Isaac Lab's
sub-terrain functions are plain numpy and trimesh, so they import and run without Isaac Sim, and the
geometry they return is the geometry training walked on.

Only the *orchestrator* is unavailable -- ``isaaclab.terrains.terrain_generator`` imports ``pxr``,
which ships with Isaac Sim. That module only tiles sub-terrains into a grid and assigns difficulty by
row, so a single sub-terrain at a chosen difficulty is generated here directly and written as an STL
that MuJoCo loads as a static mesh.

.. attention::
    ``ROUGH_TERRAINS_CFG`` is Isaac Lab's stock rough terrain, and this reads it from the local
    checkout. Whether the trained task actually uses it is **not verified** -- the task
    ``Isaac-Velocity-Rough-G1-29Dof-AirTime100-DepthDistill-W100`` lives on the training machine and
    may configure its own. Check ``env_cfg.scene.terrain.terrain_generator`` there before treating a
    number measured on this terrain as comparable to training.

Usage::

    python scripts/make_terrain.py --list
    python scripts/make_terrain.py --terrain random_rough --difficulty 0.5 --out /tmp/rough.xml
    python scripts/benchmark_depth_mujoco.py --xml /tmp/rough.xml --episodes 15 --foot_plate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ISAACLAB = Path.home() / "workspace/IsaacLab/source/isaaclab"
"""Local Isaac Lab checkout. Only its terrain sub-package is used, and only its numpy parts."""


def _load_cfg():
    """Import ``ROUGH_TERRAINS_CFG`` and the sub-terrain functions without pulling in Isaac Sim."""
    if str(ISAACLAB) not in sys.path:
        sys.path.insert(0, str(ISAACLAB))
    try:
        from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
    except ImportError as exc:
        raise SystemExit(
            f"cannot import Isaac Lab's terrain config from {ISAACLAB}: {exc}\n"
            "Needs the checkout plus: pip install lazy_loader trimesh scipy pyyaml"
        ) from exc
    return ROUGH_TERRAINS_CFG


def build_tile(name: str, difficulty: float, cfg):
    """Generate one sub-terrain tile and return it as a triangle mesh.

    Height-field terrains are converted to a mesh here rather than handed to MuJoCo as an ``hfield``,
    so that stairs and boxes -- which are meshes in Isaac Lab and cannot be represented by a
    single-valued height map -- go through the same path. One code path means one thing to get wrong.

    Args:
        name: Key in ``cfg.sub_terrains``.
        difficulty: 0 to 1, as the generator's row index would give.
        cfg: ``ROUGH_TERRAINS_CFG``.

    Returns:
        A ``trimesh.Trimesh`` in tile-local coordinates, spanning ``cfg.size``.

    Raises:
        SystemExit: On an unknown terrain name.
    """
    import trimesh

    if name == "flat":
        # Isaac Lab's own plane generator rather than a hand-rolled quad, so the pad is the same
        # thing the training config would produce if it listed one.
        from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg

        sub = MeshPlaneTerrainCfg(size=cfg.size)
    elif name not in cfg.sub_terrains:
        raise SystemExit(f"unknown terrain {name!r}; choose from {list(cfg.sub_terrains)} or 'flat'")
    else:
        sub = cfg.sub_terrains[name].copy()
        sub.size = cfg.size

    is_hf = hasattr(sub, "horizontal_scale")
    if is_hf:  # height-field config
        sub.horizontal_scale = cfg.horizontal_scale
        sub.vertical_scale = cfg.vertical_scale
        sub.slope_threshold = cfg.slope_threshold
        # Each hf function is already wrapped by @height_field_to_mesh at import time, so calling
        # it returns meshes, not a height array. Importing that decorator from
        # isaaclab.terrains.utils would drag in pxr; it lives in height_field/utils.py and is
        # already applied.
        meshes, _origin = sub.function(difficulty, sub)
        meshes = meshes if isinstance(meshes, list) else [meshes]
    else:
        meshes, _origin = sub.function(difficulty, sub)

    return trimesh.util.concatenate(meshes)


def build_grid(cfg, rows: int, cols: int, seed: int = 0, types_along_x: bool = True,
               flat_cols: int = 0):
    """Tile sub-terrains into a grid, following ``TerrainGenerator``'s curriculum layout.

    Transcribed from ``terrain_generator.py``: the sub-terrain *type* is chosen per column from the
    cumulative proportions, the *difficulty* rises along rows as ``(row + u) / num_rows``, each tile
    is translated to ``((row + 0.5) * size_x, (col + 0.5) * size_y)``, and the whole thing is finally
    centred on the origin.

    That layout is right for training, where every environment sits on its own tile and the
    curriculum walks it along rows -- but wrong for a *single* robot driven forward, which would then
    spend the whole episode on one terrain type. ``types_along_x`` transposes the grid so forward
    motion crosses types instead, which is the only reason to deviate and is stated here rather than
    hidden: with it set, +X changes type and +Y changes difficulty.

    Args:
        cfg: ``ROUGH_TERRAINS_CFG``.
        rows: Rows to generate. The full config is 10; fewer keeps the model small.
        cols: Columns to generate. The full config is 20.
        seed: Difficulty jitter seed.
        flat_cols: Columns of Isaac Lab's own ``MeshPlaneTerrainCfg`` to splice into the middle. The
            robot spawns at the grid's centre, so this is what it starts on -- it gets level ground
            under its feet for the ramp and the first strides, and meets the generated terrain only
            once it has walked off the pad. Without it the spawn lands on whatever tile happens to
            be central, which on ``pyramid_stairs`` is a step.

    Returns:
        ``(mesh, size_x, size_y, layout)`` -- the combined mesh centred on the origin, its extent,
        and the per-column terrain names.
    """
    import trimesh

    rng = np.random.default_rng(seed)
    names = list(cfg.sub_terrains)
    proportions = np.array([cfg.sub_terrains[n].proportion for n in names], dtype=float)
    proportions /= proportions.sum()
    cumulative = np.cumsum(proportions)

    # Column types first, then the flat pad spliced into the middle, so the generated columns keep
    # the proportions TerrainGenerator would have given them.
    column_types = [names[int(np.min(np.where(c / cols + 0.001 < cumulative)[0]))] for c in range(cols)]
    if flat_cols > 0:
        middle = len(column_types) // 2
        column_types[middle:middle] = ["flat"] * flat_cols

    tiles, layout = [], []
    for col, kind in enumerate(column_types):
        layout.append(kind)
        for row in range(rows):
            difficulty = (row + rng.uniform()) / rows
            tile = build_tile(kind, difficulty, cfg)
            transform = np.eye(4)
            if types_along_x:
                transform[0:2, -1] = (col + 0.5) * cfg.size[0], (row + 0.5) * cfg.size[1]
            else:
                transform[0:2, -1] = (row + 0.5) * cfg.size[0], (col + 0.5) * cfg.size[1]
            tile = tile.copy()
            tile.apply_transform(transform)
            tiles.append(tile)

    combined = trimesh.util.concatenate(tiles)
    total_cols = len(column_types)
    nx, ny = (total_cols, rows) if types_along_x else (rows, total_cols)
    centre = np.eye(4)
    centre[:2, -1] = -cfg.size[0] * nx * 0.5, -cfg.size[1] * ny * 0.5
    combined.apply_transform(centre)
    return combined, cfg.size[0] * nx, cfg.size[1] * ny, layout


def rasterise(verts: np.ndarray, size_x: float, size_y: float, res: tuple[int, int]) -> np.ndarray:
    """Sample a terrain mesh's vertices back onto a regular height grid.

    Isaac Lab's height-field terrains are generated as a grid and only then triangulated, so this
    recovers what they started as. Nearest-vertex sampling is exact for those, because every grid
    point is a vertex.

    Args:
        verts: Mesh vertices [m].
        size_x: Terrain extent [m].
        size_y: Terrain extent [m].
        res: ``(nx, ny)`` output grid size.

    Returns:
        ``(res, res)`` heights [m], row 0 at ``y = 0``.
    """
    nx, ny = res
    ix = np.clip(((verts[:, 0] - verts[:, 0].min()) / max(size_x, 1e-9) * (nx - 1)).round().astype(int), 0, nx - 1)
    iy = np.clip(((verts[:, 1] - verts[:, 1].min()) / max(size_y, 1e-9) * (ny - 1)).round().astype(int), 0, ny - 1)
    # Maximum, not nearest: a cell straddling a stair edge must read the tread the foot lands on,
    # and a height field cannot represent the riser anyway. Taking the minimum would sink the robot
    # into the step.
    grid = np.full((ny, nx), -np.inf)
    np.maximum.at(grid, (iy, ix), verts[:, 2])
    # Cells no vertex fell in: fill from the nearest neighbour that has one.
    if not np.isfinite(grid).all():
        from scipy.ndimage import distance_transform_edt

        missing = ~np.isfinite(grid)
        _, (sy, sx) = distance_transform_edt(missing, return_indices=True)
        grid[missing] = grid[sy[missing], sx[missing]]
    return grid


def write_hfield_png(path: Path, grid: np.ndarray) -> None:
    """Write a height grid as the 16-bit PNG MuJoCo's ``hfield`` loader expects.

    MuJoCo normalises the image to ``[0, 1]`` and scales it by the ``hfield`` element's elevation,
    so only the shape matters here, not the absolute values.

    Args:
        path: PNG to write.
        grid: Heights [m].
    """
    from PIL import Image

    lo, hi = float(grid.min()), float(grid.max())
    span = max(hi - lo, 1e-6)
    scaled = np.clip((grid - lo) / span, 0.0, 1.0)
    Image.fromarray((scaled * 65535).astype(np.uint16)).save(path)


def write_scene(out_xml: Path, stl: Path, size_x: float, size_y: float, robot_xml: Path,
                hfield: tuple[int, int, float] | None = None, base_z: float = 0.0) -> None:
    """Write a self-contained MuJoCo scene putting the robot on the generated terrain.

    Includes are resolved and every asset path made absolute, rather than emitting an ``<include>``.
    MuJoCo resolves ``meshdir`` against the top-level file, and the robot MJCF declares its own
    ``meshdir="meshes"`` -- so an included robot sends the compiler looking for meshes beside the
    *scene*, wherever ``--out`` happens to point. Writing one flat file removes the question.

    The terrain mesh replaces the ground plane rather than joining it: leaving the plane in would
    floor every dip in the terrain at z=0, and the robot would walk on whichever surface came first.

    Args:
        out_xml: Scene file to write.
        stl: Terrain mesh, already written.
        size_x: Terrain extent [m], for centring.
        size_y: Terrain extent [m].
        robot_xml: Robot MJCF to inline.
    """
    import xml.etree.ElementTree as ET

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from g1_deploy.sim.mjcf import resolve_includes

    root = resolve_includes(robot_xml)
    root.set("model", "g1_rough")

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    meshdir = robot_xml.parent / compiler.get("meshdir", "")
    compiler.set("meshdir", str(meshdir.resolve()))

    for plane in [g for g in root.iter("geom") if g.get("type") == "plane"]:
        for parent in root.iter():
            if plane in list(parent):
                parent.remove(plane)

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    if hfield is None:
        ET.SubElement(asset, "mesh", {"name": "terrain", "file": str(stl.resolve())})
    else:
        nrow, ncol, elevation = hfield
        ET.SubElement(asset, "hfield", {
            "name": "terrain",
            "file": str(stl.resolve()),
            "nrow": str(nrow),
            "ncol": str(ncol),
            # radius_x radius_y elevation base. MuJoCo collides a height field cell by cell rather
            # than by convex hull, which is the whole reason to prefer it here.
            "size": f"{size_x / 2:.4f} {size_y / 2:.4f} {max(elevation, 1e-3):.4f} 0.5",
        })

    world = root.find("worldbody")
    if world is None:
        world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", {
        "name": "terrain",
        "type": "mesh" if hfield is None else "hfield",
        ("mesh" if hfield is None else "hfield"): "terrain",
        # MuJoCo scales the normalised height data by `elevation` measured up from the geom's own
        # z, so the geom sits at the terrain's minimum rather than at zero.
        "pos": f"0 0 {base_z:.4f}" if hfield is not None else f"{-size_x / 2:.3f} {-size_y / 2:.3f} 0",
        "rgba": "0.55 0.55 0.55 1",
        "contype": "1",
        "conaffinity": "1",
        "friction": "1 0.005 0.0001",
    })
    ET.SubElement(world, "light", {"pos": "0 0 4", "dir": "0 0 -1", "directional": "true"})
    out_xml.write_text(ET.tostring(root, encoding="unicode"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=int, default=4,
                    help="Grid rows; difficulty rises along +X. Full config is 10.")
    ap.add_argument("--cols", type=int, default=6,
                    help="Grid columns; terrain type changes along +Y. Full config is 20.")
    ap.add_argument("--terrain", default=None,
                    help="Generate a single tile of this sub-terrain instead of the grid.")
    ap.add_argument("--difficulty", type=float, default=0.5, help="Only with --terrain. 0 easiest, 1 hardest.")
    ap.add_argument("--seed", type=int, default=0, help="Difficulty jitter, as the generator uses it.")
    ap.add_argument("--flat_cols", type=int, default=2,
                    help="Columns of flat ground spliced into the grid centre, where the robot"
                         " spawns. 0 disables it.")
    ap.add_argument("--isaac_layout", action="store_true",
                    help="Keep Isaac Lab's own axes: type along +Y, difficulty along +X. The default"
                         " transposes them so a robot walking forward crosses terrain types instead"
                         " of staying on one.")
    ap.add_argument("--cell", type=float, default=0.1,
                    help="Height-field cell size [m]. 0.1 is the config's own horizontal_scale.")
    ap.add_argument("--out", default="/tmp/g1_terrain.xml")
    ap.add_argument("--robot_xml",
                    default=str(Path.home() / "workspace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"))
    ap.add_argument("--list", action="store_true", help="Print the sub-terrains and exit.")
    args = ap.parse_args()

    cfg = _load_cfg()
    if args.list:
        print(f"ROUGH_TERRAINS_CFG: {cfg.size[0]:.0f}x{cfg.size[1]:.0f} m tiles,"
              f" {cfg.num_rows} rows x {cfg.num_cols} cols")
        for key, sub in cfg.sub_terrains.items():
            print(f"  {key:<22s} proportion {sub.proportion:.1f}  {type(sub).__name__}")
        return 0

    if args.terrain is not None:
        if args.terrain not in cfg.sub_terrains:
            raise SystemExit(f"unknown terrain {args.terrain!r}; choose from {list(cfg.sub_terrains)}")
        import trimesh

        tile = build_tile(args.terrain, args.difficulty, cfg)
        centre = np.eye(4)
        centre[:2, -1] = -cfg.size[0] / 2, -cfg.size[1] / 2
        tile = tile.copy()
        tile.apply_transform(centre)
        mesh, sx, sy = tile, float(cfg.size[0]), float(cfg.size[1])
        layout = [args.terrain]
        print(f"[..] single tile: {args.terrain} at difficulty {args.difficulty}")
    else:
        mesh, sx, sy, layout = build_grid(cfg, args.rows, args.cols, args.seed,
                                          types_along_x=not args.isaac_layout,
                                          flat_cols=args.flat_cols)
        axis_t, axis_d = ("+Y", "+X") if args.isaac_layout else ("+X", "+Y")
        print(f"[..] grid {args.rows} difficulty levels x {args.cols} terrain types,"
              f" {cfg.size[0]:.0f} m tiles")
        print(f"     type along {axis_t}: {' '.join(layout)}")
        if args.flat_cols:
            print(f"     spawn is on the {args.flat_cols * cfg.size[0]:.0f} m flat pad at the centre")
        print(f"     difficulty along {axis_d}: {1 / args.rows:.2f} to 1.00")

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    res = (max(2, int(round(sx / args.cell))), max(2, int(round(sy / args.cell))))
    grid = rasterise(verts, sx, sy, res)

    out = Path(args.out)
    png = out.with_suffix(".png")
    write_hfield_png(png, grid)
    lo, hi = float(grid.min()), float(grid.max())
    write_scene(out, png, sx, sy, Path(args.robot_xml),
                hfield=(grid.shape[0], grid.shape[1], hi - lo), base_z=lo)

    print(f"[ok] {sx:.0f} x {sy:.0f} m, height field {res[0]} x {res[1]} at {args.cell * 100:.0f} cm cells")
    print(f"     height {lo:+.3f} to {hi:+.3f} m")
    print(f"     {png}")
    print(f"     {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
