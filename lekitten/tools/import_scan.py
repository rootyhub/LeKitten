"""Turn a Polycam/LiDAR scan into MuJoCo-ready meshes (all output stays in sim/private_scan/).

    python -m lekitten.tools.import_scan "sim/private_scan/LeKitten Polycam.obj"

What it does, in order:
  1. Loads the OBJ/GLB (one mesh per material).
  2. Rotates it so z is up (OBJ/GLB files are y-up).
  3. Finds the floor (the lowest big flat area) and moves it to z = 0.
  4. Puts the origin at the play-zone corner: --origin X Y is that corner measured
     from the room's floor corner (min x, min y). Check it on outputs/scan_topdown.png.
  5. Shrinks meshes above --max-faces, converts textures to PNG (MuJoCo can't read JPG).
  6. Writes sim/private_scan/processed/ (meshes, textures, scan_info.yaml)
     and outputs/scan_topdown.png (a height map for spotting furniture).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
import yaml
from PIL import Image

from lekitten.config import REPO

OUT = REPO / "sim/private_scan/processed"

Y_UP_TO_Z_UP = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])


def find_floor_z(mesh: trimesh.Trimesh) -> float:
    """Height of the floor: area-weighted mode of upward-facing faces in the bottom 40%."""
    up = mesh.face_normals[:, 2] > 0.95
    z = mesh.triangles_center[up, 2]
    a = mesh.area_faces[up]
    zmin, zmax = mesh.bounds[:, 2]
    keep = z < zmin + 0.4 * (zmax - zmin)
    hist, edges = np.histogram(z[keep], bins=200, weights=a[keep])
    i = hist.argmax()
    return float((edges[i] + edges[i + 1]) / 2)


def write_obj(path: Path, verts: np.ndarray, faces: np.ndarray, uv: np.ndarray | None) -> None:
    """Minimal OBJ writer that keeps per-vertex UVs (trimesh's exporter drops them without a material)."""
    with open(path, "w") as f:
        f.write("# LeKitten processed scan (z up, floor at z=0)\n")
        np.savetxt(f, verts, fmt="v %.6f %.6f %.6f")
        if uv is not None:
            np.savetxt(f, uv[:, :2], fmt="vt %.6f %.6f")
            np.savetxt(f, np.repeat(faces + 1, 2, axis=1), fmt="f %d/%d %d/%d %d/%d")
        else:
            np.savetxt(f, faces + 1, fmt="f %d %d %d")


def height_map(meshes: list[trimesh.Trimesh], res: float = 0.02):
    """Top-down max height per 2 cm cell (ignores the ceiling)."""
    allv = np.vstack([m.vertices for m in meshes])
    lo, hi = allv.min(0), allv.max(0)
    w, h = int((hi[0] - lo[0]) / res) + 1, int((hi[1] - lo[1]) / res) + 1
    hm = np.full((h, w), np.nan)
    ceiling = hi[2] - 0.25
    for m in meshes:
        pts = m.sample(400_000) if len(m.faces) else m.vertices
        pts = pts[pts[:, 2] < ceiling]
        ix = ((pts[:, 0] - lo[0]) / res).astype(int)
        iy = ((pts[:, 1] - lo[1]) / res).astype(int)
        order = np.argsort(pts[:, 2])
        hm[iy[order], ix[order]] = pts[order, 2]  # last write = highest
    return hm, lo, res


def save_topdown(hm, lo, res, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h, w = hm.shape
    extent = [lo[0], lo[0] + w * res, lo[1], lo[1] + h * res]
    fig, ax = plt.subplots(figsize=(8, 8 * h / w + 0.6))
    im = ax.imshow(np.clip(hm, 0, 1.6), origin="lower", extent=extent, cmap="viridis")
    fig.colorbar(im, ax=ax, label="height above floor (m)", shrink=0.8)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Scan, top-down height map")
    ax.grid(True, color="white", alpha=0.25, lw=0.5)
    ax.set_xticks(np.arange(np.floor(extent[0]), extent[1] + 0.01, 0.5))
    ax.set_yticks(np.arange(np.floor(extent[2]), extent[3] + 0.01, 0.5))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scan", help="OBJ or GLB in sim/private_scan/")
    p.add_argument("--max-faces", type=int, default=60_000, help="Per mesh; larger ones get simplified")
    p.add_argument("--origin", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"),
                   help="Play-zone corner, measured from the room's floor corner. Becomes (0, 0).")
    p.add_argument("--y-up", action=argparse.BooleanOptionalAction, default=True,
                   help="Input is y-up (Polycam OBJ/GLB). --no-y-up if it's already z-up.")
    args = p.parse_args()

    scene = trimesh.load(args.scan, force="scene")
    meshes = []
    for name, g in scene.geometry.items():
        g = g.copy()
        if args.y_up:
            g.apply_transform(Y_UP_TO_Z_UP)
        meshes.append((name, g))

    combined = trimesh.util.concatenate([g for _, g in meshes])
    floor_z = find_floor_z(combined)
    corner = combined.bounds[0]
    shift = np.array([-corner[0] - args.origin[0], -corner[1] - args.origin[1], -floor_z])
    for _, g in meshes:
        g.apply_translation(shift)

    OUT.mkdir(parents=True, exist_ok=True)
    info = {"source": str(Path(args.scan).name), "floor_z_in_source": floor_z,
            "shift_applied": shift.round(4).tolist(), "meshes": []}
    for i, (name, g) in enumerate(meshes):
        if len(g.faces) > args.max_faces:
            try:
                g = g.simplify_quadric_decimation(face_count=args.max_faces)
            except Exception as e:  # optional dependency
                print(f"  (couldn't simplify {name}: {e}; keeping {len(g.faces)} faces)")
        stem = f"scan_{i}"
        tex_file = None
        mat = getattr(g.visual, "material", None)
        img = getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)
        if img is not None:
            tex_file = f"{stem}.png"
            Image.fromarray(np.asarray(img.convert("RGB"))).save(OUT / tex_file)
        # Geometry + UVs only; MuJoCo gets the texture through the scene XML.
        write_obj(OUT / f"{stem}.obj", g.vertices, g.faces, getattr(g.visual, "uv", None))
        info["meshes"].append({"name": name, "obj": f"{stem}.obj", "texture": tex_file,
                               "faces": int(len(g.faces))})
        print(f"  {name}: {len(g.faces)} faces -> {stem}.obj" + (f" + {tex_file}" if tex_file else ""))

    final = trimesh.util.concatenate([g for _, g in meshes])
    lo, hi = final.bounds
    info["room_size_m"] = (hi - lo).round(3).tolist()
    info["ceiling_height_m"] = round(float(hi[2]), 3)
    (OUT / "scan_info.yaml").write_text(yaml.safe_dump(info, sort_keys=False))

    hm, hlo, res = height_map([g for _, g in meshes])
    save_topdown(hm, hlo, res, REPO / "outputs/scan_topdown.png")
    print(f"Room: {hi[0]-lo[0]:.2f} m x {hi[1]-lo[1]:.2f} m, ceiling {hi[2]:.2f} m")
    print(f"Wrote {OUT.relative_to(REPO)}/ and outputs/scan_topdown.png")


if __name__ == "__main__":
    main()
