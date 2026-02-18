"""
Generate synthetic sample reference data.

Creates 5 reference folders, each containing:
  - final.dxf       (pool CAD drawing)
  - line_drawing.png (rendered line drawing of the pool)
  - metadata.json    (pool type, dimensions, features)

Requires: ezdxf, matplotlib, Pillow
"""

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure sketch_to_cad is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sketch_to_cad.cad_engine.pool_generator import SAMPLE_POOLS, save_pool_dxf, PoolSpec

SAMPLE_DIR = os.path.dirname(__file__)


def render_pool_png(spec: PoolSpec, path: str, dpi: int = 150) -> str:
    """Render a pool spec to a line-drawing PNG using matplotlib."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    for edge in spec.edges:
        ax.plot(
            [edge.x1, edge.x2],
            [edge.y1, edge.y2],
            "k-",
            linewidth=1.5,
        )
        if edge.label:
            mx, my = edge.midpoint
            dx = edge.x2 - edge.x1
            dy = edge.y2 - edge.y1
            length = edge.length
            if length > 0:
                nx, ny = -dy / length, dx / length
                tx = mx + nx * 14
                ty = my + ny * 14
                angle = math.degrees(math.atan2(dy, dx))
                ax.text(
                    tx, ty, edge.label,
                    fontsize=7, ha="center", va="center",
                    rotation=angle, color="red",
                )

    for stair in spec.stairs:
        td = stair.depth / stair.num_treads
        for i in range(stair.num_treads + 1):
            y = stair.y - i * td
            ax.plot([stair.x, stair.x + stair.width], [y, y], "g-", linewidth=1)
        ax.plot([stair.x, stair.x], [stair.y, stair.y - stair.depth], "g-", linewidth=1)
        ax.plot(
            [stair.x + stair.width, stair.x + stair.width],
            [stair.y, stair.y - stair.depth],
            "g-", linewidth=1,
        )

    ax.set_title(spec.name, fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _compute_bbox_dimensions(spec: PoolSpec) -> tuple[float, float]:
    """Compute bounding-box length (x-extent) and width (y-extent) in inches."""
    if not spec.edges:
        return (0.0, 0.0)
    xs = [e.x1 for e in spec.edges] + [e.x2 for e in spec.edges]
    ys = [e.y1 for e in spec.edges] + [e.y2 for e in spec.edges]
    return (max(xs) - min(xs), max(ys) - min(ys))


def generate_all():
    """Generate all sample reference data."""
    for folder_name, make_fn in SAMPLE_POOLS.items():
        folder = os.path.join(SAMPLE_DIR, folder_name)
        os.makedirs(folder, exist_ok=True)

        spec = make_fn()

        dxf_path = os.path.join(folder, "final.dxf")
        save_pool_dxf(spec, dxf_path)

        png_path = os.path.join(folder, "line_drawing.png")
        render_pool_png(spec, png_path)

        length_inches, width_inches = _compute_bbox_dimensions(spec)

        meta = {
            "name": spec.name,
            "pool_type": spec.pool_type,
            "num_edges": len(spec.edges),
            "has_stairs": len(spec.stairs) > 0,
            "length_inches": round(length_inches, 1),
            "width_inches": round(width_inches, 1),
            "shallow_depth": spec.shallow_depth,
            "deep_depth": spec.deep_depth,
        }
        meta_path = os.path.join(folder, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  {folder_name}: {spec.name} ({spec.pool_type}) "
              f"[{length_inches:.0f}\" x {width_inches:.0f}\"]")

    print(f"\nGenerated {len(SAMPLE_POOLS)} reference pools in {SAMPLE_DIR}")


if __name__ == "__main__":
    generate_all()
