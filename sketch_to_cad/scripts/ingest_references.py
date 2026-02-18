#!/usr/bin/env python3
"""
Ingest external pool reference data into the templates directory.

Reads a folder of candidate references and normalises each one into
the standard template structure:

    templates/<reference_NNN>/
        line_drawing.png
        final.dxf        (optional)
        metadata.json

Usage:
    python -m sketch_to_cad.scripts.ingest_references /path/to/input_folder

Input folder layout (per reference):
    <name>/
        image.png  or  image.jpg   (required)
        drawing.dxf                (optional)
        meta.json                  (optional, fields merged into output)

If meta.json is not present, the script will prompt for pool_type
and dimensions, or leave them as unknown.

If only a DXF file is present (no image), the script will attempt
to render a line_drawing.png from the DXF using matplotlib.
"""

import json
import math
import os
import shutil
import sys
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

METADATA_SCHEMA = {
    "name": "",
    "pool_type": "unknown",
    "num_edges": 0,
    "has_stairs": False,
    "length_inches": None,
    "width_inches": None,
    "shallow_depth": "",
    "deep_depth": "",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DXF_EXTENSIONS = {".dxf"}


def _next_reference_id() -> str:
    """Determine the next reference_NNN folder id."""
    existing = sorted(
        d.name for d in TEMPLATES_DIR.iterdir()
        if d.is_dir() and d.name.startswith("reference_")
    )
    if not existing:
        return "reference_006"
    last_num = int(existing[-1].split("_")[1])
    return f"reference_{last_num + 1:03d}"


def _render_dxf_to_png(dxf_path: str, png_path: str) -> bool:
    """Render a DXF file to a line-drawing PNG using matplotlib."""
    try:
        import ezdxf
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_aspect("equal")
        ax.set_facecolor("white")

        for entity in msp:
            if entity.dxftype() == "LINE":
                s = entity.dxf.start
                e = entity.dxf.end
                ax.plot([s.x, e.x], [s.y, e.y], "k-", linewidth=1.5)
            elif entity.dxftype() == "LWPOLYLINE":
                pts = list(entity.get_points(format="xy"))
                if entity.closed:
                    pts.append(pts[0])
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, "k-", linewidth=1.5)

        ax.axis("off")
        plt.tight_layout()
        fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"  Warning: could not render DXF to PNG: {e}")
        return False


def _compute_bbox_from_dxf(dxf_path: str) -> tuple[float, float]:
    """Compute bounding-box dimensions from a DXF file."""
    try:
        import ezdxf
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        xs, ys = [], []
        for entity in msp:
            if entity.dxftype() == "LINE":
                s = entity.dxf.start
                e = entity.dxf.end
                xs.extend([s.x, e.x])
                ys.extend([s.y, e.y])
            elif entity.dxftype() == "LWPOLYLINE":
                for pt in entity.get_points(format="xy"):
                    xs.append(pt[0])
                    ys.append(pt[1])
        if xs and ys:
            return (max(xs) - min(xs), max(ys) - min(ys))
    except Exception:
        pass
    return (0.0, 0.0)


def ingest_folder(input_dir: str) -> int:
    """Ingest all sub-folders from input_dir into templates/.

    Returns the number of references ingested.
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"Error: {input_dir} is not a directory.")
        return 0

    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    candidates = sorted(input_path.iterdir())
    # If input_dir itself contains image files (no sub-folders), treat it as one ref
    sub_dirs = [c for c in candidates if c.is_dir()]
    if not sub_dirs:
        sub_dirs = [input_path]

    for candidate in sub_dirs:
        if not candidate.is_dir():
            continue

        # Find image
        image_file = None
        for f in candidate.iterdir():
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                image_file = f
                break

        # Find DXF
        dxf_file = None
        for f in candidate.iterdir():
            if f.suffix.lower() in DXF_EXTENSIONS:
                dxf_file = f
                break

        # Must have at least an image or a DXF
        if image_file is None and dxf_file is None:
            print(f"  Skipping {candidate.name}: no image or DXF found.")
            continue

        ref_id = _next_reference_id()
        ref_dir = TEMPLATES_DIR / ref_id
        ref_dir.mkdir(exist_ok=True)

        # Copy / generate line_drawing.png
        if image_file:
            shutil.copy2(str(image_file), str(ref_dir / "line_drawing.png"))
        elif dxf_file:
            rendered = _render_dxf_to_png(
                str(dxf_file), str(ref_dir / "line_drawing.png")
            )
            if not rendered:
                print(f"  Skipping {candidate.name}: could not render DXF.")
                shutil.rmtree(str(ref_dir))
                continue

        # Copy DXF if available
        if dxf_file:
            shutil.copy2(str(dxf_file), str(ref_dir / "final.dxf"))

        # Build metadata
        meta = dict(METADATA_SCHEMA)
        meta["name"] = candidate.name.replace("_", " ").replace("-", " ").title()

        # Load input meta.json if present
        input_meta_file = candidate / "meta.json"
        if not input_meta_file.exists():
            input_meta_file = candidate / "metadata.json"
        if input_meta_file.exists():
            try:
                with open(input_meta_file) as f:
                    input_meta = json.load(f)
                for key in METADATA_SCHEMA:
                    if key in input_meta and input_meta[key] is not None:
                        meta[key] = input_meta[key]
            except Exception as e:
                print(f"  Warning: could not read {input_meta_file}: {e}")

        # Compute dimensions from DXF if not in metadata
        if dxf_file and (meta["length_inches"] is None or meta["width_inches"] is None):
            length, width = _compute_bbox_from_dxf(str(dxf_file))
            if length > 0 and meta["length_inches"] is None:
                meta["length_inches"] = round(length, 1)
            if width > 0 and meta["width_inches"] is None:
                meta["width_inches"] = round(width, 1)

        with open(ref_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  Ingested: {candidate.name} -> {ref_id} ({meta['pool_type']})")
        count += 1

    return count


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m sketch_to_cad.scripts.ingest_references <input_folder>")
        print()
        print("Input folder should contain sub-folders, each with at least")
        print("an image (PNG/JPG) and optionally a DXF and meta.json.")
        sys.exit(1)

    input_dir = sys.argv[1]
    print(f"Ingesting references from: {input_dir}")
    print(f"Target templates directory: {TEMPLATES_DIR}")
    print()

    count = ingest_folder(input_dir)
    print(f"\nDone. Ingested {count} reference(s).")


if __name__ == "__main__":
    main()
