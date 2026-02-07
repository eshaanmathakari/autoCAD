"""
Template deformer for the PoC.

Takes a reference pool DXF and a set of target dimensions,
then proportionally scales the reference geometry to match.

PoC approach: simple axis-aligned bounding-box scaling.
Production would use a constraint solver (scipy.optimize).
"""

import os
import math
import tempfile
from dataclasses import dataclass
from typing import Optional

import ezdxf

from poc.pool_dxf_generator import PoolEdge


@dataclass
class DeformationResult:
    """Result of template deformation."""
    edges: list[PoolEdge]
    scale_x: float
    scale_y: float
    original_bbox: tuple[float, float, float, float]  # minx, miny, maxx, maxy
    target_bbox: tuple[float, float, float, float]
    dxf_bytes: Optional[bytes] = None


def load_dxf_edges(dxf_path: str) -> list[PoolEdge]:
    """
    Load LINE and LWPOLYLINE entities from a DXF file as PoolEdges.
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    edges = []

    for entity in msp:
        if entity.dxftype() == "LINE":
            s = entity.dxf.start
            e = entity.dxf.end
            edges.append(PoolEdge(s.x, s.y, e.x, e.y))
        elif entity.dxftype() == "LWPOLYLINE":
            pts = list(entity.get_points(format="xy"))
            if entity.closed:
                pts.append(pts[0])
            for i in range(len(pts) - 1):
                edges.append(PoolEdge(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]))

    return edges


def compute_bbox(edges: list[PoolEdge]) -> tuple[float, float, float, float]:
    """Compute axis-aligned bounding box (minx, miny, maxx, maxy)."""
    if not edges:
        return (0, 0, 0, 0)
    xs = [e.x1 for e in edges] + [e.x2 for e in edges]
    ys = [e.y1 for e in edges] + [e.y2 for e in edges]
    return (min(xs), min(ys), max(xs), max(ys))


def deform_to_dimensions(
    reference_dxf_path: str,
    target_length_inches: float,
    target_width_inches: float,
    dimension_labels: Optional[dict[str, str]] = None,
) -> DeformationResult:
    """
    Scale a reference DXF to match target overall dimensions.

    Args:
        reference_dxf_path: Path to reference DXF file
        target_length_inches: Target pool length in inches
        target_width_inches: Target pool width in inches
        dimension_labels: Optional dict mapping edge indices to labels

    Returns:
        DeformationResult with scaled edges and DXF bytes
    """
    # Load reference edges
    ref_edges = load_dxf_edges(reference_dxf_path)
    if not ref_edges:
        return DeformationResult(
            edges=[], scale_x=1, scale_y=1,
            original_bbox=(0, 0, 0, 0),
            target_bbox=(0, 0, target_length_inches, target_width_inches),
        )

    # Compute reference bounding box
    minx, miny, maxx, maxy = compute_bbox(ref_edges)
    ref_w = maxx - minx
    ref_h = maxy - miny

    if ref_w == 0 or ref_h == 0:
        return DeformationResult(
            edges=ref_edges, scale_x=1, scale_y=1,
            original_bbox=(minx, miny, maxx, maxy),
            target_bbox=(0, 0, target_length_inches, target_width_inches),
        )

    # Compute scale factors
    scale_x = target_length_inches / ref_w
    scale_y = target_width_inches / ref_h

    # Scale all edges relative to origin (minx, miny)
    scaled_edges = []
    for i, e in enumerate(ref_edges):
        sx1 = (e.x1 - minx) * scale_x
        sy1 = (e.y1 - miny) * scale_y
        sx2 = (e.x2 - minx) * scale_x
        sy2 = (e.y2 - miny) * scale_y
        label = None
        if dimension_labels and str(i) in dimension_labels:
            label = dimension_labels[str(i)]
        else:
            # Auto-generate label from scaled edge length
            edge_len = math.hypot(sx2 - sx1, sy2 - sy1)
            if edge_len >= 6:  # Only label edges >= 6 inches
                feet = int(edge_len // 12)
                inches = round(edge_len % 12, 1)
                if feet > 0 and inches > 0:
                    label = f"{feet}'{inches:.0f}\""
                elif feet > 0:
                    label = f"{feet}'"
                else:
                    label = f'{inches:.0f}"'
        scaled_edges.append(PoolEdge(sx1, sy1, sx2, sy2, label=label))

    # Generate scaled DXF
    dxf_bytes = _generate_scaled_dxf(scaled_edges)

    target_bbox = (0, 0, target_length_inches, target_width_inches)

    return DeformationResult(
        edges=scaled_edges,
        scale_x=scale_x,
        scale_y=scale_y,
        original_bbox=(minx, miny, maxx, maxy),
        target_bbox=target_bbox,
        dxf_bytes=dxf_bytes,
    )


def _generate_scaled_dxf(edges: list[PoolEdge]) -> bytes:
    """Generate a DXF from scaled edges."""
    doc = ezdxf.new(dxfversion="R2010", setup=True)
    msp = doc.modelspace()

    # Setup layers
    from poc.pool_dxf_generator import POOL_LAYERS
    for name, color in POOL_LAYERS.items():
        if name not in doc.layers:
            doc.layers.add(name, dxfattribs={"color": color})

    # Add edges
    for edge in edges:
        msp.add_line(
            (edge.x1, edge.y1),
            (edge.x2, edge.y2),
            dxfattribs={"layer": "POOL_OUTLINE"},
        )
        # Add dimension label if present
        if edge.label:
            mx, my = edge.midpoint
            dx = edge.x2 - edge.x1
            dy = edge.y2 - edge.y1
            length = edge.length
            if length > 0:
                nx, ny = -dy / length, dx / length
                angle = math.degrees(math.atan2(dy, dx))
                msp.add_text(
                    edge.label,
                    dxfattribs={
                        "layer": "POOL_DIMENSIONS",
                        "height": 6,
                        "rotation": angle,
                    },
                ).set_placement((mx + nx * 12, my + ny * 12))

    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        doc.saveas(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)
