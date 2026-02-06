"""
Pool-specific DXF generator for the PoC.

Generates pool outline DXFs with proper layers for swimming pool lining:
  POOL_OUTLINE   - Main pool perimeter
  POOL_STAIRS    - Stair treads / risers
  POOL_DIMENSIONS - Dimension annotations
  POOL_TEXT      - Labels
"""

import os
import sys
import tempfile
import math
from dataclasses import dataclass, field
from typing import Optional

import ezdxf
from ezdxf.document import Drawing
from ezdxf.layouts import Modelspace

# Pool-specific layer colors (AutoCAD Color Index)
POOL_LAYERS = {
    "POOL_OUTLINE": 7,      # White
    "POOL_STAIRS": 3,       # Green
    "POOL_DIMENSIONS": 1,   # Red
    "POOL_TEXT": 4,          # Cyan
    "POOL_BENCH": 2,        # Yellow
    "POOL_EQUIPMENT": 8,    # Gray
}


@dataclass
class PoolEdge:
    """A single edge of the pool outline."""
    x1: float
    y1: float
    x2: float
    y2: float
    label: Optional[str] = None  # dimension label, e.g. "12'6\""

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def midpoint(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class StairSpec:
    """Stair definition."""
    x: float          # top-left x
    y: float          # top-left y
    width: float      # across the pool wall
    depth: float      # into the pool
    num_treads: int = 3
    direction: str = "right"  # which direction stairs face


@dataclass
class PoolSpec:
    """Complete pool specification for DXF generation."""
    name: str
    pool_type: str  # rectangular, l_shaped, kidney, freeform
    edges: list[PoolEdge] = field(default_factory=list)
    stairs: list[StairSpec] = field(default_factory=list)
    shallow_depth: str = "3'6\""
    deep_depth: str = "8'"
    units: str = "inches"  # internal units for coordinates


def setup_layers(doc: Drawing) -> None:
    """Create pool-specific layers."""
    for name, color in POOL_LAYERS.items():
        if name not in doc.layers:
            doc.layers.add(name, dxfattribs={"color": color})


def add_pool_outline(msp: Modelspace, edges: list[PoolEdge]) -> None:
    """Draw pool outline as closed polyline."""
    if not edges:
        return
    points = [(e.x1, e.y1) for e in edges]
    msp.add_lwpolyline(points, close=True, dxfattribs={"layer": "POOL_OUTLINE"})


def add_dimensions(msp: Modelspace, edges: list[PoolEdge], offset: float = 18) -> None:
    """Add dimension annotations along edges."""
    for edge in edges:
        if not edge.label:
            continue
        mx, my = edge.midpoint
        # offset perpendicular
        dx = edge.x2 - edge.x1
        dy = edge.y2 - edge.y1
        length = edge.length
        if length == 0:
            continue
        nx, ny = -dy / length, dx / length
        tx = mx + nx * offset
        ty = my + ny * offset
        # rotation
        angle = math.degrees(math.atan2(dy, dx))
        msp.add_text(
            edge.label,
            dxfattribs={
                "layer": "POOL_DIMENSIONS",
                "height": 6,
                "rotation": angle,
            },
        ).set_placement((tx, ty))


def add_stairs(msp: Modelspace, stair: StairSpec) -> None:
    """Draw stair treads."""
    tread_depth = stair.depth / stair.num_treads
    for i in range(stair.num_treads + 1):
        y = stair.y - i * tread_depth
        msp.add_line(
            (stair.x, y),
            (stair.x + stair.width, y),
            dxfattribs={"layer": "POOL_STAIRS"},
        )
    # side lines
    msp.add_line(
        (stair.x, stair.y),
        (stair.x, stair.y - stair.depth),
        dxfattribs={"layer": "POOL_STAIRS"},
    )
    msp.add_line(
        (stair.x + stair.width, stair.y),
        (stair.x + stair.width, stair.y - stair.depth),
        dxfattribs={"layer": "POOL_STAIRS"},
    )


def add_label(msp: Modelspace, text: str, x: float, y: float, height: float = 8) -> None:
    """Add a text label."""
    msp.add_text(
        text,
        dxfattribs={"layer": "POOL_TEXT", "height": height},
    ).set_placement((x, y))


def generate_pool_dxf(spec: PoolSpec, version: str = "R2010") -> bytes:
    """Generate a DXF file from a PoolSpec. Returns bytes."""
    doc = ezdxf.new(dxfversion=version, setup=True)
    msp = doc.modelspace()
    setup_layers(doc)

    add_pool_outline(msp, spec.edges)
    add_dimensions(msp, spec.edges)
    for stair in spec.stairs:
        add_stairs(msp, stair)

    # center label
    if spec.edges:
        xs = [e.x1 for e in spec.edges]
        ys = [e.y1 for e in spec.edges]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        add_label(msp, spec.name, cx - 20, cy)

    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        doc.saveas(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def save_pool_dxf(spec: PoolSpec, path: str) -> str:
    """Generate and save a pool DXF to disk."""
    data = generate_pool_dxf(spec)
    with open(path, "wb") as f:
        f.write(data)
    return path


# ---------------------------------------------------------------------------
# Pre-built pool specs for sample data
# ---------------------------------------------------------------------------

def make_rectangular_pool(
    length: float = 240,  # 20'
    width: float = 120,   # 10'
    name: str = "Rectangular Pool",
) -> PoolSpec:
    """Simple rectangular pool."""
    edges = [
        PoolEdge(0, 0, length, 0, label=f"{length/12:.0f}'"),
        PoolEdge(length, 0, length, width, label=f"{width/12:.0f}'"),
        PoolEdge(length, width, 0, width, label=f"{length/12:.0f}'"),
        PoolEdge(0, width, 0, 0, label=f"{width/12:.0f}'"),
    ]
    stairs = [StairSpec(x=0, y=width, width=48, depth=36, num_treads=3)]
    return PoolSpec(name=name, pool_type="rectangular", edges=edges, stairs=stairs)


def make_l_shaped_pool(
    long_len: float = 300,   # 25'
    long_width: float = 144, # 12'
    short_len: float = 144,  # 12'
    short_width: float = 96, # 8'
    name: str = "L-Shaped Pool",
) -> PoolSpec:
    """L-shaped pool (rectangular with a notch)."""
    # Shape:
    #  ┌──────────┐
    #  │          │
    #  │    ┌─────┘
    #  │    │
    #  └────┘
    edges = [
        PoolEdge(0, 0, long_len, 0, label=f"{long_len/12:.0f}'"),
        PoolEdge(long_len, 0, long_len, short_width, label=f"{short_width/12:.0f}'"),
        PoolEdge(long_len, short_width, short_len, short_width, label=f"{(long_len - short_len)/12:.0f}'"),
        PoolEdge(short_len, short_width, short_len, long_width, label=f"{(long_width - short_width)/12:.0f}'"),
        PoolEdge(short_len, long_width, 0, long_width, label=f"{short_len/12:.0f}'"),
        PoolEdge(0, long_width, 0, 0, label=f"{long_width/12:.0f}'"),
    ]
    stairs = [StairSpec(x=0, y=long_width, width=48, depth=36, num_treads=3)]
    return PoolSpec(name=name, pool_type="l_shaped", edges=edges, stairs=stairs)


def make_kidney_pool(
    length: float = 360,   # 30'
    width: float = 180,    # 15'
    segments: int = 40,
    name: str = "Kidney Pool",
) -> PoolSpec:
    """Kidney-shaped pool approximated with polyline edges."""
    edges = []
    hw = width / 2
    hl = length / 2
    indent = width * 0.15  # kidney indent

    for i in range(segments):
        t0 = 2 * math.pi * i / segments
        t1 = 2 * math.pi * (i + 1) / segments

        def kidney_point(t):
            x = hl * math.cos(t)
            r = hw - indent * math.cos(t)
            y = r * math.sin(t)
            return (x + hl, y + hw)

        p0 = kidney_point(t0)
        p1 = kidney_point(t1)
        label = None
        if i == 0:
            label = f"{length/12:.0f}'"
        elif i == segments // 4:
            label = f"{width/12:.0f}'"
        edges.append(PoolEdge(p0[0], p0[1], p1[0], p1[1], label=label))

    return PoolSpec(name=name, pool_type="kidney", edges=edges)


def make_freeform_pool(
    name: str = "Freeform Pool",
) -> PoolSpec:
    """Irregular freeform pool shape."""
    # Hand-specified vertices (in inches)
    pts = [
        (0, 0), (60, -24), (180, -12), (300, 0),
        (336, 48), (300, 120), (240, 156),
        (120, 168), (36, 144), (0, 96),
    ]
    edges = []
    for i in range(len(pts)):
        p0 = pts[i]
        p1 = pts[(i + 1) % len(pts)]
        label = None
        if i == 0:
            label = "28'"
        elif i == 3:
            label = "14'"
        edges.append(PoolEdge(p0[0], p0[1], p1[0], p1[1], label=label))

    stairs = [StairSpec(x=0, y=96, width=48, depth=36, num_treads=3)]
    return PoolSpec(name=name, pool_type="freeform", edges=edges, stairs=stairs)


def make_rectangular_with_spa(
    pool_len: float = 240,
    pool_width: float = 120,
    spa_diam: float = 72,  # 6'
    name: str = "Pool with Spa",
) -> PoolSpec:
    """Rectangular pool with an attached circular spa (approximated)."""
    # Main pool
    edges = [
        PoolEdge(0, 0, pool_len, 0, label=f"{pool_len/12:.0f}'"),
        PoolEdge(pool_len, 0, pool_len, pool_width, label=f"{pool_width/12:.0f}'"),
        PoolEdge(pool_len, pool_width, 0, pool_width, label=f"{pool_len/12:.0f}'"),
        PoolEdge(0, pool_width, 0, 0, label=f"{pool_width/12:.0f}'"),
    ]
    # Spa as polygon (approximate circle at top-right)
    spa_cx = pool_len + spa_diam / 2 + 12
    spa_cy = pool_width / 2
    spa_r = spa_diam / 2
    segs = 20
    spa_edges = []
    for i in range(segs):
        t0 = 2 * math.pi * i / segs
        t1 = 2 * math.pi * (i + 1) / segs
        x0 = spa_cx + spa_r * math.cos(t0)
        y0 = spa_cy + spa_r * math.sin(t0)
        x1 = spa_cx + spa_r * math.cos(t1)
        y1 = spa_cy + spa_r * math.sin(t1)
        label = f"{spa_diam/12:.0f}' dia" if i == 0 else None
        spa_edges.append(PoolEdge(x0, y0, x1, y1, label=label))

    stairs = [StairSpec(x=0, y=pool_width, width=48, depth=36, num_treads=3)]
    return PoolSpec(
        name=name,
        pool_type="rectangular",
        edges=edges + spa_edges,
        stairs=stairs,
    )


# All sample specs
SAMPLE_POOLS = {
    "reference_001": make_rectangular_pool,
    "reference_002": make_l_shaped_pool,
    "reference_003": make_kidney_pool,
    "reference_004": make_freeform_pool,
    "reference_005": make_rectangular_with_spa,
}
