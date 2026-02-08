"""
Proper DXF DIMENSION entities for pool drawings.

Replaces TEXT-based dimension labels with real ALIGNED DIMENSION entities
that render with arrows, extension lines, and measurement text in AutoCAD.

Critical: always call dim.render() after creating dimension entities.
This generates the anonymous geometry block that AutoCAD requires.
"""

from ezdxf.document import Drawing
from ezdxf.layouts import Modelspace


def add_aligned_dimension(
    msp: Modelspace,
    doc: Drawing,
    start: tuple[float, float],
    end: tuple[float, float],
    offset: float = 18,
    text: str = "",
    layer: str = "DIMENSIONS",
) -> None:
    """
    Add an ALIGNED DIMENSION entity with arrows and extension lines.

    Args:
        msp: Modelspace to add the dimension to
        doc: DXF document (needed for dimension style)
        start: Start point (x, y)
        end: End point (x, y)
        offset: Distance of dimension line from the measured edge
        text: Override text (e.g. "12'6\""). Empty = auto-measure.
        layer: DXF layer name
    """
    override = {}
    if text:
        override["text"] = text

    dim = msp.add_aligned_dim(
        p1=start,
        p2=end,
        distance=offset,
        override=override,
        dxfattribs={"layer": layer},
    )
    dim.render()


def add_pool_dimensions(
    msp: Modelspace,
    doc: Drawing,
    edges,
    offset: float = 18,
) -> None:
    """Add DIMENSION entities for all labeled edges."""
    for edge in edges:
        if not edge.label:
            continue
        add_aligned_dimension(
            msp,
            doc,
            start=(edge.x1, edge.y1),
            end=(edge.x2, edge.y2),
            offset=offset,
            text=edge.label,
        )
