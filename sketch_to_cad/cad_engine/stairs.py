"""
Stair generation for pool DXF drawings.
"""

from ezdxf.layouts import Modelspace


def add_stairs(msp: Modelspace, stair) -> None:
    """Draw stair treads and side lines on the STAIRS layer."""
    tread_depth = stair.depth / stair.num_treads
    for i in range(stair.num_treads + 1):
        y = stair.y - i * tread_depth
        msp.add_line(
            (stair.x, y),
            (stair.x + stair.width, y),
            dxfattribs={"layer": "STAIRS"},
        )
    # side lines
    msp.add_line(
        (stair.x, stair.y),
        (stair.x, stair.y - stair.depth),
        dxfattribs={"layer": "STAIRS"},
    )
    msp.add_line(
        (stair.x + stair.width, stair.y),
        (stair.x + stair.width, stair.y - stair.depth),
        dxfattribs={"layer": "STAIRS"},
    )
