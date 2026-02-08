"""
DXF layer management for pool drawings.

Creates all 7 pool layers with proper colors and linetypes.
"""

from ezdxf.document import Drawing

from sketch_to_cad.domain.constants import POOL_LAYERS, DASHED_LAYERS


def setup_layers(doc: Drawing) -> None:
    """Create all pool layers in a DXF document."""
    for name, color in POOL_LAYERS.items():
        if name not in doc.layers:
            attrs = {"color": color}
            if name in DASHED_LAYERS:
                attrs["linetype"] = "DASHED"
            doc.layers.add(name, dxfattribs=attrs)
