"""
DXF file generation from extracted geometry.

Uses ezdxf library to create AutoCAD-compatible DXF files
from the structured geometry representation.
"""

import os
import tempfile
from typing import Callable

import ezdxf
from ezdxf.document import Drawing
from ezdxf.layouts import Modelspace

from typing import Optional

from .config import (
    DEFAULT_DRAWING_WIDTH,
    DEFAULT_DRAWING_HEIGHT,
    LAYER_COLORS,
)
from .geometry_models import (
    SketchGeometry,
    GeometryEntity,
    LineEntity,
    CircleEntity,
    ArcEntity,
    RectangleEntity,
    PolylineEntity,
    TextEntity,
    DimensionEntity,
    Point2D,
    LayerType,
)
from .calibration import CalibrationResult


class DXFSynthesizer:
    """
    Generate DXF files from extracted geometry.

    Converts normalized coordinates (0-1000) to drawing units (mm by default)
    and creates properly layered DXF output.
    """

    def __init__(
        self,
        version: str = "R2010",
        drawing_width: float = DEFAULT_DRAWING_WIDTH,
        drawing_height: float = DEFAULT_DRAWING_HEIGHT,
    ):
        """
        Initialize DXF synthesizer.

        Args:
            version: DXF version (R12, R2000, R2004, R2007, R2010, R2013, R2018)
            drawing_width: Output drawing width in units (default: 297mm = A4)
            drawing_height: Output drawing height in units (default: 210mm = A4)
        """
        self.version = version.upper()
        self.drawing_width = drawing_width
        self.drawing_height = drawing_height
        self.errors: list[str] = []

    def _denormalize_x(self, x: int, scale: float = 1.0) -> float:
        """Convert normalized X (0-1000) to drawing units."""
        return (x / 1000.0) * self.drawing_width * scale

    def _denormalize_y(self, y: int, scale: float = 1.0) -> float:
        """
        Convert normalized Y (0-1000) to drawing units.

        Note: Y is flipped because image coordinates have Y increasing downward,
        but CAD coordinates have Y increasing upward.
        """
        return (1.0 - y / 1000.0) * self.drawing_height * scale

    def _denormalize_point(self, point: Point2D, scale: float = 1.0) -> tuple[float, float]:
        """Convert Point2D to drawing coordinates."""
        return (
            self._denormalize_x(point.x, scale),
            self._denormalize_y(point.y, scale)
        )

    def _denormalize_radius(self, radius: int, scale: float = 1.0) -> float:
        """Convert normalized radius to drawing units."""
        # Use average of width/height for radius scaling
        avg_size = (self.drawing_width + self.drawing_height) / 2
        return (radius / 1000.0) * avg_size * scale

    def _denormalize_length(self, length: int, scale: float = 1.0) -> float:
        """Convert normalized length to drawing units (width-based)."""
        return (length / 1000.0) * self.drawing_width * scale

    def _denormalize_height(self, height: int, scale: float = 1.0) -> float:
        """Convert normalized height to drawing units (height-based)."""
        return (height / 1000.0) * self.drawing_height * scale

    def _setup_layers(self, doc: Drawing) -> None:
        """Create standard layers with colors."""
        for layer_name, color in LAYER_COLORS.items():
            if layer_name not in doc.layers:
                doc.layers.add(layer_name, dxfattribs={"color": color})

    def _get_layer_name(self, layer: LayerType) -> str:
        """Get DXF layer name from LayerType enum."""
        return layer.value

    def _add_line(self, msp: Modelspace, entity: LineEntity, scale: float) -> None:
        """Add LINE entity to modelspace."""
        start = self._denormalize_point(entity.start, scale)
        end = self._denormalize_point(entity.end, scale)
        layer = self._get_layer_name(entity.layer)
        msp.add_line(start, end, dxfattribs={"layer": layer})

    def _add_circle(self, msp: Modelspace, entity: CircleEntity, scale: float) -> None:
        """Add CIRCLE entity to modelspace."""
        center = self._denormalize_point(entity.center, scale)
        radius = self._denormalize_radius(entity.radius, scale)
        layer = self._get_layer_name(entity.layer)
        msp.add_circle(center, radius=radius, dxfattribs={"layer": layer})

    def _add_arc(self, msp: Modelspace, entity: ArcEntity, scale: float) -> None:
        """Add ARC entity to modelspace."""
        center = self._denormalize_point(entity.center, scale)
        radius = self._denormalize_radius(entity.radius, scale)
        layer = self._get_layer_name(entity.layer)

        # ezdxf expects counter-clockwise arcs
        # Since Y is flipped, we need to adjust angles
        # start_angle and end_angle should be swapped and negated
        adjusted_start = -entity.end_angle
        adjusted_end = -entity.start_angle

        # Normalize to 0-360 range
        while adjusted_start < 0:
            adjusted_start += 360
        while adjusted_end < 0:
            adjusted_end += 360

        msp.add_arc(
            center,
            radius=radius,
            start_angle=adjusted_start,
            end_angle=adjusted_end,
            dxfattribs={"layer": layer}
        )

    def _add_rectangle(self, msp: Modelspace, entity: RectangleEntity, scale: float) -> None:
        """Add RECTANGLE as closed LWPOLYLINE."""
        top_left = self._denormalize_point(entity.top_left, scale)
        width = self._denormalize_length(entity.width, scale)
        height = self._denormalize_height(entity.height, scale)
        layer = self._get_layer_name(entity.layer)

        # Since Y is flipped, "top" in image coords becomes "bottom" in CAD
        # and height goes upward
        points = [
            top_left,
            (top_left[0] + width, top_left[1]),
            (top_left[0] + width, top_left[1] - height),
            (top_left[0], top_left[1] - height),
        ]

        msp.add_lwpolyline(points, close=True, dxfattribs={"layer": layer})

    def _add_polyline(self, msp: Modelspace, entity: PolylineEntity, scale: float) -> None:
        """Add LWPOLYLINE entity to modelspace."""
        points = [self._denormalize_point(p, scale) for p in entity.points]
        layer = self._get_layer_name(entity.layer)
        msp.add_lwpolyline(points, close=entity.closed, dxfattribs={"layer": layer})

    def _add_text(self, msp: Modelspace, entity: TextEntity, scale: float) -> None:
        """Add TEXT entity to modelspace."""
        position = self._denormalize_point(entity.position, scale)
        height = self._denormalize_height(entity.height, scale)
        layer = self._get_layer_name(entity.layer)

        # Ensure minimum readable height
        height = max(height, 2.0)

        # Adjust rotation for Y-flip
        rotation = -entity.rotation if entity.rotation != 0 else 0

        text = msp.add_text(
            entity.content,
            dxfattribs={
                "layer": layer,
                "height": height,
                "rotation": rotation,
            }
        )
        text.set_placement(position)

    def _add_dimension(self, msp: Modelspace, entity: DimensionEntity, scale: float) -> None:
        """Add linear DIMENSION entity to modelspace."""
        start = self._denormalize_point(entity.start, scale)
        end = self._denormalize_point(entity.end, scale)
        layer = self._get_layer_name(entity.layer)

        # Calculate dimension line position (offset from measured points)
        # Use midpoint with vertical offset
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2

        # Offset perpendicular to the line
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        # Default offset distance
        offset = self.drawing_height * 0.05  # 5% of drawing height

        # Add dimension
        try:
            dim = msp.add_linear_dim(
                base=(mid_x, mid_y + offset),
                p1=start,
                p2=end,
                dxfattribs={"layer": layer}
            )
            dim.render()
        except Exception as e:
            # Fall back to simple text annotation if dimension fails
            self.errors.append(f"Dimension creation failed, using text: {e}")
            if entity.value:
                msp.add_text(
                    entity.value,
                    dxfattribs={
                        "layer": layer,
                        "height": 2.5,
                    }
                ).set_placement((mid_x, mid_y + offset))

    def generate(self, geometry: SketchGeometry, scale: float = 1.0) -> bytes:
        """
        Generate DXF file from extracted geometry.

        Args:
            geometry: Validated SketchGeometry object
            scale: Scale factor for output (1.0 = 1:1)

        Returns:
            DXF file as bytes
        """
        self.errors = []  # Reset errors

        # Create new DXF document
        doc = ezdxf.new(dxfversion=self.version, setup=True)
        msp = doc.modelspace()

        # Setup layers
        self._setup_layers(doc)

        # Entity handler dispatch table
        handlers: dict[str, Callable] = {
            "line": self._add_line,
            "circle": self._add_circle,
            "arc": self._add_arc,
            "rectangle": self._add_rectangle,
            "polyline": self._add_polyline,
            "text": self._add_text,
            "dimension": self._add_dimension,
        }

        # Process all entities
        for entity in geometry.entities:
            handler = handlers.get(entity.type)
            if handler:
                try:
                    handler(msp, entity, scale)
                except Exception as e:
                    self.errors.append(f"Failed to add {entity.type}: {e}")

        # Export to bytes via temp file (ezdxf.saveas doesn't work with BytesIO)
        with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tmp:
            temp_path = tmp.name

        try:
            doc.saveas(temp_path)
            with open(temp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def generate_to_file(
        self,
        geometry: SketchGeometry,
        output_path: str,
        scale: float = 1.0
    ) -> str:
        """
        Generate DXF file and save to disk.

        Args:
            geometry: Validated SketchGeometry object
            output_path: Path to save the DXF file
            scale: Scale factor for output

        Returns:
            Path to the saved file
        """
        dxf_bytes = self.generate(geometry, scale)

        with open(output_path, "wb") as f:
            f.write(dxf_bytes)

        return output_path

    def get_errors(self) -> list[str]:
        """Get list of errors encountered during generation."""
        return self.errors.copy()

    def generate_with_calibration(
        self,
        geometry: SketchGeometry,
        calibration: Optional[CalibrationResult] = None,
        user_scale: float = 1.0
    ) -> bytes:
        """
        Generate DXF file with automatic scale calibration.

        Uses the calibration result to determine the correct scale factor
        so that dimensions in the output match the real-world measurements
        from the original sketch.

        Args:
            geometry: Validated SketchGeometry object
            calibration: CalibrationResult from scale detection (optional)
            user_scale: Additional user-specified scale factor (default 1.0)

        Returns:
            DXF file as bytes
        """
        # Calculate effective scale
        if calibration:
            # The calibration tells us how many pixels per real-world mm
            # We need to convert from normalized (0-1000) to real-world mm
            #
            # normalized_coord / 1000 * drawing_width = output_mm
            # We want: output_mm = real_world_mm
            #
            # If scale_factor is the drawing scale (e.g., 100 for 1:100),
            # then the drawing represents real_world dimensions scaled down.
            # For a 1:100 drawing, a 10m wall appears as 100mm on paper.
            #
            # To output at 1:1, we multiply by scale_factor
            # To output at user's preferred scale, we then apply user_scale
            effective_scale = user_scale * calibration.scale_factor
        else:
            effective_scale = user_scale

        return self.generate(geometry, scale=effective_scale)

    @staticmethod
    def compute_scale_from_calibration(
        calibration: CalibrationResult,
        target_scale: float = 1.0
    ) -> float:
        """
        Compute the DXF scale factor from calibration data.

        Args:
            calibration: CalibrationResult with scale_factor
            target_scale: Target output scale (1.0 = same as detected,
                         0.5 = half size, 2.0 = double size)

        Returns:
            Scale factor to pass to generate()
        """
        return calibration.scale_factor * target_scale
