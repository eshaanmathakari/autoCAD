"""
Pydantic models for geometry representation.

These models define the JSON schema used for:
1. Gemini structured output (response_schema)
2. Intermediate geometry representation
3. Input to DXF synthesizer

Coordinate system: 0-1000 normalized (Gemini's native format)
- (0, 0) = top-left corner
- (1000, 1000) = bottom-right corner
"""

from pydantic import BaseModel, Field, field_validator
from typing import Annotated, Literal, Union
from enum import Enum


class LayerType(str, Enum):
    """Standard CAD layers for organizing entities."""
    GEOMETRY = "GEOMETRY"
    TEXT = "TEXT"
    DIMENSIONS = "DIMENSIONS"
    CENTERLINES = "CENTERLINES"
    HIDDEN = "HIDDEN"
    CONSTRUCTION = "CONSTRUCTION"


class Point2D(BaseModel):
    """2D coordinate point in normalized 0-1000 space."""
    x: Annotated[int, Field(ge=0, le=1000, description="X coordinate (0-1000)")]
    y: Annotated[int, Field(ge=0, le=1000, description="Y coordinate (0-1000)")]


class LineEntity(BaseModel):
    """Straight line segment between two points."""
    type: Literal["line"] = "line"
    start: Point2D = Field(..., description="Start point of the line")
    end: Point2D = Field(..., description="End point of the line")
    layer: LayerType = Field(default=LayerType.GEOMETRY, description="Layer assignment")
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=1.0, description="Detection confidence (0-1)"
    )


class CircleEntity(BaseModel):
    """Full circle defined by center and radius."""
    type: Literal["circle"] = "circle"
    center: Point2D = Field(..., description="Center point of the circle")
    radius: Annotated[int, Field(ge=1, le=500, description="Radius in normalized units")]
    layer: LayerType = Field(default=LayerType.GEOMETRY)
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0


class ArcEntity(BaseModel):
    """Circular arc segment."""
    type: Literal["arc"] = "arc"
    center: Point2D = Field(..., description="Center point of the arc")
    radius: Annotated[int, Field(ge=1, le=500)]
    start_angle: Annotated[float, Field(ge=0.0, le=360.0, description="Start angle in degrees")]
    end_angle: Annotated[float, Field(ge=0.0, le=360.0, description="End angle in degrees")]
    layer: LayerType = Field(default=LayerType.GEOMETRY)
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0


class RectangleEntity(BaseModel):
    """Axis-aligned rectangle."""
    type: Literal["rectangle"] = "rectangle"
    top_left: Point2D = Field(..., description="Top-left corner")
    width: Annotated[int, Field(ge=1, le=1000, description="Width in normalized units")]
    height: Annotated[int, Field(ge=1, le=1000, description="Height in normalized units")]
    layer: LayerType = Field(default=LayerType.GEOMETRY)
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0


class PolylineEntity(BaseModel):
    """Connected series of line segments."""
    type: Literal["polyline"] = "polyline"
    points: list[Point2D] = Field(
        ..., min_length=2, description="Ordered list of vertices"
    )
    closed: bool = Field(default=False, description="Whether the polyline forms a closed shape")
    layer: LayerType = Field(default=LayerType.GEOMETRY)
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0


class TextEntity(BaseModel):
    """Text annotation or label."""
    type: Literal["text"] = "text"
    position: Point2D = Field(..., description="Anchor position for the text")
    content: str = Field(..., min_length=1, description="Text content")
    height: Annotated[int, Field(ge=1, le=100, default=20)] = Field(
        description="Text height in normalized units"
    )
    rotation: Annotated[float, Field(ge=0.0, le=360.0, default=0.0)] = Field(
        description="Rotation angle in degrees"
    )
    layer: LayerType = Field(default=LayerType.TEXT)
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0


class DimensionEntity(BaseModel):
    """Linear dimension with measurement annotation."""
    type: Literal["dimension"] = "dimension"
    start: Point2D = Field(..., description="First measurement point")
    end: Point2D = Field(..., description="Second measurement point")
    text_position: Point2D | None = Field(
        default=None, description="Position of dimension text (optional)"
    )
    value: str | None = Field(
        default=None, description="Dimension value as string (e.g., '50mm', '2.5\"')"
    )
    unit: str = Field(default="mm", description="Unit of measurement")
    layer: LayerType = Field(default=LayerType.DIMENSIONS)
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0


# Union type for all geometry entities
GeometryEntity = Union[
    LineEntity,
    CircleEntity,
    ArcEntity,
    RectangleEntity,
    PolylineEntity,
    TextEntity,
    DimensionEntity,
]


class SketchMetadata(BaseModel):
    """Metadata about the analyzed sketch."""
    title: str | None = Field(default=None, description="Title if found in sketch")
    drawing_scale: str | None = Field(
        default=None, description="Scale notation if present (e.g., '1:100')"
    )
    detected_units: str = Field(default="mm", description="Detected or assumed units")
    sketch_type: str = Field(
        default="2d_orthographic",
        description="Type of sketch (2d_orthographic, isometric, perspective)"
    )
    image_width: int | None = Field(default=None, description="Original image width in pixels")
    image_height: int | None = Field(default=None, description="Original image height in pixels")
    confidence_score: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ..., description="Overall extraction confidence (0-1)"
    )


class SketchGeometry(BaseModel):
    """
    Complete geometry extraction result from a sketch.

    This is the root model used for:
    - Gemini structured output schema
    - Intermediate storage format
    - Input to DXF synthesis
    """
    metadata: SketchMetadata = Field(..., description="Sketch metadata and properties")
    entities: list[GeometryEntity] = Field(
        default_factory=list, description="List of extracted geometric entities"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Warnings or issues during extraction"
    )

    @field_validator("entities")
    @classmethod
    def validate_entities(cls, v: list) -> list:
        """Warn if no entities detected."""
        if len(v) == 0:
            pass  # Allow empty for graceful handling
        return v

    def get_entities_by_type(self, entity_type: str) -> list[GeometryEntity]:
        """Filter entities by type."""
        return [e for e in self.entities if e.type == entity_type]

    def get_entities_by_layer(self, layer: LayerType) -> list[GeometryEntity]:
        """Filter entities by layer."""
        return [e for e in self.entities if e.layer == layer]

    @property
    def entity_counts(self) -> dict[str, int]:
        """Count entities by type."""
        counts: dict[str, int] = {}
        for entity in self.entities:
            counts[entity.type] = counts.get(entity.type, 0) + 1
        return counts
