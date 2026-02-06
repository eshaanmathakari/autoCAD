"""
Local line extraction fallback using OpenCV.

When Gemini API is unavailable or times out, this module produces LINE
entities from a preprocessed (line-only) image using Hough line detection.
Used to still yield usable DXF geometry when the API fails.
"""

from PIL import Image
import cv2
import numpy as np

from .geometry_models import (
    SketchGeometry,
    SketchMetadata,
    LineEntity,
    Point2D,
    LayerType,
)
from .image_processor import ImageProcessor


# Default confidence for locally extracted lines (lower than API)
LOCAL_EXTRACTION_CONFIDENCE = 0.5


def _clamp(value: float, low: int, high: int) -> int:
    """Clamp value to [low, high] and return as int."""
    return max(low, min(high, int(round(value))))


def extract_lines_from_image(
    line_image: Image.Image,
    image_width: int,
    image_height: int,
    min_line_length: int = 40,
    max_line_gap: int = 15,
    hough_threshold: int = 80,
) -> SketchGeometry:
    """
    Extract line segments from a line-only (preprocessed) image using Hough transform.

    Converts pixel coordinates to normalized 0-1000 space to match the
    rest of the pipeline.

    Args:
        line_image: Preprocessed image (e.g. from ImageProcessor.extract_lines_only)
        image_width: Original image width in pixels (for normalization)
        image_height: Original image height in pixels
        min_line_length: Minimum segment length in pixels (HoughLinesP)
        max_line_gap: Max gap between segments to merge (HoughLinesP)
        hough_threshold: Accumulator threshold for line detection

    Returns:
        SketchGeometry with LINE entities only; confidence_score set to
        LOCAL_EXTRACTION_CONFIDENCE; includes a warning that geometry
        is from local detection.
    """
    img_array = np.array(line_image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Optional: invert if lines are white on black (Canny-style output)
    # Our extract_lines_only produces black on white, so we detect dark lines
    # HoughLinesP works on edge images; use Canny for cleaner segments
    edges = cv2.Canny(gray, 50, 150)

    lines_p = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    entities = []
    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            # Normalize to 0-1000 (same convention as Gemini)
            nx1 = _clamp(x1 / image_width * 1000, 0, 1000)
            ny1 = _clamp(y1 / image_height * 1000, 0, 1000)
            nx2 = _clamp(x2 / image_width * 1000, 0, 1000)
            ny2 = _clamp(y2 / image_height * 1000, 0, 1000)
            # Skip degenerate segments
            if (nx1, ny1) == (nx2, ny2):
                continue
            entities.append(
                LineEntity(
                    start=Point2D(x=nx1, y=ny1),
                    end=Point2D(x=nx2, y=ny2),
                    layer=LayerType.GEOMETRY,
                    confidence=LOCAL_EXTRACTION_CONFIDENCE,
                )
            )

    return SketchGeometry(
        metadata=SketchMetadata(
            title=None,
            detected_units="mm",
            sketch_type="2d_orthographic",
            image_width=image_width,
            image_height=image_height,
            confidence_score=LOCAL_EXTRACTION_CONFIDENCE,
        ),
        entities=entities,
        warnings=["Geometry from local line detection (Gemini unavailable)."],
    )


def extract_lines_from_original(
    original_image: Image.Image,
) -> SketchGeometry:
    """
    Convenience: build line-only image from original and run local extraction.

    Args:
        original_image: Original sketch/floor plan image (RGB PIL Image)

    Returns:
        SketchGeometry with LINE entities from local detection.
    """
    line_image = ImageProcessor.extract_lines_only(original_image)
    width, height = ImageProcessor.get_dimensions(original_image)
    return extract_lines_from_image(line_image, width, height)
