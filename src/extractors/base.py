"""
Base protocol for geometry extraction.

All extractors take original (and optional preprocessed) image
and return SketchGeometry in normalized 0-1000 coordinates.
"""

from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image

# Import at runtime to avoid circular dependency on geometry_models
from ..geometry_models import SketchGeometry


class GeometryExtractor(ABC):
    """
    Abstract base for geometry extraction from sketch images.

    Implementations may use cloud APIs (Gemini) or local vision (OpenCV).
    """

    @abstractmethod
    def extract(
        self,
        original_image: Image.Image,
        preprocessed_image: Optional[Image.Image] = None,
    ) -> SketchGeometry:
        """
        Extract geometry from the image(s).

        Args:
            original_image: Original sketch/floor plan image (RGB).
            preprocessed_image: Optional line-only image (e.g. from ImageProcessor.extract_lines_only).
                               If None, implementation may derive it or use original.

        Returns:
            SketchGeometry with entities in normalized 0-1000 coordinates.
        """
        ...
