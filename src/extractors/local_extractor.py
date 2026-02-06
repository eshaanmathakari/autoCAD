"""
Local line-only extractor using OpenCV Hough detection.
"""

from typing import Optional

from PIL import Image

from ..geometry_models import SketchGeometry
from ..image_processor import ImageProcessor
from ..local_line_extractor import extract_lines_from_image, extract_lines_from_original
from .base import GeometryExtractor


class LocalLineExtractor(GeometryExtractor):
    """
    Extracts LINE entities only from a preprocessed (line-only) image
    using OpenCV Hough line detection. No API calls.
    """

    def extract(
        self,
        original_image: Image.Image,
        preprocessed_image: Optional[Image.Image] = None,
    ) -> SketchGeometry:
        """
        Build line-only image if needed and run local line extraction.
        Returns SketchGeometry with LINE entities only.
        """
        if preprocessed_image is not None:
            width, height = ImageProcessor.get_dimensions(original_image)
            return extract_lines_from_image(
                preprocessed_image,
                width,
                height,
            )
        return extract_lines_from_original(original_image)
