"""
Primary geometry extractor using Gemini two-pass (walls + text).
"""

from typing import Optional

from PIL import Image

from ..geometry_models import SketchGeometry
from ..gemini_extractor import GeminiGeometryExtractor
from .base import GeometryExtractor


class PrimaryGeminiExtractor(GeometryExtractor):
    """
    Extracts geometry using Gemini Vision API with two-pass extraction
    (walls from preprocessed image, text from original).
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize with optional API key and model override.

        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env).
            model: Gemini model ID (defaults to GEMINI_MODEL env).
        """
        self._gemini = GeminiGeometryExtractor(api_key=api_key, model=model)

    def extract(
        self,
        original_image: Image.Image,
        preprocessed_image: Optional[Image.Image] = None,
    ) -> SketchGeometry:
        """Run Gemini two-pass extraction (walls then text)."""
        return self._gemini.extract_two_pass(original_image, preprocessed_image)
