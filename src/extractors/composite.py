"""
Composite extractor: try primary, on failure or empty use local fallback.
"""

from typing import Optional

from PIL import Image

from ..geometry_models import SketchGeometry
from .base import GeometryExtractor


class CompositeExtractor(GeometryExtractor):
    """
    Tries the primary extractor first. On exception or when the result
    has no entities, runs the local fallback and returns its result
    (or merges if useful).
    """

    def __init__(
        self,
        primary: GeometryExtractor,
        local: GeometryExtractor,
        use_local_on_empty: bool = True,
    ):
        """
        Args:
            primary: Main extractor (e.g. Gemini two-pass).
            local: Fallback extractor (e.g. OpenCV line detection).
            use_local_on_empty: If True, run local when primary returns no entities.
        """
        self.primary = primary
        self.local = local
        self.use_local_on_empty = use_local_on_empty

    def extract(
        self,
        original_image: Image.Image,
        preprocessed_image: Optional[Image.Image] = None,
    ) -> SketchGeometry:
        """
        Run primary extractor; on exception or empty entities, run local fallback.
        """
        try:
            geometry = self.primary.extract(original_image, preprocessed_image)
            if self.use_local_on_empty and not geometry.entities:
                fallback = self.local.extract(original_image, preprocessed_image)
                if fallback.entities:
                    return fallback
            return geometry
        except Exception:
            return self.local.extract(original_image, preprocessed_image)
