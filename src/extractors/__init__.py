"""
Geometry extractor abstraction for sketch-to-DXF.

Provides a common interface and implementations:
- GeometryExtractor: protocol for extract(original_image, preprocessed_image?) -> SketchGeometry
- PrimaryGeminiExtractor: Gemini two-pass (walls + text)
- LocalLineExtractor: OpenCV Hough-based line detection
- CompositeExtractor: try primary; on failure or empty, run local fallback
"""

from .base import GeometryExtractor
from .primary import PrimaryGeminiExtractor
from .local_extractor import LocalLineExtractor
from .composite import CompositeExtractor

__all__ = [
    "GeometryExtractor",
    "PrimaryGeminiExtractor",
    "LocalLineExtractor",
    "CompositeExtractor",
]
