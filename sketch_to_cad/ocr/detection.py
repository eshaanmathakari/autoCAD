"""
Text region detection using PaddleOCR.

Wraps PaddleOCR's detection + recognition into a simple interface
that returns TextRegion objects with pixel coordinates.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from sketch_to_cad.ocr.imperial_parser import parse as parse_imperial, ImperialDimension

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """Single OCR detection from one engine."""
    text: str
    center_x: int
    center_y: int
    confidence: float
    engine: str
    parsed: Optional[ImperialDimension] = None


class PaddleDetector:
    """Wrapper around PaddleOCR for text detection + recognition."""

    def __init__(self):
        self._ocr = None
        self._init()

    def _init(self):
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False,
            )
            logger.info("PaddleOCR initialized")
        except Exception as e:
            logger.info(f"PaddleOCR not available: {e}")

    @property
    def available(self) -> bool:
        return self._ocr is not None

    def detect(self, image: Image.Image) -> list[TextRegion]:
        """Run PaddleOCR on an image, return detected text regions."""
        if self._ocr is None:
            return []
        try:
            img_array = np.array(image.convert("RGB"))
            results = self._ocr.ocr(img_array, cls=True)
            detections = []
            if results and results[0]:
                for line in results[0]:
                    bbox, (text, conf) = line[0], line[1]
                    cx = int(sum(p[0] for p in bbox) / 4)
                    cy = int(sum(p[1] for p in bbox) / 4)
                    parsed = parse_imperial(text)
                    detections.append(TextRegion(
                        text=text,
                        center_x=cx,
                        center_y=cy,
                        confidence=float(conf),
                        engine="paddle",
                        parsed=parsed,
                    ))
            return detections
        except Exception as e:
            logger.warning(f"PaddleOCR failed: {e}")
            return []
