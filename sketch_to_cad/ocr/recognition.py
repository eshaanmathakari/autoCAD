"""
Gemini Vision dimension extraction.

Sends pool sketch images to Gemini with a targeted prompt
to extract dimension annotations as structured JSON.
"""

import io
import json
import logging
import os
from typing import Optional

from PIL import Image

from sketch_to_cad.ocr.detection import TextRegion
from sketch_to_cad.ocr.imperial_parser import parse as parse_imperial

logger = logging.getLogger(__name__)


class GeminiRecognizer:
    """Gemini Vision targeted dimension extraction."""

    def __init__(self, api_key: Optional[str] = None):
        self._key = api_key or os.environ.get("GOOGLE_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self._key)

    def recognize(self, image: Image.Image) -> list[TextRegion]:
        """Extract dimensions from image using Gemini Vision."""
        if not self._key:
            return []
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self._key)

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            w, h = image.size

            prompt = f"""Analyze this swimming pool sketch image ({w}x{h} pixels).
Extract ALL visible dimension annotations / measurements.
Pool dimensions use feet and inches notation like: 2', 2'4", 12'6", 4", etc.

Return a JSON array where each element is:
{{"text": "<dimension text as written>", "x": <center x pixel>, "y": <center y pixel>}}

Only return dimension/measurement text. Ignore labels like "POOL" or "STAIRS".
Return ONLY the JSON array, no other text."""

            response = client.models.generate_content(
                model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
                contents=[
                    types.Content(parts=[
                        types.Part(text=prompt),
                        types.Part(inline_data=types.Blob(
                            mime_type="image/png", data=image_bytes
                        )),
                    ])
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )

            raw = response.text.strip()
            items = json.loads(raw)
            detections = []
            for item in items:
                text = str(item.get("text", ""))
                x = int(item.get("x", 0))
                y = int(item.get("y", 0))
                parsed = parse_imperial(text)
                detections.append(TextRegion(
                    text=text,
                    center_x=x,
                    center_y=y,
                    confidence=0.85,
                    engine="gemini",
                    parsed=parsed,
                ))
            return detections
        except Exception as e:
            logger.warning(f"Gemini OCR failed: {e}")
            return []
