"""
Drawing fingerprint extraction.

Analyses a pool sketch image using a vision model (Gemini) and OCR
to produce a structured JSON "fingerprint" containing pool type,
detected dimensions, and structural features.
"""

import io
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

# Known pool types for normalisation
_KNOWN_TYPES = {
    "rectangular", "l_shaped", "l-shaped", "kidney", "freeform",
    "oval", "roman", "figure_8", "lazy_l", "grecian",
}


def _normalise_pool_type(raw: str) -> str:
    """Normalise a raw pool type string to a canonical form."""
    cleaned = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if cleaned in _KNOWN_TYPES:
        return cleaned
    for known in _KNOWN_TYPES:
        if known in cleaned or cleaned in known:
            return known
    return cleaned


def _extract_via_gemini(image: Image.Image) -> Optional[dict]:
    """Call Gemini Vision to identify pool type and structural features."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        w, h = image.size

        prompt = f"""Analyse this swimming pool sketch image ({w}x{h} pixels).

Identify the following structural features and return them as a JSON object:

{{
  "pool_type": "<one of: rectangular, l_shaped, kidney, freeform, oval, roman, figure_8, lazy_l, grecian, or other>",
  "has_stairs": <true or false>,
  "num_edges": <approximate number of straight edges in the pool outline>,
  "shape_notes": "<brief description of pool shape>",
  "dimensions": [
    {{"label": "<optional label like length or width>", "text": "<dimension text as written>", "value_inches": <numeric value in inches>}}
  ],
  "length_inches": <overall pool length in inches if identifiable, else null>,
  "width_inches": <overall pool width in inches if identifiable, else null>
}}

For dimensions, pool measurements use feet/inches notation: 20', 12'6", 4", etc.
Convert all measurements to inches (e.g. 20' = 240 inches, 12'6" = 150 inches).
If you cannot identify length/width confidently, set them to null.
Return ONLY the JSON object, no other text."""

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
        data = json.loads(raw)
        return data

    except Exception as e:
        logger.warning("Gemini fingerprint extraction failed: %s", e)
        return None


def _extract_via_ocr(image: Image.Image) -> list[dict]:
    """Use the existing OCR ensemble to extract dimension data."""
    try:
        from sketch_to_cad.ocr.ensemble import OCREnsemble
    except ImportError:
        return []

    try:
        ensemble = OCREnsemble()
        if not ensemble.available_engines:
            return []
        results = ensemble.extract_dimensions(image)
        dims = []
        for r in results:
            parsed = getattr(r, "parsed", None)
            dims.append({
                "text": getattr(r, "text", ""),
                "value_inches": parsed.total_inches if parsed else None,
                "confidence": getattr(r, "confidence", 0.0),
            })
        return dims
    except Exception as e:
        logger.warning("OCR fingerprint extraction failed: %s", e)
        return []


def _infer_length_width(dimensions: list[dict]) -> tuple[Optional[float], Optional[float]]:
    """Infer length and width from a list of dimension dicts.

    Heuristic: the two largest distinct values are length (larger)
    and width (smaller).
    """
    values = sorted(
        {d["value_inches"] for d in dimensions if d.get("value_inches") and d["value_inches"] > 0},
        reverse=True,
    )
    if len(values) >= 2:
        return values[0], values[1]
    if len(values) == 1:
        return values[0], None
    return None, None


def extract_fingerprint(image: Image.Image) -> dict:
    """Extract a structured fingerprint from a pool sketch image.

    Returns a JSON-serialisable dict with pool type, dimensions,
    and structural features.
    """
    w, h = image.size
    engines_used = []

    gemini_data = _extract_via_gemini(image)
    if gemini_data:
        engines_used.append("gemini")

    ocr_dims = _extract_via_ocr(image)
    if ocr_dims:
        engines_used.append("ocr_ensemble")

    # Build fingerprint from Gemini data, falling back to OCR
    pool_type = None
    has_stairs = None
    num_edges = None
    shape_notes = None
    length_inches = None
    width_inches = None
    dimensions = []

    if gemini_data:
        pool_type = _normalise_pool_type(gemini_data.get("pool_type", "") or "")
        has_stairs = gemini_data.get("has_stairs")
        num_edges = gemini_data.get("num_edges")
        shape_notes = gemini_data.get("shape_notes")
        length_inches = gemini_data.get("length_inches")
        width_inches = gemini_data.get("width_inches")
        dimensions = gemini_data.get("dimensions", [])

    # Merge OCR dimensions if Gemini did not produce them
    if not dimensions and ocr_dims:
        dimensions = ocr_dims

    # If length/width not from Gemini, try inferring from dimensions
    if (length_inches is None or width_inches is None) and dimensions:
        inferred_l, inferred_w = _infer_length_width(dimensions)
        if length_inches is None:
            length_inches = inferred_l
        if width_inches is None:
            width_inches = inferred_w

    fingerprint = {
        "pool_type": pool_type,
        "has_stairs": has_stairs,
        "num_edges": num_edges,
        "shape_notes": shape_notes,
        "dimensions": dimensions,
        "length_inches": length_inches,
        "width_inches": width_inches,
        "image_size": [w, h],
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "engines_used": engines_used,
    }

    return fingerprint
