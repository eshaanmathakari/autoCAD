"""
OCR result post-processing.

Unicode normalization, learned corrections, and dimension filtering.
"""

from typing import Optional

from sketch_to_cad.ocr.detection import TextRegion
from sketch_to_cad.ocr.imperial_parser import parse as parse_imperial

# Unicode quote/prime variants that OCR engines commonly produce
_UNICODE_MAP = {
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote
    "\u02BC": "'",   # modifier letter apostrophe
    "\u02B9": "'",   # modifier letter prime
    "\u0060": "'",   # grave accent
    "\u00B4": "'",   # acute accent
    "\u201C": '"',   # left double quote
    "\u201D": '"',   # right double quote
    "\u2033": '"',   # double prime
}


def clean_ocr_text(text: str) -> str:
    """Normalize unicode quotes/primes to ASCII equivalents."""
    result = text
    for src, dst in _UNICODE_MAP.items():
        result = result.replace(src, dst)
    return result


def apply_corrections(text: str, correction_dict: dict[str, str]) -> str:
    """Apply learned corrections from feedback system."""
    return correction_dict.get(text.strip(), text)


def reparse_with_cleaning(
    region: TextRegion,
    correction_dict: Optional[dict[str, str]] = None,
) -> TextRegion:
    """Clean text, apply corrections, and re-parse."""
    cleaned = clean_ocr_text(region.text)
    if correction_dict:
        cleaned = apply_corrections(cleaned, correction_dict)
    parsed = parse_imperial(cleaned)
    return TextRegion(
        text=cleaned,
        center_x=region.center_x,
        center_y=region.center_y,
        confidence=region.confidence,
        engine=region.engine,
        parsed=parsed,
    )


def filter_dimensions(regions: list[TextRegion]) -> list[TextRegion]:
    """Keep only regions that parse as imperial dimensions."""
    return [r for r in regions if r.parsed is not None]
