"""
OCR Utilities for CAD Sketch Extraction.

Provides text correction for common OCR errors in dimension annotations
and fuzzy matching for room labels.
"""

import re
from difflib import SequenceMatcher
from typing import Optional


# Standard room names for fuzzy matching
STANDARD_ROOM_NAMES = [
    "BEDROOM", "MASTER BEDROOM", "GUEST BEDROOM",
    "BATHROOM", "MASTER BATHROOM", "TOILET", "WC",
    "KITCHEN", "KITCHENETTE",
    "LIVING ROOM", "LIVING", "LOUNGE",
    "DINING ROOM", "DINING",
    "OFFICE", "STUDY", "LIBRARY",
    "GARAGE", "CARPORT",
    "HALL", "HALLWAY", "CORRIDOR", "PASSAGE",
    "FOYER", "ENTRY", "ENTRANCE",
    "BALCONY", "TERRACE", "PATIO", "DECK",
    "STORE", "STORAGE", "UTILITY", "LAUNDRY",
    "CLOSET", "WARDROBE", "PANTRY",
    "PORCH", "VERANDA",
    "BASEMENT", "ATTIC", "LOFT",
]

# Standard dimension values for snapping (in mm)
STANDARD_DIMENSIONS = [
    50, 100, 150, 200, 250, 300, 400, 500, 600, 750,
    800, 900, 1000, 1200, 1500, 1800, 2000, 2400, 3000,
    3600, 4000, 4500, 5000, 6000,
]

# Standard wall thicknesses (in mm)
STANDARD_WALL_THICKNESSES = [100, 115, 150, 200, 230, 300]

# Standard door widths (in mm)
STANDARD_DOOR_WIDTHS = [700, 750, 800, 900, 1000, 1200]

# Standard window widths (in mm)
STANDARD_WINDOW_WIDTHS = [600, 900, 1000, 1200, 1500, 1800]


class OCRCorrector:
    """
    Corrects common OCR errors in dimension text and room labels.

    Designed for handwritten text on technical drawings where letters
    are often misread as numbers and vice versa.
    """

    # Character substitution map for dimension text
    # Maps commonly confused characters to their numeric equivalents
    CHAR_FIXES = {
        'O': '0', 'o': '0',      # Letter O to zero
        'l': '1', 'I': '1',       # Letter l/I to one
        '|': '1',                 # Pipe to one
        'S': '5', 's': '5',       # Letter S to five
        'B': '8',                 # Letter B to eight (lowercase b is often valid)
        'G': '6',                 # Letter G to six
        'g': '9',                 # Letter g to nine
        'Z': '2', 'z': '2',       # Letter Z to two
        'T': '7',                 # Letter T to seven (in some fonts)
        'D': '0',                 # Letter D to zero (in handwriting)
    }

    # Valid units for dimensions
    VALID_UNITS = ['mm', 'cm', 'm', 'in', 'ft', "'", '"', '°', 'deg']

    # Dimension text pattern: number + optional unit
    DIMENSION_PATTERN = re.compile(
        r'^[\d.,\s]+\s*(mm|cm|m|in|ft|\'|"|°|deg)?$',
        re.IGNORECASE
    )

    @classmethod
    def fix_dimension_text(cls, text: str, confidence: float = 1.0) -> str:
        """
        Apply OCR fixes to dimension text (conservative approach).

        Only applies corrections when:
        - Text does NOT already parse as a valid number
        - Or confidence is low (< 0.7)
        - Character substitution is context-aware (only fix if surrounded by digits)

        Args:
            text: Raw OCR text (e.g., "1O5mm", "S00")
            confidence: OCR confidence (0-1). Only apply aggressive fixes when low.

        Returns:
            Corrected text (e.g., "105mm", "500") or original if already valid
        """
        if not text:
            return text

        text = text.strip()

        # Separate the unit suffix (if present) to protect it
        unit_match = re.search(r'(mm|cm|m|in|ft|\'|"|°|deg)\s*$', text, re.IGNORECASE)
        unit = ''
        numeric_part = text
        if unit_match:
            unit = unit_match.group(1)
            numeric_part = text[:unit_match.start()].strip()

        # FIRST: Check if text already parses as valid number - don't modify!
        try:
            test_value = numeric_part.replace(',', '.').replace(' ', '')
            float(test_value)
            # Already valid! Return as-is (just normalize unit case)
            if unit:
                return numeric_part + unit.lower()
            return numeric_part
        except ValueError:
            pass  # Needs correction

        # Only apply aggressive corrections if confidence is low
        if confidence >= 0.7:
            # High confidence: only do context-aware corrections
            result = cls._context_aware_fix(numeric_part)
        else:
            # Low confidence: apply all substitutions
            result = []
            for char in numeric_part:
                if char in cls.CHAR_FIXES:
                    result.append(cls.CHAR_FIXES[char])
                elif char.isdigit() or char in '.,- ':
                    result.append(char)
            result = ''.join(result).strip()

        # Reattach unit
        if unit:
            result = result + unit.lower()

        return result

    @classmethod
    def _context_aware_fix(cls, text: str) -> str:
        """
        Apply context-aware character fixes.

        Only substitutes a character if it's surrounded by digits.
        e.g., "1O5" -> "105" but "BOX" stays as "BOX"

        Args:
            text: Numeric text to fix

        Returns:
            Corrected text
        """
        if not text:
            return text

        result = list(text)
        for i, char in enumerate(text):
            if char in cls.CHAR_FIXES:
                # Check if surrounded by digits (context-aware)
                prev_is_digit = i > 0 and text[i-1].isdigit()
                next_is_digit = i < len(text) - 1 and text[i+1].isdigit()

                # Only substitute if surrounded by digits OR at start/end with adjacent digit
                if (prev_is_digit and next_is_digit) or \
                   (i == 0 and next_is_digit) or \
                   (i == len(text) - 1 and prev_is_digit):
                    result[i] = cls.CHAR_FIXES[char]
            elif not char.isdigit() and char not in '.,- ':
                # Remove other non-numeric characters
                result[i] = ''

        return ''.join(result).strip()

    @classmethod
    def extract_numeric_value(cls, text: str) -> tuple[Optional[float], Optional[str]]:
        """
        Extract numeric value and unit from dimension text.

        Applies OCR corrections before parsing.

        Args:
            text: Dimension text (e.g., "500mm", "25.5cm", "1O5")

        Returns:
            Tuple of (value, unit) or (None, None) if parsing fails.
            Unit defaults to 'mm' if not specified.
        """
        if not text:
            return None, None

        # Apply OCR corrections first
        corrected = cls.fix_dimension_text(text)

        # Extract unit
        unit_match = re.search(r'(mm|cm|m|in|ft|\'|"|°|deg)\s*$', corrected, re.IGNORECASE)
        unit = 'mm'  # Default unit
        if unit_match:
            unit = unit_match.group(1).lower()
            corrected = corrected[:unit_match.start()].strip()

        # Normalize unit
        if unit == "'":
            unit = 'ft'
        elif unit == '"':
            unit = 'in'
        elif unit == 'deg':
            unit = '°'

        # Parse numeric value
        try:
            # Handle comma as decimal separator
            corrected = corrected.replace(',', '.').replace(' ', '')
            value = float(corrected)
            return value, unit
        except ValueError:
            return None, None

    @classmethod
    def convert_to_mm(cls, value: float, unit: str) -> float:
        """
        Convert a dimension value to millimeters.

        Args:
            value: Numeric value
            unit: Unit string (mm, cm, m, in, ft)

        Returns:
            Value in millimeters
        """
        unit = unit.lower()
        conversion = {
            'mm': 1.0,
            'cm': 10.0,
            'm': 1000.0,
            'in': 25.4,
            'ft': 304.8,
        }
        return value * conversion.get(unit, 1.0)

    @classmethod
    def snap_to_standard(
        cls,
        value: float,
        standards: list[float] = None,
        threshold: float = 0.05
    ) -> tuple[float, bool]:
        """
        Snap a dimension value to the nearest standard if within threshold.

        Args:
            value: Dimension value in mm
            standards: List of standard values to snap to (defaults to STANDARD_DIMENSIONS)
            threshold: Maximum relative difference to snap (default 5%)

        Returns:
            Tuple of (snapped_value, was_snapped)
        """
        if standards is None:
            standards = STANDARD_DIMENSIONS

        if value <= 0:
            return value, False

        closest = min(standards, key=lambda s: abs(s - value))
        relative_diff = abs(closest - value) / value

        if relative_diff <= threshold:
            return closest, True

        return value, False

    @classmethod
    def fuzzy_match_room_name(
        cls,
        text: str,
        known_rooms: list[str] = None,
        min_ratio: float = 0.7
    ) -> Optional[str]:
        """
        Match OCR'd room name to known room types using fuzzy matching.

        Args:
            text: OCR'd text (e.g., "BEDR00M", "K1TCHEN")
            known_rooms: List of valid room names (defaults to STANDARD_ROOM_NAMES)
            min_ratio: Minimum similarity ratio (0-1) for a match

        Returns:
            Best matching room name, or None if no match found
        """
        if not text:
            return None

        if known_rooms is None:
            known_rooms = STANDARD_ROOM_NAMES

        # Apply character fixes (numbers to letters for room names)
        text_fixed = text.upper()
        reverse_fixes = {
            '0': 'O',
            '1': 'I',
            '5': 'S',
            '8': 'B',
            '6': 'G',
            '9': 'G',
            '2': 'Z',
        }
        for num, letter in reverse_fixes.items():
            text_fixed = text_fixed.replace(num, letter)

        best_match = None
        best_ratio = 0.0

        for room in known_rooms:
            # Try both original and fixed text
            ratio1 = SequenceMatcher(None, text.upper(), room).ratio()
            ratio2 = SequenceMatcher(None, text_fixed, room).ratio()
            ratio = max(ratio1, ratio2)

            if ratio > best_ratio and ratio >= min_ratio:
                best_ratio = ratio
                best_match = room

        return best_match

    @classmethod
    def is_dimension_text(cls, text: str) -> bool:
        """
        Check if text appears to be a dimension annotation.

        Args:
            text: Text to check

        Returns:
            True if text looks like a dimension (numbers with optional unit)
        """
        if not text:
            return False

        corrected = cls.fix_dimension_text(text)
        value, unit = cls.extract_numeric_value(corrected)

        return value is not None

    @classmethod
    def is_room_label(cls, text: str) -> bool:
        """
        Check if text appears to be a room label.

        Args:
            text: Text to check

        Returns:
            True if text is primarily alphabetic and matches room pattern
        """
        if not text:
            return False

        # Remove spaces and check if mostly alphabetic
        clean = text.replace(' ', '')
        if len(clean) < 2:
            return False

        alpha_count = sum(1 for c in clean if c.isalpha())
        ratio = alpha_count / len(clean)

        # At least 60% alphabetic characters
        if ratio < 0.6:
            return False

        # Check against known room names
        matched = cls.fuzzy_match_room_name(text)

        return matched is not None or ratio > 0.8


def correct_dimension_entity(entity: dict) -> dict:
    """
    Apply OCR corrections to a dimension entity.

    Args:
        entity: Dictionary with 'value' and optionally 'unit' keys

    Returns:
        Corrected entity dictionary
    """
    if 'value' not in entity:
        return entity

    value_text = str(entity.get('value', ''))
    corrected = OCRCorrector.fix_dimension_text(value_text)
    numeric_value, unit = OCRCorrector.extract_numeric_value(corrected)

    if numeric_value is not None:
        entity['value'] = corrected
        entity['numeric_value'] = numeric_value
        entity['unit'] = unit or entity.get('unit', 'mm')

        # Try snapping to standard
        value_mm = OCRCorrector.convert_to_mm(numeric_value, entity['unit'])
        snapped, was_snapped = OCRCorrector.snap_to_standard(value_mm)
        if was_snapped:
            entity['snapped_value'] = snapped
            entity['was_snapped'] = True

    return entity


def correct_text_entity(entity: dict) -> dict:
    """
    Apply corrections to a text entity (room label).

    Args:
        entity: Dictionary with 'content' key

    Returns:
        Corrected entity dictionary
    """
    if 'content' not in entity:
        return entity

    content = entity.get('content', '')
    matched_room = OCRCorrector.fuzzy_match_room_name(content)

    if matched_room:
        entity['original_content'] = content
        entity['content'] = matched_room
        entity['was_corrected'] = True

    return entity
