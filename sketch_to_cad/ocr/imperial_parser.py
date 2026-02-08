"""
Imperial dimension parser for swimming pool measurements.

Handles all common feet/inches notations found on hand-drawn pool sketches:
  2'        -> 2 feet 0 inches = 24 inches
  2'4"      -> 2 feet 4 inches = 28 inches
  2' 4"     -> 2 feet 4 inches = 28 inches
  2'-4"     -> 2 feet 4 inches = 28 inches
  12'6"     -> 12 feet 6 inches = 150 inches
  4"        -> 0 feet 4 inches = 4 inches
  2.5'      -> 2 feet 6 inches = 30 inches
  3ft 6in   -> 3 feet 6 inches = 42 inches
  3 ft 6 in -> 3 feet 6 inches = 42 inches
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ImperialDimension:
    """Parsed imperial dimension."""
    feet: int
    inches: float
    total_inches: float
    total_mm: float
    raw_text: str
    confidence: float = 1.0

    @property
    def display(self) -> str:
        """Human-readable display string."""
        if self.feet == 0:
            return f'{self.inches}"'
        if self.inches == 0:
            return f"{self.feet}'"
        return f"{self.feet}'{self.inches:.0f}\""


# Ordered by specificity — most specific patterns first
_PATTERNS = [
    # 2'4", 2' 4", 2'-4", 12'6", 2' - 4"
    (r"(\d+)\s*['''ʼ\u2019]\s*[-–]?\s*(\d+(?:\.\d+)?)\s*[\"″\"\u2033]", "feet_inches"),
    # 3ft 6in, 3 ft 6 in
    (r"(\d+)\s*ft\.?\s*(\d+(?:\.\d+)?)\s*in\.?", "ft_in_word"),
    # 3ft (no inches)
    (r"(\d+)\s*ft\.?(?!\s*\d)", "ft_only_word"),
    # 2.5' (decimal feet)
    (r"(\d+\.\d+)\s*['''ʼ\u2019]", "decimal_feet"),
    # 2' (whole feet, no inches following)
    (r"(\d+)\s*['''ʼ\u2019](?!\s*[-–]?\s*\d)", "feet_only"),
    # 4", 4.5"
    (r"(\d+(?:\.\d+)?)\s*[\"″\"\u2033]", "inches_only"),
    # 6in, 6 in
    (r"(\d+(?:\.\d+)?)\s*in\.?", "in_word"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), name) for p, name in _PATTERNS]


def parse(text: str) -> Optional[ImperialDimension]:
    """
    Parse a text string into an ImperialDimension.

    Returns None if the text does not match any known imperial notation.
    """
    cleaned = text.strip()
    if not cleaned:
        return None

    for pattern, kind in _COMPILED:
        m = pattern.search(cleaned)
        if not m:
            continue

        feet = 0
        inches = 0.0

        if kind == "feet_inches":
            feet = int(m.group(1))
            inches = float(m.group(2))
        elif kind == "ft_in_word":
            feet = int(m.group(1))
            inches = float(m.group(2))
        elif kind == "ft_only_word":
            feet = int(m.group(1))
        elif kind == "decimal_feet":
            dec = float(m.group(1))
            feet = int(dec)
            inches = (dec - feet) * 12
        elif kind == "feet_only":
            feet = int(m.group(1))
        elif kind == "inches_only":
            inches = float(m.group(1))
        elif kind == "in_word":
            inches = float(m.group(1))

        total_inches = feet * 12 + inches
        total_mm = total_inches * 25.4

        return ImperialDimension(
            feet=feet,
            inches=round(inches, 2),
            total_inches=round(total_inches, 2),
            total_mm=round(total_mm, 2),
            raw_text=cleaned,
        )

    return None


def parse_strict(text: str) -> ImperialDimension:
    """
    Parse text or raise ValueError if it doesn't match.
    """
    result = parse(text)
    if result is None:
        raise ValueError(f"Cannot parse imperial dimension: {text!r}")
    return result


def is_imperial(text: str) -> bool:
    """Check if text looks like an imperial dimension."""
    return parse(text) is not None


# ---------------------------------------------------------------------------
# Built-in test suite
# ---------------------------------------------------------------------------
def _self_test():
    """Run built-in tests — call from CLI or import."""
    cases = [
        # (input, expected_feet, expected_inches, expected_total_inches)
        ("2'", 2, 0, 24),
        ("2'4\"", 2, 4, 28),
        ("2' 4\"", 2, 4, 28),
        ("2'-4\"", 2, 4, 28),
        ("12'6\"", 12, 6, 150),
        ("4\"", 0, 4, 4),
        ("2.5'", 2, 6, 30),
        ("3ft 6in", 3, 6, 42),
        ("3 ft 6 in", 3, 6, 42),
        ("3ft", 3, 0, 36),
        ("10'0\"", 10, 0, 120),
        ("0'8\"", 0, 8, 8),
        ("15'", 15, 0, 180),
        ("6\"", 0, 6, 6),
        ("1'11\"", 1, 11, 23),
        ("8'3\"", 8, 3, 99),
        ("2\u20194\u2033", 2, 4, 28),  # unicode right-quote / double-prime
        ("5.25'", 5, 3, 63),
        ("6 in", 0, 6, 6),
        # Edge: text with surrounding noise
        ("depth 3'6\" approx", 3, 6, 42),
    ]

    passed = 0
    failed = 0
    for text, exp_ft, exp_in, exp_total in cases:
        result = parse(text)
        ok = (
            result is not None
            and result.feet == exp_ft
            and result.inches == exp_in
            and result.total_inches == exp_total
        )
        if ok:
            passed += 1
        else:
            failed += 1
            got = f"feet={result.feet}, inches={result.inches}, total={result.total_inches}" if result else "None"
            print(f"FAIL: {text!r}  expected ({exp_ft}, {exp_in}, {exp_total})  got {got}")

    # Negative cases — should NOT parse
    negatives = ["hello", "500mm", "2.5m", "", "abc123"]
    for text in negatives:
        result = parse(text)
        if result is None:
            passed += 1
        else:
            failed += 1
            print(f"FAIL (should be None): {text!r}  got {result}")

    total = passed + failed
    print(f"\nImperial parser tests: {passed}/{total} passed")
    if failed:
        print(f"  {failed} FAILED")
    return failed == 0


if __name__ == "__main__":
    _self_test()
