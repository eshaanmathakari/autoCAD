"""
Adaptive OCR Correction Dictionary.

Builds a raw_text -> corrected_text mapping from accumulated user corrections
in the feedback store. Fed into OCREnsemble(correction_dict=...) to auto-apply
learned patterns before showing results to users.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sketch_to_cad.learning.feedback_store import FeedbackStore


class CorrectionDict:
    """Builds and caches an OCR correction dictionary from the feedback store."""

    def __init__(self, feedback_store: "FeedbackStore", min_count: int = 2):
        self._store = feedback_store
        self._min_count = min_count
        self._dict: dict[str, str] = {}
        self.refresh()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def as_dict(self) -> dict[str, str]:
        """Return the correction dictionary for OCREnsemble consumption.

        Returns:
            Mapping of {raw_ocr_text: corrected_text} for patterns seen
            at least ``min_count`` times.
        """
        return dict(self._dict)

    def refresh(self):
        """Rebuild the dictionary from the latest feedback store data."""
        self._dict = self._store.get_correction_pairs(min_count=self._min_count)

    def apply(self, text: str) -> str:
        """Apply known corrections to a single OCR text string.

        Returns the corrected text if a match is found, otherwise the
        original text unchanged.
        """
        return self._dict.get(text, text)

    def __len__(self) -> int:
        return len(self._dict)

    def __contains__(self, key: str) -> bool:
        return key in self._dict

    def __repr__(self) -> str:
        return f"CorrectionDict(patterns={len(self._dict)}, min_count={self._min_count})"
