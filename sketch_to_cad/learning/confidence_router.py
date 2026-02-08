"""
Confidence-based routing for OCR and matching results.

Routes each result to one of three tracks:
  - auto_accept  (confidence > high threshold)  — log silently
  - review       (between thresholds)           — show with "Confirm?" prompt
  - manual       (below low threshold)          — require manual entry

Provides colour helpers for the Streamlit UI traffic-light indicators.
"""

from __future__ import annotations


class ConfidenceRouter:
    """Threshold-based confidence router with UI colour helpers."""

    def __init__(
        self,
        auto_threshold: float = 0.95,
        review_threshold: float = 0.70,
    ):
        self.auto_threshold = auto_threshold
        self.review_threshold = review_threshold

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def route(self, confidence: float) -> str:
        """Return the track for a given confidence score.

        Returns one of ``"auto_accept"``, ``"review"``, or ``"manual"``.
        """
        if confidence >= self.auto_threshold:
            return "auto_accept"
        if confidence >= self.review_threshold:
            return "review"
        return "manual"

    def route_dimension(self, dim) -> str:
        """Convenience wrapper accepting a DimensionResult-like object.

        The object must have a ``.confidence`` attribute.
        """
        return self.route(getattr(dim, "confidence", 0.0))

    # ------------------------------------------------------------------
    # UI helpers (dark-theme friendly colours)
    # ------------------------------------------------------------------
    def get_indicator_color(self, confidence: float) -> str:
        """Return a hex colour suitable for traffic-light display.

        Green  (#4caf50) — high confidence (auto-accept)
        Yellow (#ff9800) — medium confidence (review)
        Red    (#f44336) — low confidence (manual)
        """
        track = self.route(confidence)
        return {
            "auto_accept": "#4caf50",
            "review": "#ff9800",
            "manual": "#f44336",
        }[track]

    def get_indicator_emoji(self, confidence: float) -> str:
        """Return a status emoji for inline display."""
        track = self.route(confidence)
        return {
            "auto_accept": "\u2705",   # green check
            "review": "\U0001F7E1",     # yellow circle
            "manual": "\U0001F534",     # red circle
        }[track]

    def get_label(self, confidence: float) -> str:
        """Human-readable label for the confidence track."""
        track = self.route(confidence)
        return {
            "auto_accept": "High Confidence",
            "review": "Needs Review",
            "manual": "Manual Entry Required",
        }[track]

    def __repr__(self) -> str:
        return (
            f"ConfidenceRouter(auto={self.auto_threshold}, "
            f"review={self.review_threshold})"
        )
