"""
Learning subsystem — recursive feedback loop for the Sketch-to-CAD pipeline.

Re-exports the key classes so consumers can write:
    from sketch_to_cad.learning import FeedbackStore, CorrectionDict, ...
"""

from sketch_to_cad.learning.feedback_store import FeedbackStore
from sketch_to_cad.learning.correction_dict import CorrectionDict
from sketch_to_cad.learning.confidence_router import ConfidenceRouter
from sketch_to_cad.learning.metrics import render_sidebar_metrics, get_metrics_summary

__all__ = [
    "FeedbackStore",
    "CorrectionDict",
    "ConfidenceRouter",
    "render_sidebar_metrics",
    "get_metrics_summary",
]
