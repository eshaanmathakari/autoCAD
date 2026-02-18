"""
Swimming Pool Sketch-to-CAD — 4-Step Wizard UI

Sequential wizard:
  Step 1 — Upload
  Step 2 — Reference Matching
  Step 3 — Dimension Verification (generates DXF on approval)
  Step 4 — Download & Feedback

Imports pipeline logic from sketch_to_cad.* and orchestrates
the recursive learning system from sketch_to_cad.learning.*.
"""

import os
import sys

# Avoid OpenMP duplicate-library crash on macOS when using CLIP/PyTorch
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load Streamlit Cloud secrets into env if available (no-op when no secrets file)
try:
    if hasattr(st, "secrets"):
        for key in ("GOOGLE_API_KEY", "GEMINI_MODEL"):
            if key in st.secrets:
                os.environ.setdefault(key, st.secrets[key])
except StreamlitSecretNotFoundError:
    pass

# ---------------------------------------------------------------------------
# Learning system (always available — no Part 1 dependency)
# ---------------------------------------------------------------------------
from sketch_to_cad.learning.feedback_store import FeedbackStore
from sketch_to_cad.learning.correction_dict import CorrectionDict
from sketch_to_cad.learning.confidence_router import ConfidenceRouter

# Step renderers
from sketch_to_cad.steps.upload import render_upload
from sketch_to_cad.steps.matching import render_matching
from sketch_to_cad.steps.verification import render_verification
from sketch_to_cad.steps.download import render_download

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pool Sketch-to-CAD",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached singletons
# ---------------------------------------------------------------------------
@st.cache_resource
def get_feedback_store() -> FeedbackStore:
    db_path = os.path.join(os.path.dirname(__file__), "data", "feedback.db")
    store = FeedbackStore(db_path=db_path)
    store.seed_demo_data()
    return store


@st.cache_resource
def get_correction_dict(_store: FeedbackStore) -> CorrectionDict:
    return CorrectionDict(_store)


@st.cache_resource
def get_confidence_router() -> ConfidenceRouter:
    return ConfidenceRouter(auto_threshold=0.95, review_threshold=0.70)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
DEFAULTS = {
    "current_step": 1,
    "session_id": None,
    "input_image": None,
    "drawing_fingerprint": None,
    "match_results": [],
    "selected_ref": None,
    "ocr_results": [],
    "approved_dimensions": None,
    "deformation": None,
    "verification": None,
    "sandbox_result": None,
    "final_dxf": None,
    "feedback_submitted": False,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# Auto-create a session ID on first load
if st.session_state.session_id is None:
    fb = get_feedback_store()
    st.session_state.session_id = fb.create_session()


# ---------------------------------------------------------------------------
# Progress bar across the top
# ---------------------------------------------------------------------------
STEP_LABELS = [
    "1. Upload",
    "2. Match",
    "3. Dimensions",
    "4. Download",
]


def render_progress_bar():
    """Render the 4-step progress indicator across the top."""
    current = st.session_state.current_step
    cols = st.columns(len(STEP_LABELS))
    for i, (col, label) in enumerate(zip(cols, STEP_LABELS)):
        step_num = i + 1
        with col:
            if step_num < current:
                st.markdown(
                    f"<div style='text-align:center; padding:8px; "
                    f"background-color:#4a4a4a; border-radius:8px; "
                    f"color:#e0e0e0; font-weight:bold;'>"
                    f"{label}</div>",
                    unsafe_allow_html=True,
                )
            elif step_num == current:
                st.markdown(
                    f"<div style='text-align:center; padding:8px; "
                    f"background-color:#2d2d2d; border-radius:8px; "
                    f"color:#ffffff; font-weight:bold; "
                    f"border:2px solid #757575;'>"
                    f"{label}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='text-align:center; padding:8px; "
                    f"background-color:#e8e8e8; border-radius:8px; "
                    f"color:#737373;'>"
                    f"{label}</div>",
                    unsafe_allow_html=True,
                )
    st.markdown("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.title("Pool Sketch-to-CAD")
    st.caption("Upload | Match | Dimensions | Download")

    render_progress_bar()

    fb = get_feedback_store()
    cd = get_correction_dict(fb)
    cr = get_confidence_router()

    step = st.session_state.current_step

    if step == 1:
        render_upload()
    elif step == 2:
        render_matching(feedback_store=fb)
    elif step == 3:
        render_verification(
            feedback_store=fb,
            correction_dict=cd,
            confidence_router=cr,
        )
    elif step == 4:
        render_download(feedback_store=fb)
    else:
        st.error(f"Unknown step: {step}")


if __name__ == "__main__":
    main()
