"""
Swimming Pool Sketch-to-CAD — 5-Screen Wizard UI

Replaces the 3-tab PoC with a sequential wizard:
  Screen 1 — Upload
  Screen 2 — Reference Matching
  Screen 3 — Dimension Verification
  Screen 4 — CAD Preview & Verification
  Screen 5 — Download & Feedback

Imports pipeline logic from sketch_to_cad.* (Part 1) and orchestrates
the recursive learning system from sketch_to_cad.learning.*.
"""

import os
import sys
import uuid

import streamlit as st

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load Streamlit Cloud secrets into env if available
if hasattr(st, "secrets"):
    for key in ("GOOGLE_API_KEY", "GEMINI_MODEL"):
        if key in st.secrets:
            os.environ.setdefault(key, st.secrets[key])

# ---------------------------------------------------------------------------
# Learning system (always available — no Part 1 dependency)
# ---------------------------------------------------------------------------
from sketch_to_cad.learning.feedback_store import FeedbackStore
from sketch_to_cad.learning.correction_dict import CorrectionDict
from sketch_to_cad.learning.confidence_router import ConfidenceRouter
from sketch_to_cad.learning.metrics import render_sidebar_metrics

# Step renderers
from sketch_to_cad.steps.upload import render_upload
from sketch_to_cad.steps.matching import render_matching
from sketch_to_cad.steps.verification import render_verification
from sketch_to_cad.steps.cad_preview import render_cad_preview
from sketch_to_cad.steps.download import render_download

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pool Sketch-to-CAD",
    layout="wide",
    page_icon="\U0001F3CA",
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
    "4. CAD Preview",
    "5. Download",
]


def render_progress_bar():
    """Render the 5-step progress indicator across the top."""
    current = st.session_state.current_step
    cols = st.columns(len(STEP_LABELS))
    for i, (col, label) in enumerate(zip(cols, STEP_LABELS)):
        step_num = i + 1
        with col:
            if step_num < current:
                # Completed
                st.markdown(
                    f"<div style='text-align:center; padding:8px; "
                    f"background-color:#1b5e20; border-radius:8px; "
                    f"color:#a5d6a7; font-weight:bold;'>"
                    f"\u2705 {label}</div>",
                    unsafe_allow_html=True,
                )
            elif step_num == current:
                # Active
                st.markdown(
                    f"<div style='text-align:center; padding:8px; "
                    f"background-color:#00838f; border-radius:8px; "
                    f"color:#ffffff; font-weight:bold; "
                    f"border:2px solid #00bcd4;'>"
                    f"\u25B6 {label}</div>",
                    unsafe_allow_html=True,
                )
            else:
                # Pending
                st.markdown(
                    f"<div style='text-align:center; padding:8px; "
                    f"background-color:#263238; border-radius:8px; "
                    f"color:#78909c;'>"
                    f"\u26AA {label}</div>",
                    unsafe_allow_html=True,
                )
    st.markdown("")  # spacer


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    """Render the sidebar with pipeline status and learning metrics."""
    fb = get_feedback_store()

    with st.sidebar:
        st.header("Pipeline Status")
        step = st.session_state.current_step
        status_items = [
            ("1. Upload", step >= 1, st.session_state.input_image is not None),
            ("2. Reference Match", step >= 2, st.session_state.selected_ref is not None),
            ("3. Dimensions", step >= 3, st.session_state.approved_dimensions is not None),
            ("4. CAD Preview", step >= 4, st.session_state.final_dxf is not None),
            ("5. Download", step >= 5, st.session_state.feedback_submitted),
        ]
        for label, active, done in status_items:
            if done:
                st.write(f"\u2705 {label}")
            elif active:
                st.write(f"\U0001F7E1 {label}")
            else:
                st.write(f"\u26AA {label}")

        st.divider()

        # Reset button
        if st.button("Reset Pipeline"):
            old_session = st.session_state.session_id
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            # Create a fresh session
            st.session_state.session_id = fb.create_session()
            st.rerun()

        # Learning dashboard
        render_sidebar_metrics(fb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.title("\U0001F3CA Swimming Pool Sketch-to-CAD")
    st.caption(
        "5-step wizard: Upload \u2192 Match \u2192 Dimensions \u2192 "
        "CAD Preview \u2192 Download"
    )

    render_progress_bar()
    render_sidebar()

    # Get shared resources
    fb = get_feedback_store()
    cd = get_correction_dict(fb)
    cr = get_confidence_router()

    # Route to current step
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
        render_cad_preview(feedback_store=fb)
    elif step == 5:
        render_download(feedback_store=fb)
    else:
        st.error(f"Unknown step: {step}")


if __name__ == "__main__":
    main()
