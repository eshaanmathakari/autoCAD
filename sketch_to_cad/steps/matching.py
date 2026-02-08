"""
Step 2 — Reference Matching: display top-5 matches, select gate, feedback logging.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from sketch_to_cad.learning.feedback_store import FeedbackStore


# ---------------------------------------------------------------------------
# Part 1 imports — wrapped in try/except for independent development
# ---------------------------------------------------------------------------
_PART1_AVAILABLE = True
try:
    from sketch_to_cad.matching.index import ReferenceMatcher, MatchResult, ReferencePool  # noqa: F401
except ImportError:
    _PART1_AVAILABLE = False


def _get_matcher():
    """Return a cached ReferenceMatcher (heavy: loads embeddings)."""
    if not _PART1_AVAILABLE:
        return None

    @st.cache_resource(show_spinner="Loading reference library & embeddings...")
    def _init():
        sample_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "templates"
        )
        # Fall back to poc/sample_data if templates/ doesn't exist yet
        if not os.path.isdir(sample_dir):
            sample_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "poc",
                "sample_data",
            )
        return ReferenceMatcher(sample_dir)

    return _init()


def render_matching(feedback_store: "FeedbackStore | None" = None):
    """Render the Reference Matching screen (wizard step 2)."""
    st.header("Step 2 — Reference Matching")
    st.markdown(
        "We'll find the closest matching reference pool templates from the "
        "library using image embeddings."
    )

    if st.session_state.get("input_image") is None:
        st.warning("Please complete Step 1 first — upload a sketch.")
        return

    if not _PART1_AVAILABLE:
        st.error(
            "Matching module not available. Ensure the core pipeline "
            "(Part 1) is installed under `sketch_to_cad.matching`."
        )
        return

    matcher = _get_matcher()
    if matcher is None:
        st.error("Could not initialise the reference matcher.")
        return

    st.caption(
        f"Backend: **{matcher.backend}** | "
        f"References: **{matcher.num_references}**"
    )

    # --- Run matching ---
    if st.button("Find Matches", type="primary"):
        with st.spinner("Computing embeddings & matching..."):
            results = matcher.match(st.session_state.input_image, top_k=5)
            st.session_state.match_results = results

    # --- Display results ---
    results = st.session_state.get("match_results", [])
    if not results:
        st.info("Click **Find Matches** to search the reference library.")
        return

    st.subheader("Top Matches")
    for mr in results:
        ref = mr.reference
        meta = ref.metadata

        col_img, col_info = st.columns([1, 1])
        with col_img:
            if os.path.exists(ref.line_drawing_path):
                st.image(
                    ref.line_drawing_path,
                    caption=f"#{mr.rank} — {meta.get('name', ref.folder_id)}",
                    use_container_width=True,
                )

        with col_info:
            st.metric("Match Score", f"{mr.score:.4f}")
            st.write(f"**Type:** {meta.get('pool_type', 'unknown')}")
            st.write(
                f"**Stairs:** {'Yes' if meta.get('has_stairs') else 'No'}"
            )

            if st.button("Select this match", key=f"sel_{ref.folder_id}"):
                st.session_state.selected_ref = ref

                # Log to feedback store
                session_id = st.session_state.get("session_id")
                if feedback_store and session_id:
                    feedback_store.log_match_selection(
                        session_id=session_id,
                        reference_id=ref.folder_id,
                        rank=mr.rank,
                        score=mr.score,
                        action="accept",
                    )
                    # Log skipped matches as rejections
                    for other in results:
                        if other.reference.folder_id != ref.folder_id:
                            feedback_store.log_match_selection(
                                session_id=session_id,
                                reference_id=other.reference.folder_id,
                                rank=other.rank,
                                score=other.score,
                                action="skip",
                            )

                st.session_state.current_step = 3
                st.rerun()

        st.divider()
