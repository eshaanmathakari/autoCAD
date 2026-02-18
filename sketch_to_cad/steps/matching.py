"""
Step 2 — Reference Matching: display top matches, selection, feedback logging.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from sketch_to_cad.learning.feedback_store import FeedbackStore


# ---------------------------------------------------------------------------
# Part 1 imports
# ---------------------------------------------------------------------------
_PART1_AVAILABLE = True
try:
    from sketch_to_cad.matching.index import ReferenceMatcher, MatchResult, ReferencePool  # noqa: F401
except ImportError:
    _PART1_AVAILABLE = False


def _get_matcher():
    """Return a cached ReferenceMatcher (loads embeddings once)."""
    if not _PART1_AVAILABLE:
        return None

    @st.cache_resource(show_spinner="Loading reference library...")
    def _init():
        sample_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "templates"
        )
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
        "Find the closest matching reference pool template from the library."
    )

    if st.session_state.get("input_image") is None:
        st.warning("Complete Step 1 first.")
        return

    if not _PART1_AVAILABLE:
        st.error(
            "Matching module not available. Ensure the core pipeline "
            "is installed under sketch_to_cad.matching."
        )
        return

    matcher = _get_matcher()
    if matcher is None:
        st.error("Could not initialise the reference matcher.")
        return

    st.caption(
        f"Backend: {matcher.backend}  |  "
        f"References: {matcher.num_references}"
    )
    type_counts = getattr(matcher, "reference_type_counts", {})
    if type_counts:
        histogram = ", ".join(
            f"{pool_type}: {count}" for pool_type, count in sorted(type_counts.items())
        )
        st.caption(f"Reference types: {histogram}")
    if matcher.num_references <= 5:
        st.warning(
            "Reference library is very small. Upload/ingest additional references "
            "for better matching quality."
        )

    # Display fingerprint summary if available
    fp = st.session_state.get("drawing_fingerprint")
    if fp:
        fp_parts = []
        if fp.get("pool_type"):
            fp_parts.append(f"Type: {fp['pool_type']}")
        if fp.get("length_inches") and fp.get("width_inches"):
            fp_parts.append(
                f"Dimensions: {fp['length_inches']:.0f}\" x {fp['width_inches']:.0f}\""
            )
        if fp.get("has_stairs") is not None:
            fp_parts.append(f"Stairs: {'Yes' if fp['has_stairs'] else 'No'}")
        if fp_parts:
            st.info("Fingerprint: " + "  |  ".join(fp_parts))

    # --- Run matching ---
    if st.button("Find Matches", type="primary"):
        with st.spinner("Computing similarity..."):
            if fp and hasattr(matcher, "match_with_fingerprint"):
                results = matcher.match_with_fingerprint(
                    st.session_state.input_image, fp, top_k=3
                )
            else:
                results = matcher.match(st.session_state.input_image, top_k=3)
            st.session_state.match_results = results

    # --- Display results ---
    results = st.session_state.get("match_results", [])
    if not results:
        st.info("Press Find Matches to search the reference library.")
        return

    if fp and getattr(matcher, "last_filter_applied", False):
        if getattr(matcher, "last_filter_fallback", False):
            predicted = getattr(matcher, "last_predicted_type", None) or "unknown"
            st.warning(
                f"No references matched predicted type '{predicted}'. "
                "Showing closest matches from all types."
            )
        else:
            predicted = getattr(matcher, "last_predicted_type", None) or "unknown"
            st.info(f"Strict type filter applied: {predicted}")

    st.subheader(f"Top matches ({len(results)})")
    for mr in results:
        ref = mr.reference
        meta = ref.metadata

        col_img, col_info = st.columns([1, 1])
        with col_img:
            if os.path.exists(ref.line_drawing_path):
                st.image(
                    ref.line_drawing_path,
                    caption=f"#{mr.rank} - {meta.get('name', ref.folder_id)}",
                    width="stretch",
                )

        with col_info:
            st.metric("Score", f"{mr.score:.4f}")
            st.write(f"Type: {meta.get('pool_type', 'unknown')}")
            st.write(
                f"Stairs: {'Yes' if meta.get('has_stairs') else 'No'}"
            )

            if st.button("Select", key=f"sel_{ref.folder_id}"):
                st.session_state.selected_ref = ref

                session_id = st.session_state.get("session_id")
                if feedback_store and session_id:
                    feedback_store.log_match_selection(
                        session_id=session_id,
                        reference_id=ref.folder_id,
                        rank=mr.rank,
                        score=mr.score,
                        action="accept",
                    )
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
