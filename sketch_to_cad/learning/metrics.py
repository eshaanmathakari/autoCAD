"""
Learning Metrics Dashboard — sidebar rendering + analytics queries.

Displays aggregate stats from the FeedbackStore in the Streamlit sidebar:
total sessions, average accuracy, correction trends, match accept rate,
and top correction patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from sketch_to_cad.learning.feedback_store import FeedbackStore


def render_sidebar_metrics(feedback_store: "FeedbackStore"):
    """Render the learning metrics panel inside ``st.sidebar``."""
    with st.sidebar:
        st.markdown("---")
        st.subheader("Learning Dashboard")

        # -- Key metrics row ------------------------------------------------
        total_sessions = feedback_store.get_session_count()
        sessions_this_week = feedback_store.get_sessions_this_week()
        avg_accuracy = feedback_store.get_avg_accuracy()
        correction_count = feedback_store.get_correction_count()
        accept_rate = feedback_store.get_match_accept_rate()

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Total Sessions",
                total_sessions,
                delta=f"+{sessions_this_week} this week" if sessions_this_week else None,
            )
        with col2:
            st.metric(
                "Avg Accuracy",
                f"{avg_accuracy:.1f}/5" if avg_accuracy else "N/A",
                delta=f"+{avg_accuracy - 3.0:.1f}" if avg_accuracy and avg_accuracy > 3.0 else None,
            )

        col3, col4 = st.columns(2)
        with col3:
            st.metric("OCR Corrections", correction_count)
        with col4:
            st.metric("Match Accept", f"{accept_rate:.0f}%")

        # -- Verification pass rate -----------------------------------------
        vpr = feedback_store.get_verification_pass_rate()
        if vpr > 0:
            st.metric("Verification Pass", f"{vpr:.0f}%")

        # -- Top correction patterns ----------------------------------------
        corrections = feedback_store.get_recent_corrections(limit=5)
        if corrections:
            st.markdown("**Top Learned Corrections**")
            for c in corrections:
                st.caption(
                    f'`{c["raw"]}` -> `{c["corrected"]}` ({c["count"]}x)'
                )


def get_metrics_summary(feedback_store: "FeedbackStore") -> dict:
    """Return a dictionary of all key metrics (for JSON export)."""
    return {
        "total_sessions": feedback_store.get_session_count(),
        "completed_sessions": feedback_store.get_completed_session_count(),
        "sessions_this_week": feedback_store.get_sessions_this_week(),
        "avg_accuracy_rating": feedback_store.get_avg_accuracy(),
        "avg_match_rating": feedback_store.get_avg_match_rating(),
        "match_accept_rate_pct": feedback_store.get_match_accept_rate(),
        "ocr_correction_count": feedback_store.get_correction_count(),
        "verification_pass_rate_pct": feedback_store.get_verification_pass_rate(),
        "top_corrections": feedback_store.get_recent_corrections(limit=10),
    }
