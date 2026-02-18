"""
Step 4 — Download: DXF and JSON downloads, feedback form, session summary.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from sketch_to_cad.learning.feedback_store import FeedbackStore

# Try to import ImperialDimension for display formatting
try:
    from sketch_to_cad.ocr.imperial_parser import ImperialDimension
except ImportError:
    try:
        from poc.imperial_parser import ImperialDimension
    except ImportError:
        ImperialDimension = None


def render_download(feedback_store: "FeedbackStore | None" = None):
    """Render the Download screen (wizard step 4)."""
    st.header("Step 4 — Download")

    final_dxf = st.session_state.get("final_dxf")
    if final_dxf is None:
        st.warning("Complete Step 3 to generate output files.")
        return

    ref = st.session_state.get("selected_ref")
    meta = ref.metadata if ref and hasattr(ref, "metadata") else {}
    dims = st.session_state.get("approved_dimensions", {})

    col_preview, col_download = st.columns([2, 1])

    # ------------------------------------------------------------------
    # Left: session summary
    # ------------------------------------------------------------------
    with col_preview:
        st.subheader("Session Summary")

        st.markdown(
            f"""
            | Item | Value |
            |------|-------|
            | **Reference** | {meta.get('name', getattr(ref, 'folder_id', 'N/A') if ref else 'N/A')} |
            | **Pool Type** | {meta.get('pool_type', 'Unknown')} |
            | **Length** | {dims.get('length_display', 'N/A')} ({dims.get('length_inches', 'N/A')}") |
            | **Width** | {dims.get('width_display', 'N/A')} ({dims.get('width_inches', 'N/A')}") |
            """
        )

        # Deformation details
        deform = st.session_state.get("deformation")
        if deform:
            st.caption(
                f"Scale factors: X={deform.scale_x:.3f}, Y={deform.scale_y:.3f} | "
                f"Edges: {len(deform.edges)}"
            )

        # Sandbox details
        sandbox_result = st.session_state.get("sandbox_result")
        if sandbox_result:
            st.caption(
                f"Sandbox: {sandbox_result.total_iterations} iteration(s), "
                f"converged: {'Yes' if sandbox_result.converged else 'No'}"
            )

        verification = st.session_state.get("verification")
        if verification:
            st.caption(
                f"Verification: {verification.total_passed}/"
                f"{verification.total_checked} dimensions passed"
            )

        st.markdown("---")
        st.markdown(
            """
            **Pipeline Steps Completed:**
            1. Reference Match
            2. Dimension Extraction
            3. Template Deformation and Verification
            4. Output (DXF with pool layers)
            """
        )

    # ------------------------------------------------------------------
    # Right: downloads + feedback
    # ------------------------------------------------------------------
    with col_download:
        st.subheader("Downloads")

        # DXF download
        st.download_button(
            label="Download DXF",
            data=final_dxf,
            file_name="pool_output.dxf",
            mime="application/dxf",
            type="primary",
        )

        # JSON metadata download
        export_data = {
            "reference": meta.get("name", ""),
            "pool_type": meta.get("pool_type", ""),
            "dimensions": dims,
        }
        if deform:
            export_data.update({
                "scale_x": deform.scale_x,
                "scale_y": deform.scale_y,
                "num_edges": len(deform.edges),
            })
        if verification:
            export_data["verification_passed"] = verification.all_passed
            export_data["verification_pass_rate"] = (
                f"{verification.total_passed}/{verification.total_checked}"
            )
        if sandbox_result:
            export_data["sandbox_iterations"] = sandbox_result.total_iterations
            export_data["sandbox_converged"] = sandbox_result.converged

        st.download_button(
            "Download JSON Metadata",
            data=json.dumps(export_data, indent=2, default=str),
            file_name="pool_metadata.json",
            mime="application/json",
        )

        # Feedback form
        st.divider()
        st.subheader("Feedback")

        if st.session_state.get("feedback_submitted"):
            st.info("Feedback submitted.")
        else:
            accuracy_rating = st.slider(
                "Accuracy Rating",
                min_value=1,
                max_value=5,
                value=4,
                help="How accurate were the final CAD dimensions?",
            )
            match_rating = st.slider(
                "Match Quality",
                min_value=1,
                max_value=5,
                value=4,
                help="How well did the reference template match your sketch?",
            )
            comments = st.text_area(
                "Comments (optional)",
                placeholder="Any feedback on the results...",
            )

            if st.button("Submit Feedback", type="primary"):
                session_id = st.session_state.get("session_id")
                if feedback_store and session_id:
                    feedback_store.log_feedback(
                        session_id=session_id,
                        accuracy_rating=accuracy_rating,
                        match_rating=match_rating,
                        comments=comments,
                    )
                    feedback_store.complete_session(session_id)
                st.session_state.feedback_submitted = True
                st.rerun()
