"""
Step 3 — Dimension Verification: OCR results with confidence indicators,
editable dimension inputs, approval gate, and DXF generation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from sketch_to_cad.learning.feedback_store import FeedbackStore
    from sketch_to_cad.learning.correction_dict import CorrectionDict
    from sketch_to_cad.learning.confidence_router import ConfidenceRouter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Part 1 imports — resolved lazily to avoid heavy deps at load time
# ---------------------------------------------------------------------------
_OCR_AVAILABLE = None
_PIPELINE_AVAILABLE = None
OCREnsemble = None
DimensionResult = None
parse_imperial = None
ImperialDimension = None
deform_to_dimensions = None
DeformationResult = None
run_sandbox = None
SandboxResult = None
verify_bbox = None
VerificationResult = None


def _resolve_ocr_imports():
    """Lazily resolve OCR imports from Part 1 or PoC fallback."""
    global _OCR_AVAILABLE, OCREnsemble, DimensionResult, parse_imperial, ImperialDimension
    if _OCR_AVAILABLE is not None:
        return
    try:
        from sketch_to_cad.ocr.ensemble import OCREnsemble as _E, DimensionResult as _D
        from sketch_to_cad.ocr.imperial_parser import parse as _p, ImperialDimension as _I
        OCREnsemble, DimensionResult = _E, _D
        parse_imperial, ImperialDimension = _p, _I
        _OCR_AVAILABLE = True
    except (ImportError, Exception):
        try:
            from poc.ocr_ensemble import OCREnsemble as _E, DimensionResult as _D
            from poc.imperial_parser import parse as _p, ImperialDimension as _I
            OCREnsemble, DimensionResult = _E, _D
            parse_imperial, ImperialDimension = _p, _I
            _OCR_AVAILABLE = True
        except (ImportError, Exception):
            _OCR_AVAILABLE = False


def _resolve_pipeline_imports():
    """Lazily resolve deformation and sandbox imports."""
    global _PIPELINE_AVAILABLE
    global deform_to_dimensions, DeformationResult
    global run_sandbox, SandboxResult
    global verify_bbox, VerificationResult
    if _PIPELINE_AVAILABLE is not None:
        return
    try:
        from sketch_to_cad.cad_engine.deformation import (
            deform_to_dimensions as _dtd,
            DeformationResult as _dr,
        )
        from sketch_to_cad.verification.sandbox import (
            run_sandbox as _rs,
            SandboxResult as _sr,
        )
        from sketch_to_cad.verification.dimension_compare import (
            verify_bbox as _vb,
            VerificationResult as _vr,
        )
        deform_to_dimensions, DeformationResult = _dtd, _dr
        run_sandbox, SandboxResult = _rs, _sr
        verify_bbox, VerificationResult = _vb, _vr
        _PIPELINE_AVAILABLE = True
    except (ImportError, Exception) as exc:
        logger.info("Pipeline imports unavailable: %s", exc)
        _PIPELINE_AVAILABLE = False


def _generate_dxf(ref, length_inches: float, width_inches: float):
    """Run deformation (with sandbox) to produce final DXF bytes."""
    _resolve_pipeline_imports()
    if not _PIPELINE_AVAILABLE:
        return

    dxf_path = getattr(ref, "dxf_path", "")
    if not dxf_path:
        return

    try:
        result = run_sandbox(
            reference_dxf_path=dxf_path,
            target_length_inches=length_inches,
            target_width_inches=width_inches,
        )
        if result.final_dxf_bytes:
            st.session_state.final_dxf = result.final_dxf_bytes
            st.session_state.sandbox_result = result
            if result.final_verification:
                st.session_state.verification = result.final_verification
    except Exception:
        try:
            result = deform_to_dimensions(
                reference_dxf_path=dxf_path,
                target_length_inches=length_inches,
                target_width_inches=width_inches,
            )
            if result.dxf_bytes:
                st.session_state.final_dxf = result.dxf_bytes
                st.session_state.deformation = result
        except Exception as e:
            logger.warning("DXF generation failed: %s", e)


def render_verification(
    feedback_store: "FeedbackStore | None" = None,
    correction_dict: "CorrectionDict | None" = None,
    confidence_router: "ConfidenceRouter | None" = None,
):
    """Render the Dimension Verification screen (wizard step 3)."""
    _resolve_ocr_imports()

    st.header("Step 3 — Dimension Verification")

    ref = st.session_state.get("selected_ref")
    if ref is None:
        st.warning("Complete Step 2 first.")
        return

    meta = ref.metadata if hasattr(ref, "metadata") else {}
    st.info(
        f"Reference: {meta.get('name', getattr(ref, 'folder_id', '?'))} "
        f"(type: {meta.get('pool_type', '?')})"
    )

    col_input, col_output = st.columns(2)

    # ------------------------------------------------------------------
    # Left column: input image + OCR results + manual entry
    # ------------------------------------------------------------------
    with col_input:
        st.subheader("Input and Dimensions")

        if st.session_state.get("input_image") is not None:
            st.image(
                st.session_state.input_image,
                caption="Input Sketch",
                width="stretch",
            )

        st.markdown("#### OCR Extraction")

        if not _OCR_AVAILABLE:
            st.warning(
                "OCR module not available. Enter dimensions manually below."
            )
        else:
            run_ocr = st.button(
                "Run OCR",
                disabled=st.session_state.get("input_image") is None,
            )
            if run_ocr and st.session_state.get("input_image") is not None:
                with st.spinner("Running OCR..."):
                    try:
                        corr = correction_dict.as_dict() if correction_dict else None
                        ensemble = OCREnsemble(correction_dict=corr)
                        engines = ensemble.available_engines
                        if not engines:
                            st.warning(
                                "No OCR engines available. Configure "
                                "GOOGLE_API_KEY in Streamlit secrets."
                            )
                        else:
                            st.caption(f"Engines: {', '.join(engines)}")
                            results = ensemble.extract_dimensions(
                                st.session_state.input_image
                            )
                            st.session_state.ocr_results = results
                    except Exception as e:
                        st.error(f"OCR failed: {e}")

            ocr_results = st.session_state.get("ocr_results", [])
            if ocr_results:
                st.caption(f"Found {len(ocr_results)} dimension(s)")
                for i, r in enumerate(ocr_results):
                    conf = getattr(r, "confidence", 0.0)
                    if confidence_router:
                        label = confidence_router.get_label(conf)
                    else:
                        label = "High" if conf > 0.9 else ("Review" if conf > 0.7 else "Low")

                    text_val = getattr(r, "text", "")
                    st.text_input(
                        f"Dim {i + 1} [{label}, {conf:.0%}]",
                        value=text_val,
                        key=f"ocr_dim_{i}",
                    )

        # Pre-fill from fingerprint if available
        fp = st.session_state.get("drawing_fingerprint")
        default_length = st.session_state.get("manual_length", "20'")
        default_width = st.session_state.get("manual_width", "10'")
        if fp:
            if fp.get("length_inches") and default_length == "20'":
                ft = int(fp["length_inches"] // 12)
                inc = int(fp["length_inches"] % 12)
                default_length = f"{ft}'{inc}\"" if inc else f"{ft}'"
            if fp.get("width_inches") and default_width == "10'":
                ft = int(fp["width_inches"] // 12)
                inc = int(fp["width_inches"] % 12)
                default_width = f"{ft}'{inc}\"" if inc else f"{ft}'"

        st.markdown("---")
        st.markdown("**Manual / Edit Dimensions**")
        target_length = st.text_input(
            "Pool Length (e.g. 20', 25'6\")",
            value=default_length,
            key="manual_length_input",
        )
        target_width = st.text_input(
            "Pool Width (e.g. 10', 12')",
            value=default_width,
            key="manual_width_input",
        )

        parsed_len = None
        parsed_wid = None
        if _OCR_AVAILABLE:
            parsed_len = parse_imperial(target_length)
            parsed_wid = parse_imperial(target_width)
        if parsed_len and parsed_wid:
            st.write(
                f"Length: {parsed_len.display} "
                f"({parsed_len.total_inches}\" = {parsed_len.total_mm:.0f} mm)"
            )
            st.write(
                f"Width: {parsed_wid.display} "
                f"({parsed_wid.total_inches}\" = {parsed_wid.total_mm:.0f} mm)"
            )
        elif _OCR_AVAILABLE:
            st.error(
                "Cannot parse dimensions. Use notation like 20', 12'6\", 4\""
            )

    # ------------------------------------------------------------------
    # Right column: dimensions summary
    # ------------------------------------------------------------------
    with col_output:
        st.subheader("Dimensions Summary")

        if parsed_len and parsed_wid:
            st.markdown(
                f"""
                | Dimension | Imperial | Inches | mm |
                |-----------|----------|--------|----|
                | Length | {parsed_len.display} | {parsed_len.total_inches}" | {parsed_len.total_mm:.0f} |
                | Width  | {parsed_wid.display} | {parsed_wid.total_inches}" | {parsed_wid.total_mm:.0f} |
                """
            )

            if ref and hasattr(ref, "metadata"):
                st.caption(
                    f"Pool type: {meta.get('pool_type', 'unknown')} | "
                    f"Stairs: {'Yes' if meta.get('has_stairs') else 'No'}"
                )
        else:
            st.info("Enter valid dimensions in the left panel.")

    # ------------------------------------------------------------------
    # Approve gate
    # ------------------------------------------------------------------
    st.divider()
    if parsed_len and parsed_wid:
        if st.button("Approve Dimensions and Generate", type="primary"):
            st.session_state.approved_dimensions = {
                "length_inches": parsed_len.total_inches,
                "width_inches": parsed_wid.total_inches,
                "length_display": parsed_len.display,
                "width_display": parsed_wid.display,
            }

            # Log OCR corrections
            session_id = st.session_state.get("session_id")
            ocr_results = st.session_state.get("ocr_results", [])
            if feedback_store and session_id:
                for i, r in enumerate(ocr_results):
                    edited_val = st.session_state.get(f"ocr_dim_{i}", "")
                    original_val = getattr(r, "text", "")
                    if edited_val and edited_val != original_val:
                        feedback_store.log_ocr_correction(
                            session_id=session_id,
                            raw_text=original_val,
                            corrected_text=edited_val,
                            field_type="dimension",
                        )
                feedback_store.log_dimension_edit(
                    session_id=session_id,
                    original_value=target_length,
                    edited_value=target_length,
                    dimension_label="length",
                )
                feedback_store.log_dimension_edit(
                    session_id=session_id,
                    original_value=target_width,
                    edited_value=target_width,
                    dimension_label="width",
                )

            with st.spinner("Generating CAD output..."):
                _generate_dxf(
                    ref,
                    parsed_len.total_inches,
                    parsed_wid.total_inches,
                )

            st.session_state.current_step = 4
            st.rerun()
    else:
        st.caption("Enter and validate dimensions to continue.")
