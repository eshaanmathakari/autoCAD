"""
Step 4 — CAD Preview: DXF viewer with layer toggles, side-by-side comparison,
sandbox verification loop, and visual comparison overlay.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import streamlit as st
import streamlit.components.v1 as components

if TYPE_CHECKING:
    from sketch_to_cad.learning.feedback_store import FeedbackStore

# ---------------------------------------------------------------------------
# Part 1 imports — resolved lazily to avoid heavy PoC deps at load time
# ---------------------------------------------------------------------------
_PIPELINE_AVAILABLE = None
_VIEWER_AVAILABLE = None
_SANDBOX_AVAILABLE = None
_VISUAL_COMPARE_AVAILABLE = None

deform_to_dimensions = None
DeformationResult = None
verify_bbox = None
VerificationResult = None
generate_viewer_html = None
run_sandbox = None
SandboxResult = None
compare_visual = None

_DEFAULT_POOL_LAYERS = [
    "POOL_OUTLINE",
    "STAIRS",
    "DIMENSIONS",
    "EQUIPMENT",
    "COPING",
    "SAFETY_LEDGE",
    "LINER_SEAMS",
]
_POOL_LAYERS = _DEFAULT_POOL_LAYERS


def _resolve_pipeline_imports():
    """Lazily resolve pipeline imports from Part 1 or PoC fallback."""
    global _PIPELINE_AVAILABLE, _VIEWER_AVAILABLE, _SANDBOX_AVAILABLE
    global _VISUAL_COMPARE_AVAILABLE, _POOL_LAYERS
    global deform_to_dimensions, DeformationResult, verify_bbox, VerificationResult
    global generate_viewer_html, run_sandbox, SandboxResult, compare_visual

    if _PIPELINE_AVAILABLE is not None:
        return  # already resolved

    # -- Pipeline (deformation + verification) --
    _PIPELINE_AVAILABLE = False
    try:
        from sketch_to_cad.verification.sandbox import run_sandbox as _rs, SandboxResult as _sr
        from sketch_to_cad.cad_engine.deformation import deform_to_dimensions as _dtd, DeformationResult as _dr
        from sketch_to_cad.verification.dimension_compare import verify_bbox as _vb, VerificationResult as _vr
        deform_to_dimensions, DeformationResult = _dtd, _dr
        verify_bbox, VerificationResult = _vb, _vr
        run_sandbox, SandboxResult = _rs, _sr
        _PIPELINE_AVAILABLE = True
        _SANDBOX_AVAILABLE = True
    except (ImportError, Exception):
        _SANDBOX_AVAILABLE = False
        try:
            from poc.template_deformer import deform_to_dimensions as _dtd, DeformationResult as _dr
            from poc.dimension_checker import verify_bbox as _vb, VerificationResult as _vr
            deform_to_dimensions, DeformationResult = _dtd, _dr
            verify_bbox, VerificationResult = _vb, _vr
            _PIPELINE_AVAILABLE = True
        except (ImportError, Exception):
            pass

    # -- Viewer --
    _VIEWER_AVAILABLE = False
    try:
        from sketch_to_cad.cad_engine.preview import generate_viewer_html as _gvh
        generate_viewer_html = _gvh
        _VIEWER_AVAILABLE = True
    except (ImportError, Exception):
        try:
            from poc.dxf_viewer_lite import generate_viewer_html as _gvh
            generate_viewer_html = _gvh
            _VIEWER_AVAILABLE = True
        except (ImportError, Exception):
            pass

    # -- Visual compare --
    _VISUAL_COMPARE_AVAILABLE = False
    try:
        from sketch_to_cad.verification.visual_compare import compare_visual as _cv
        compare_visual = _cv
        _VISUAL_COMPARE_AVAILABLE = True
    except (ImportError, Exception):
        pass

    # -- Pool layers --
    try:
        from sketch_to_cad.domain.constants import POOL_LAYERS as _imported
        _POOL_LAYERS = list(_imported) if _imported else _DEFAULT_POOL_LAYERS
    except (ImportError, Exception):
        pass  # keep defaults


def _show_dxf_viewer(dxf_bytes: bytes, height: int = 500):
    """Render the DXF viewer using Three.js."""
    _resolve_pipeline_imports()
    if not dxf_bytes or len(dxf_bytes) < 100:
        st.error(
            "DXF generation failed — file is empty or too small. "
            "Check reference template."
        )
        return
    if not _VIEWER_AVAILABLE:
        st.warning("DXF viewer not available.")
        return
    try:
        html = generate_viewer_html(dxf_bytes, height)
        components.html(html, height=height + 20, scrolling=False)
    except Exception as e:
        st.warning(f"DXF viewer unavailable: {e}")


def render_cad_preview(feedback_store: "FeedbackStore | None" = None):
    """Render the CAD Preview screen (wizard step 4)."""
    _resolve_pipeline_imports()

    st.header("Step 4 — CAD Preview & Verification")

    ref = st.session_state.get("selected_ref")
    dims = st.session_state.get("approved_dimensions")
    if ref is None or dims is None:
        st.warning("Please complete Step 3 first — approve dimensions.")
        return

    if not _PIPELINE_AVAILABLE:
        st.error(
            "Pipeline modules not available. Ensure the core pipeline "
            "(Part 1) is installed under `sketch_to_cad`."
        )
        return

    target_length = dims["length_inches"]
    target_width = dims["width_inches"]
    ref_dxf_path = getattr(ref, "dxf_path", "")

    # ------------------------------------------------------------------
    # Generate & Verify button
    # ------------------------------------------------------------------
    gen_col, info_col = st.columns([2, 1])

    with gen_col:
        if _SANDBOX_AVAILABLE:
            if st.button("Generate & Verify (Sandbox)", type="primary"):
                with st.spinner("Running verification sandbox..."):
                    progress_bar = st.progress(0, text="Iteration 1 of 3...")
                    try:
                        result = run_sandbox(
                            ref_dxf_path,
                            target_length,
                            target_width,
                            max_iterations=3,
                        )
                        st.session_state.sandbox_result = result
                        st.session_state.final_dxf = result.final_dxf_bytes

                        # Update progress
                        for i in range(result.total_iterations):
                            progress_bar.progress(
                                (i + 1) / 3,
                                text=f"Iteration {i + 1} of 3 "
                                f"{'— converged!' if result.converged and i == result.total_iterations - 1 else ''}",
                            )

                        if result.converged:
                            st.success(
                                f"Sandbox converged in {result.total_iterations} "
                                f"iteration(s)."
                            )
                        else:
                            st.warning(
                                f"Sandbox did not converge after "
                                f"{result.total_iterations} iterations. "
                                f"Best result is shown."
                            )

                        # Log to feedback store
                        session_id = st.session_state.get("session_id")
                        if feedback_store and session_id:
                            feedback_store.log_verification(
                                session_id=session_id,
                                iteration=result.total_iterations,
                                passed=1 if result.converged else 0,
                                total_checked=1,
                                max_error=0.0,
                            )
                    except Exception as e:
                        st.error(f"Sandbox failed: {e}")
        else:
            # Fallback: single-pass generate + verify
            if st.button("Generate & Verify", type="primary"):
                with st.spinner("Deforming template..."):
                    try:
                        result = deform_to_dimensions(
                            ref_dxf_path,
                            target_length_inches=target_length,
                            target_width_inches=target_width,
                        )
                        st.session_state.deformation = result

                        verification = verify_bbox(
                            result.edges,
                            target_length_inches=target_length,
                            target_width_inches=target_width,
                            tolerance_inches=1.0,
                        )
                        st.session_state.verification = verification
                        st.session_state.final_dxf = result.dxf_bytes

                        session_id = st.session_state.get("session_id")
                        if feedback_store and session_id:
                            feedback_store.log_verification(
                                session_id=session_id,
                                iteration=1,
                                passed=verification.total_passed,
                                total_checked=verification.total_checked,
                                max_error=verification.max_error_inches,
                            )
                    except Exception as e:
                        st.error(f"Generation failed: {e}")

    with info_col:
        # Show sandbox iteration details
        sandbox_result = st.session_state.get("sandbox_result")
        if sandbox_result:
            st.caption(
                f"Iterations: {sandbox_result.total_iterations} | "
                f"Converged: {'Yes' if sandbox_result.converged else 'No'}"
            )

        # Show deformation details (fallback path)
        deform = st.session_state.get("deformation")
        if deform and not sandbox_result:
            st.caption(
                f"Scale: X={deform.scale_x:.3f}, Y={deform.scale_y:.3f} | "
                f"Edges: {len(deform.edges)}"
            )

    # ------------------------------------------------------------------
    # DXF viewer + layer toggles
    # ------------------------------------------------------------------
    dxf_bytes = st.session_state.get("final_dxf")
    if dxf_bytes:
        st.divider()

        # Layer toggle checkboxes
        st.markdown("**Layer Visibility:**")
        layer_cols = st.columns(len(_POOL_LAYERS))
        active_layers = []
        for i, layer in enumerate(_POOL_LAYERS):
            with layer_cols[i]:
                if st.checkbox(layer.replace("_", " ").title(), value=True, key=f"layer_{layer}"):
                    active_layers.append(layer)

        # Side-by-side: original sketch vs generated CAD
        col_sketch, col_cad = st.columns(2)
        with col_sketch:
            st.subheader("Original Sketch")
            if st.session_state.get("input_image") is not None:
                st.image(
                    st.session_state.input_image,
                    caption="Uploaded Sketch",
                    width="stretch",
                )
        with col_cad:
            st.subheader("Generated CAD")
            _show_dxf_viewer(dxf_bytes, height=450)

        # Visual comparison overlay (if available)
        if _VISUAL_COMPARE_AVAILABLE and st.session_state.get("input_image"):
            with st.expander("Visual Comparison Overlay"):
                try:
                    comp_result = compare_visual(
                        st.session_state.input_image, dxf_bytes
                    )
                    if hasattr(comp_result, "overlay_image") and comp_result.overlay_image is not None:
                        st.image(
                            comp_result.overlay_image,
                            caption="Overlay: Sketch vs CAD",
                            width="stretch",
                        )
                    if hasattr(comp_result, "ssim_score"):
                        st.metric("SSIM Score", f"{comp_result.ssim_score:.4f}")
                except Exception as e:
                    st.caption(f"Visual comparison not available: {e}")

        # Verification details
        verification = st.session_state.get("verification")
        if verification:
            if verification.all_passed:
                st.success(
                    f"Verification PASSED: {verification.total_passed}/"
                    f"{verification.total_checked} dimensions within tolerance"
                )
            else:
                st.error(
                    f"Verification: {verification.total_passed}/"
                    f"{verification.total_checked} passed, "
                    f"{verification.total_failed} failed "
                    f"(max error: {verification.max_error_inches:.2f}\")"
                )
            with st.expander("Dimension Checks"):
                for check in verification.checks:
                    icon = "\u2705" if check.passed else "\u274C"
                    label = check.label or f"Edge {check.edge_index}"
                    st.write(
                        f"{icon} {label}: expected {check.expected_inches}\" "
                        f"got {check.actual_inches}\" "
                        f"(error: {check.error_inches:.2f}\")"
                    )

        # Action buttons
        st.divider()
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Request Revision"):
                # Clear previous results so user can re-generate
                st.session_state.pop("sandbox_result", None)
                st.session_state.pop("deformation", None)
                st.session_state.pop("verification", None)
                st.session_state.pop("final_dxf", None)
                st.rerun()
        with btn_col2:
            if st.button("Approve & Continue to Download", type="primary"):
                st.session_state.current_step = 5
                st.rerun()
