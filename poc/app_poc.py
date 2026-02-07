"""
Swimming Pool Sketch-to-CAD — Proof of Concept

Three-tab Streamlit app demonstrating the 3-step pipeline:
  Tab 1 — Reference Matching:  Upload sketch → find nearest pool template
  Tab 2 — Dimension Extraction: OCR dimensions → generate line drawing → verify
  Tab 3 — CAD Output:          Generate final DXF → preview → download
"""

import os
import sys
import json

import streamlit as st
from PIL import Image

# Ensure poc package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load Streamlit Cloud secrets into env if available
if hasattr(st, "secrets"):
    for key in ("GOOGLE_API_KEY", "GEMINI_MODEL"):
        if key in st.secrets:
            os.environ.setdefault(key, st.secrets[key])

from poc.imperial_parser import parse as parse_imperial, ImperialDimension
from poc.reference_matcher import ReferenceMatcher
from poc.template_deformer import deform_to_dimensions
from poc.dimension_checker import verify_dimensions

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
st.set_page_config(page_title="Pool CAD PoC", layout="wide", page_icon="\U0001F3CA")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
DEFAULTS = {
    "step": 1,
    "input_image": None,
    "matcher": None,
    "match_results": [],
    "selected_ref": None,
    "dimensions": [],
    "deformation": None,
    "verification": None,
    "final_dxf": None,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ---------------------------------------------------------------------------
# Lazy-init matcher (heavy: loads CLIP / FAISS on first call)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading reference library & embeddings...")
def get_matcher():
    return ReferenceMatcher(SAMPLE_DIR)


# ---------------------------------------------------------------------------
# DXF viewer (reuse from existing codebase)
# ---------------------------------------------------------------------------
def show_dxf_viewer(dxf_bytes: bytes, height: int = 450):
    """Render the Three.js DXF viewer."""
    if not dxf_bytes or len(dxf_bytes) < 100:
        st.error("DXF generation failed — file is empty or too small. Check reference template.")
        return
    try:
        # Try the main project viewer first, fall back to bundled one
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from src.dxf_viewer import generate_viewer_html
        except ImportError:
            from poc.dxf_viewer_lite import generate_viewer_html
        import streamlit.components.v1 as components
        html = generate_viewer_html(dxf_bytes, height)
        components.html(html, height=height + 20, scrolling=False)
    except Exception as e:
        st.warning(f"DXF viewer unavailable: {e}")


# ===================================================================
# TAB 1: Reference Matching
# ===================================================================
def tab_reference_matching():
    st.header("Step 1 — Reference Matching")
    st.markdown(
        "Upload a scanned pool sketch. We'll find the closest matching "
        "reference from the library using image embeddings."
    )

    col_upload, col_results = st.columns([1, 2])

    with col_upload:
        uploaded = st.file_uploader(
            "Upload pool sketch", type=["png", "jpg", "jpeg"], key="upload"
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.session_state.input_image = image
            st.image(image, caption="Input Sketch", width="stretch")

    with col_results:
        if st.session_state.input_image is not None:
            matcher = get_matcher()
            st.caption(
                f"Backend: **{matcher.backend}** | "
                f"References: **{matcher.num_references}**"
            )

            if st.button("Find Matches", type="primary"):
                with st.spinner("Computing embeddings & matching..."):
                    results = matcher.match(st.session_state.input_image, top_k=5)
                    st.session_state.match_results = results

            if st.session_state.match_results:
                st.subheader("Top Matches")
                for mr in st.session_state.match_results:
                    ref = mr.reference
                    meta = ref.metadata
                    col_img, col_info = st.columns([1, 1])
                    with col_img:
                        if os.path.exists(ref.line_drawing_path):
                            st.image(
                                ref.line_drawing_path,
                                caption=f"#{mr.rank} — {meta.get('name', ref.folder_id)}",
                                width="stretch",
                            )
                    with col_info:
                        st.write(f"**Score:** {mr.score:.4f}")
                        st.write(f"**Type:** {meta.get('pool_type', 'unknown')}")
                        st.write(f"**Stairs:** {'Yes' if meta.get('has_stairs') else 'No'}")
                        if st.button(
                            f"Select this match",
                            key=f"sel_{ref.folder_id}",
                        ):
                            st.session_state.selected_ref = ref
                            st.session_state.step = 2
                            st.rerun()
                    st.divider()
        else:
            st.info("Upload a pool sketch image to begin.")


# ===================================================================
# TAB 2: Dimension Extraction & Verification
# ===================================================================
def tab_dimension_extraction():
    st.header("Step 2 — Dimension Extraction & Line Drawing")

    ref = st.session_state.selected_ref
    if ref is None:
        st.warning("Please complete Step 1 first — select a reference match.")
        return

    meta = ref.metadata
    st.success(
        f"Reference: **{meta.get('name', ref.folder_id)}** "
        f"(type: {meta.get('pool_type', '?')})"
    )

    col_input, col_output = st.columns(2)

    # --- Left: input image + OCR ---
    with col_input:
        st.subheader("Input & Dimensions")
        if st.session_state.input_image:
            st.image(st.session_state.input_image, caption="Input Sketch", width="stretch")

        st.markdown("#### Extracted Dimensions")
        st.caption(
            "These would come from OCR ensemble in production. "
            "For the PoC, enter target dimensions manually."
        )

        # OCR button
        run_ocr = st.button("Run OCR Ensemble", disabled=st.session_state.input_image is None)
        if run_ocr and st.session_state.input_image is not None:
            with st.spinner("Running OCR ensemble (PaddleOCR + Gemini)..."):
                try:
                    from poc.ocr_ensemble import OCREnsemble
                    ensemble = OCREnsemble()
                    results = ensemble.extract_dimensions(st.session_state.input_image)
                    st.session_state.dimensions = [
                        {
                            "label": r.text,
                            "inches": r.parsed.total_inches if r.parsed else 0,
                            "confidence": r.confidence,
                            "needs_review": r.needs_review,
                        }
                        for r in results
                    ]
                    st.write(f"Found **{len(results)}** dimensions via: {ensemble.available_engines}")
                except Exception as e:
                    st.error(f"OCR failed: {e}")

        # Manual dimension entry
        st.markdown("**Manual / Edit Dimensions:**")
        target_length = st.text_input("Pool Length (e.g. 20', 25'6\")", value="20'")
        target_width = st.text_input("Pool Width (e.g. 10', 12')", value="10'")

        parsed_len = parse_imperial(target_length)
        parsed_wid = parse_imperial(target_width)

        if parsed_len and parsed_wid:
            st.write(
                f"Length: **{parsed_len.display}** ({parsed_len.total_inches}\" = {parsed_len.total_mm:.0f}mm)"
            )
            st.write(
                f"Width: **{parsed_wid.display}** ({parsed_wid.total_inches}\" = {parsed_wid.total_mm:.0f}mm)"
            )
        else:
            st.error("Cannot parse dimensions. Use notation like 20', 12'6\", 4\"")

    # --- Right: deformed line drawing + verification ---
    with col_output:
        st.subheader("Generated Line Drawing")

        if parsed_len and parsed_wid and ref.dxf_path and os.path.exists(ref.dxf_path):
            if st.button("Generate & Verify", type="primary"):
                with st.spinner("Deforming template..."):
                    result = deform_to_dimensions(
                        ref.dxf_path,
                        target_length_inches=parsed_len.total_inches,
                        target_width_inches=parsed_wid.total_inches,
                    )
                    st.session_state.deformation = result

                    # Auto-verify (check both length and width)
                    target_dims = {
                        0: parsed_len.total_inches,
                        1: parsed_wid.total_inches,
                    }

                    verification = verify_dimensions(
                        result.edges,
                        target_dims,
                        tolerance_inches=1.0,
                    )
                    st.session_state.verification = verification

            if st.session_state.deformation and st.session_state.deformation.dxf_bytes:
                deform = st.session_state.deformation
                st.write(f"Scale: X={deform.scale_x:.3f}, Y={deform.scale_y:.3f}")
                st.write(f"Edges: {len(deform.edges)}")

                # Show DXF preview
                show_dxf_viewer(deform.dxf_bytes, height=350)

                # Verification results
                if st.session_state.verification:
                    vr = st.session_state.verification
                    if vr.all_passed:
                        st.success(
                            f"Verification PASSED: {vr.total_passed}/{vr.total_checked} "
                            f"dimensions within tolerance"
                        )
                    else:
                        st.error(
                            f"Verification: {vr.total_passed}/{vr.total_checked} passed, "
                            f"{vr.total_failed} failed (max error: {vr.max_error_inches:.2f}\")"
                        )

                    with st.expander("Dimension Checks"):
                        for check in vr.checks:
                            icon = "\u2705" if check.passed else "\u274C"
                            st.write(
                                f"{icon} Edge {check.edge_index}: "
                                f"expected {check.expected_inches}\" "
                                f"got {check.actual_inches}\" "
                                f"(error: {check.error_inches}\")"
                            )

                # Approve button
                if st.button("Approve & Continue to Step 3"):
                    st.session_state.step = 3
                    st.session_state.final_dxf = deform.dxf_bytes
                    st.rerun()
        else:
            st.info("Enter valid dimensions and click Generate.")


# ===================================================================
# TAB 3: CAD Output
# ===================================================================
def tab_cad_output():
    st.header("Step 3 — Professional CAD Output")

    if st.session_state.final_dxf is None:
        st.warning("Please complete Step 2 first — generate and verify the line drawing.")
        return

    ref = st.session_state.selected_ref
    meta = ref.metadata if ref else {}

    col_preview, col_download = st.columns([2, 1])

    with col_preview:
        st.subheader("DXF Preview")
        show_dxf_viewer(st.session_state.final_dxf, height=500)

    with col_download:
        st.subheader("Download")
        st.download_button(
            label="Download DXF",
            data=st.session_state.final_dxf,
            file_name="pool_output.dxf",
            mime="application/dxf",
            type="primary",
        )

        st.divider()
        st.subheader("Pool Info")
        st.write(f"**Reference:** {meta.get('name', 'Unknown')}")
        st.write(f"**Type:** {meta.get('pool_type', 'Unknown')}")

        if st.session_state.deformation:
            d = st.session_state.deformation
            bbox = d.target_bbox
            length = bbox[2]
            width = bbox[3]
            p_len = ImperialDimension(
                feet=int(length // 12), inches=round(length % 12, 1),
                total_inches=length, total_mm=length * 25.4,
                raw_text="",
            )
            p_wid = ImperialDimension(
                feet=int(width // 12), inches=round(width % 12, 1),
                total_inches=width, total_mm=width * 25.4,
                raw_text="",
            )
            st.write(f"**Length:** {p_len.display}")
            st.write(f"**Width:** {p_wid.display}")

        if st.session_state.verification:
            vr = st.session_state.verification
            if vr.all_passed:
                st.success(f"Verified: {vr.total_passed}/{vr.total_checked} dims OK")
            else:
                st.warning(f"Verified: {vr.total_passed}/{vr.total_checked} dims OK")

        st.divider()
        st.subheader("Pipeline Summary")
        st.markdown("""
        1. **Reference Match** — found closest pool template
        2. **Dimension Extraction** — parsed imperial measurements
        3. **Template Deformation** — scaled reference to target dims
        4. **Verification** — checked generated vs target dimensions
        5. **Output** — professional DXF with pool layers
        """)

        # JSON export
        if st.session_state.deformation:
            export = {
                "reference": meta.get("name", ""),
                "pool_type": meta.get("pool_type", ""),
                "scale_x": st.session_state.deformation.scale_x,
                "scale_y": st.session_state.deformation.scale_y,
                "num_edges": len(st.session_state.deformation.edges),
                "verification_passed": st.session_state.verification.all_passed if st.session_state.verification else None,
            }
            st.download_button(
                "Download JSON Metadata",
                data=json.dumps(export, indent=2),
                file_name="pool_metadata.json",
                mime="application/json",
            )


# ===================================================================
# Main layout
# ===================================================================
def main():
    st.title("\U0001F3CA Swimming Pool Sketch-to-CAD — PoC")
    st.caption(
        "Three-step agentic pipeline: Match → Extract & Verify → Generate CAD"
    )

    # Navigation
    tab1, tab2, tab3 = st.tabs([
        "\U0001F50D Step 1: Reference Match",
        "\U0001F4D0 Step 2: Dimensions & Verify",
        "\U0001F4E6 Step 3: CAD Output",
    ])

    with tab1:
        tab_reference_matching()
    with tab2:
        tab_dimension_extraction()
    with tab3:
        tab_cad_output()

    # Sidebar status
    with st.sidebar:
        st.header("Pipeline Status")
        step = st.session_state.step
        steps = [
            ("1. Reference Match", step >= 1, st.session_state.selected_ref is not None),
            ("2. Dimensions & Verify", step >= 2, st.session_state.verification is not None),
            ("3. CAD Output", step >= 3, st.session_state.final_dxf is not None),
        ]
        for label, active, done in steps:
            if done:
                st.write(f"\u2705 {label}")
            elif active:
                st.write(f"\U0001F7E1 {label}")
            else:
                st.write(f"\u26AA {label}")

        st.divider()
        if st.button("Reset Pipeline"):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

        st.divider()
        st.caption("**PoC Assumptions Tested:**")
        st.caption("1. Imperial OCR parsing")
        st.caption("2. Embedding-based matching")
        st.caption("3. Template deformation")
        st.caption("4. Dimension verification")


if __name__ == "__main__":
    main()
