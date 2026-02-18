"""
Step 1 — Upload: file uploader, image preview, fingerprint extraction, and continue gate.
"""

import logging

import streamlit as st
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy import for fingerprint extraction
_FP_AVAILABLE = None
_extract_fingerprint = None


def _resolve_fp_import():
    global _FP_AVAILABLE, _extract_fingerprint
    if _FP_AVAILABLE is not None:
        return
    try:
        from sketch_to_cad.fingerprint import extract_fingerprint as _efp
        _extract_fingerprint = _efp
        _FP_AVAILABLE = True
    except (ImportError, Exception) as exc:
        logger.info("Fingerprint module not available: %s", exc)
        _FP_AVAILABLE = False


def render_upload():
    """Render the Upload screen (wizard step 1)."""
    st.header("Step 1 — Upload")
    st.markdown(
        "Upload a scanned or photographed swimming pool sketch. "
        "Supported formats: JPEG, PNG."
    )

    col_upload, col_preview = st.columns([1, 2])

    with col_upload:
        uploaded = st.file_uploader(
            "Select image",
            type=["png", "jpg", "jpeg"],
            key="sketch_uploader",
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.session_state.input_image = image

            w, h = image.size
            size_kb = uploaded.size / 1024
            st.caption(f"Resolution: {w} x {h} px  |  Size: {size_kb:.0f} KB")

    with col_preview:
        if st.session_state.get("input_image") is not None:
            st.image(
                st.session_state.input_image,
                caption="Input Sketch",
                width="stretch",
            )
        else:
            st.info("Upload an image to see a preview.")

    st.divider()
    if st.session_state.get("input_image") is not None:
        if st.button("Continue to Matching", type="primary"):
            _resolve_fp_import()
            if _FP_AVAILABLE:
                with st.spinner("Extracting drawing fingerprint..."):
                    try:
                        fp = _extract_fingerprint(st.session_state.input_image)
                        st.session_state.drawing_fingerprint = fp
                    except Exception as e:
                        logger.warning("Fingerprint extraction failed: %s", e)
                        st.session_state.drawing_fingerprint = None
            st.session_state.current_step = 2
            st.rerun()
    else:
        st.caption("Upload an image to continue.")
