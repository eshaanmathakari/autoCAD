"""
Step 1 — Upload: file uploader, image preview, and continue gate.
"""

import streamlit as st
from PIL import Image


def render_upload():
    """Render the Upload screen (wizard step 1)."""
    st.header("Step 1 — Upload Pool Sketch")
    st.markdown(
        "Upload a scanned or photographed swimming pool sketch. "
        "Supported formats: **JPEG**, **PNG**."
    )

    col_upload, col_preview = st.columns([1, 2])

    with col_upload:
        uploaded = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg"],
            key="sketch_uploader",
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.session_state.input_image = image

            # Display basic image info
            w, h = image.size
            size_kb = uploaded.size / 1024
            st.caption(f"Resolution: {w} x {h} px  |  Size: {size_kb:.0f} KB")

    with col_preview:
        if st.session_state.get("input_image") is not None:
            st.image(
                st.session_state.input_image,
                caption="Input Sketch",
                use_container_width=True,
            )
        else:
            st.info("Upload a pool sketch image to see a preview here.")

    # Continue gate
    st.divider()
    if st.session_state.get("input_image") is not None:
        if st.button("Continue to Matching", type="primary"):
            st.session_state.current_step = 2
            st.rerun()
    else:
        st.caption("Upload an image to continue.")
