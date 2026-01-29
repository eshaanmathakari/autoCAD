"""
Sketch-to-DXF Converter - Streamlit Application

Converts hand-drawn floor plans and engineering sketches to DXF CAD files
using Google's Gemini Vision API with two-pass extraction for noisy images.
"""

import streamlit as st

from src.config import SUPPORTED_IMAGE_TYPES, GOOGLE_API_KEY
from src.image_processor import ImageProcessor
from src.gemini_extractor import GeminiGeometryExtractor, ExtractionError, create_mock_geometry
from src.dxf_synthesizer import DXFSynthesizer
from src.dxf_viewer import display_dxf_viewer


def main():
    st.set_page_config(page_title="Sketch to DXF", layout="wide")
    st.title("Sketch to DXF Converter")

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        # API status
        if GOOGLE_API_KEY:
            st.success("API Key: Configured")
        else:
            st.error("API Key: Missing")
            st.caption("Set GOOGLE_API_KEY in .env file")

        st.divider()

        # Extraction mode
        st.subheader("Extraction Mode")
        extraction_mode = st.radio(
            "Mode",
            ["Two-Pass (Recommended)", "Single-Pass"],
            help="Two-pass: extracts walls first, then text. Better for noisy plans."
        )

        # Preprocessing
        st.subheader("Preprocessing")
        show_preprocessed = st.checkbox(
            "Show preprocessed image",
            value=True,
            help="Display the line-extracted image used for wall detection"
        )

        st.divider()

        # DXF options
        st.subheader("DXF Output")
        dxf_version = st.selectbox("DXF Version", ["R2010", "R2007", "R2000"])
        scale = st.number_input("Scale", value=1.0, min_value=0.1, max_value=10.0)

        st.divider()

        # Debug
        use_mock = st.checkbox("Use Mock Data (no API)")

    # Main area
    uploaded = st.file_uploader(
        "Upload floor plan or sketch",
        type=SUPPORTED_IMAGE_TYPES
    )

    if not uploaded:
        st.info("Upload an image to get started.")
        return

    # Load image
    try:
        image = ImageProcessor.load_from_upload(uploaded)
        width, height = ImageProcessor.get_dimensions(image)
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        return

    # Display columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        st.caption(f"{width} x {height} px")

    # Preprocess image
    preprocessed = None
    if extraction_mode == "Two-Pass (Recommended)":
        try:
            preprocessed = ImageProcessor.extract_lines_only(image)
            if show_preprocessed:
                with col2:
                    st.subheader("Preprocessed (Lines Only)")
                    st.image(preprocessed, use_container_width=True)
                    st.caption("Edge detection applied")
        except Exception as e:
            st.warning(f"Preprocessing failed: {e}. Using original image.")

    # Extract button
    if st.button("Extract Geometry", type="primary"):
        geometry = None

        if use_mock:
            geometry = create_mock_geometry(width, height)
            st.info("Using mock data")
        elif not GOOGLE_API_KEY:
            st.error("Configure API key first")
            return
        else:
            with st.spinner("Analyzing with Gemini..."):
                try:
                    extractor = GeminiGeometryExtractor()

                    if extraction_mode == "Two-Pass (Recommended)":
                        st.text("Pass 1: Extracting walls...")
                        geometry = extractor.extract_two_pass(image, preprocessed)
                        st.text("Pass 2: Extracting text...")
                    else:
                        geometry = extractor.extract_validated(image)

                except ExtractionError as e:
                    st.error(f"Extraction failed: {e}")
                    return
                except Exception as e:
                    st.error(f"Error: {e}")
                    return

        if geometry:
            st.session_state["geometry"] = geometry

            # Generate DXF
            synth = DXFSynthesizer(version=dxf_version)
            dxf_bytes = synth.generate(geometry, scale=scale)
            st.session_state["dxf"] = dxf_bytes

            st.success(f"Extracted {len(geometry.entities)} entities (confidence: {geometry.metadata.confidence_score:.0%})")

    # Results
    if "geometry" in st.session_state:
        geometry = st.session_state["geometry"]
        dxf_bytes = st.session_state.get("dxf")

        st.divider()

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Extracted Entities")

            # Entity counts
            counts = geometry.entity_counts
            if counts:
                for entity_type, count in counts.items():
                    st.write(f"- {entity_type}: {count}")
            else:
                st.warning("No entities extracted")

            # Warnings
            if geometry.warnings:
                with st.expander(f"Warnings ({len(geometry.warnings)})"):
                    for w in geometry.warnings:
                        st.write(f"- {w}")

            # JSON
            with st.expander("Raw JSON"):
                st.json(geometry.model_dump())

        with col_b:
            st.subheader("Download")

            if dxf_bytes:
                st.download_button(
                    "Download DXF",
                    data=dxf_bytes,
                    file_name="output.dxf",
                    mime="application/dxf"
                )

            st.download_button(
                "Download JSON",
                data=geometry.model_dump_json(indent=2),
                file_name="geometry.json",
                mime="application/json"
            )

        # DXF Preview section
        if dxf_bytes:
            st.divider()
            st.subheader("DXF Preview")
            st.caption("Pan: drag | Zoom: scroll | Click buttons for Fit/Reset/Grid/Layers")

            display_dxf_viewer(dxf_bytes, height=450)


if __name__ == "__main__":
    main()
