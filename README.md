# autoCAD

Transforming hand-drawn sketches to AutoCAD DXF files with a sandbox to preview and edit them.

## Overview

This app converts hand-drawn floor plans and engineering sketches into DXF CAD files. It uses:

- **Vision API (primary):** Google Gemini for geometry and text extraction (two-pass: walls from a line-only image, then text from the original).
- **Local fallback:** OpenCV Hough line detection when the API is unavailable or times out, so you still get LINE entities for DXF.
- **OCR:** PaddleOCR for handwriting and dimension text (optional, adds to Gemini results).
- **Validation & calibration:** OCR corrections, dimension ratio checks, and scale calibration from annotations.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your Google API key:

```
GOOGLE_API_KEY=your_key_here
```

Optional env vars:

- `GEMINI_MODEL` – Gemini model (default: `gemini-3-pro-preview`).
- `GEMINI_FALLBACK_MODEL` – Fallback model on timeout (default: `gemini-2.0-flash`).
- `GEMINI_REQUEST_TIMEOUT_SEC` – Request timeout in seconds (default: `120`; SDK default is 60 and can cause "Server disconnected" on large images).

Run the app:

```bash
streamlit run app.py
```

## Extraction Architecture

1. **Primary extraction:** Gemini two-pass (walls from preprocessed image, text from original). Timeout and retries with exponential backoff; on repeated failure, a faster fallback model is tried once.
2. **Local line fallback:** If both Gemini passes return no geometry, OpenCV Hough line detection runs on the line-only image and produces LINE entities so DXF output is still usable.
3. **Composite extractor:** The app uses a composite that tries the primary (Gemini) extractor first; on exception or empty result, it uses the local line extractor.
4. **PaddleOCR:** After geometry extraction, PaddleOCR can add dimension and label text (optional, for better handwriting).
5. **Calibration & validation:** Scale is inferred from dimension annotations or scale notation; validation applies OCR corrections and dimension consistency checks.

## External / Deep Learning Options

- **Vision APIs:** Gemini is the main backend. You could add optional fallbacks to OpenAI GPT-4V or Anthropic Claude as alternative cloud extractors behind the same `GeometryExtractor` interface.
- **Local OCR:** PaddleOCR is already integrated for text/dimensions.
- **Local geometry:** OpenCV line/contour fallback is built-in when Gemini fails.
- **Optional local DL (advanced/future):** Line-detection networks (e.g. [LCNN](https://github.com/zhou13/lcnn), [HAWP](https://github.com/cherubicxn/hawp)) or Segment Anything for regions could be added as extra extractors. Keep such models behind an optional dependency group or env flag so the core app runs with only Gemini + OpenCV + PaddleOCR.

For more detail, see [docs/architecture.md](docs/architecture.md).
