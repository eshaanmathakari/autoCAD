"""
Vendor JS dependencies for the DXF viewer.

Loads Three.js and dxf-parser from local vendor files so they can be
inlined into the viewer HTML — avoids CDN requests that are blocked by
Streamlit Cloud's Content Security Policy.
"""

import os

_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "vendor")


def _read_vendor(filename: str) -> str:
    path = os.path.join(_VENDOR_DIR, filename)
    with open(path, "r") as f:
        return f.read()


def get_three_js() -> str:
    """Return minified Three.js source code."""
    return _read_vendor("three.min.js")


def get_dxf_parser_js() -> str:
    """Return minified dxf-parser source code."""
    return _read_vendor("dxf-parser.min.js")
