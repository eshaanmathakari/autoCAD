"""
Wizard step renderers — one module per screen of the 5-step wizard UI.

Each module exports a ``render_<step>()`` function called by the main
``app.py`` controller based on ``st.session_state.current_step``.

Imports are lazy to avoid triggering heavy PoC fallback imports
(e.g. PaddleOCR) at package-import time.
"""


def render_upload():
    from sketch_to_cad.steps.upload import render_upload as _fn
    return _fn()


def render_matching(*args, **kwargs):
    from sketch_to_cad.steps.matching import render_matching as _fn
    return _fn(*args, **kwargs)


def render_verification(*args, **kwargs):
    from sketch_to_cad.steps.verification import render_verification as _fn
    return _fn(*args, **kwargs)


def render_cad_preview(*args, **kwargs):
    from sketch_to_cad.steps.cad_preview import render_cad_preview as _fn
    return _fn(*args, **kwargs)


def render_download(*args, **kwargs):
    from sketch_to_cad.steps.download import render_download as _fn
    return _fn(*args, **kwargs)


__all__ = [
    "render_upload",
    "render_matching",
    "render_verification",
    "render_cad_preview",
    "render_download",
]
