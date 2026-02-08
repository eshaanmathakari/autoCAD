"""
Pool industry constants used across all modules.

Layer definitions, building code compliance values, verification thresholds,
and plausible dimension ranges for OCR validation.
"""

# 7 DXF layers (AutoCAD Color Index)
POOL_LAYERS = {
    "POOL_OUTLINE": 7,      # White - main perimeter
    "STAIRS": 5,            # Blue - stair treads/risers
    "DIMENSIONS": 1,        # Red - dimension entities
    "EQUIPMENT": 8,         # Gray - skimmers, returns, drains
    "COPING": 6,            # Magenta - coping edge
    "SAFETY_LEDGE": 4,      # Cyan - safety/tanning ledge
    "LINER_SEAMS": 3,       # Green - seam lines
}

# Layers that use dashed linetypes
DASHED_LAYERS = {"EQUIPMENT", "LINER_SEAMS"}

# Pool code compliance (ANSI/APSP/ICC-5, ISPSC)
MAX_RISER_HEIGHT_RESIDENTIAL = 12   # inches
MAX_RISER_HEIGHT_COMMERCIAL = 9     # inches
MIN_TREAD_DEPTH = 10                # inches
MIN_TREAD_AREA = 240                # square inches
MAX_COPING_OVERHANG = 1.5           # inches
MAX_FLOOR_SLOPE_SHALLOW = 1 / 10   # for depth <= 5'
MAX_FLOOR_SLOPE_DEEP = 1 / 3       # for depth > 5'
DEPTH_TRANSITION_POINT = 60         # 5' in inches
MIN_CORNER_RADIUS = 2               # inches
SAFETY_LEDGE_WIDTH_RANGE = (4, 6)   # inches

# Verification thresholds
DEFAULT_TOLERANCE_INCHES = 0.5
BBOX_TOLERANCE_INCHES = 1.0
MAX_SANDBOX_ITERATIONS = 3
SSIM_PASS_THRESHOLD = 0.85
HU_MOMENT_TOLERANCE = 0.15

# Plausible pool dimension ranges (for OCR validation)
POOL_LENGTH_RANGE = (120, 600)      # 10'-50'
POOL_WIDTH_RANGE = (60, 300)        # 5'-25'
STAIR_RISER_RANGE = (6, 18)         # 6"-18"
POOL_DEPTH_RANGE = (24, 132)        # 2'-11'
