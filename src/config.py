"""
Configuration and constants for Sketch-to-DXF Converter.

Contains:
- API configuration
- Gemini extraction prompts
- DXF layer definitions
- Drawing defaults
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro-preview-05-06")

# File handling
SUPPORTED_IMAGE_TYPES = ["png", "jpg", "jpeg"]
MAX_FILE_SIZE_MB = 10
MAX_IMAGE_DIMENSION = 2048  # Max dimension for API efficiency

# Drawing defaults (A4 landscape in mm)
DEFAULT_DRAWING_WIDTH = 297
DEFAULT_DRAWING_HEIGHT = 210

# DXF Layer configuration with AutoCAD Color Index (ACI)
LAYER_COLORS = {
    "GEOMETRY": 7,       # White (on dark) / Black (on light)
    "TEXT": 3,           # Green
    "DIMENSIONS": 1,     # Red
    "CENTERLINES": 4,    # Cyan
    "HIDDEN": 8,         # Gray
    "CONSTRUCTION": 2,   # Yellow
}

# Gemini extraction prompt
EXTRACTION_PROMPT = """You are an expert CAD engineer analyzing a hand-drawn architectural floor plan.
Your task is to extract ALL geometric primitives and convert them to a structured JSON format suitable for CAD software.

## COORDINATE SYSTEM
- Use normalized coordinates from 0 to 1000
- (0, 0) = TOP-LEFT corner of the image
- (1000, 1000) = BOTTOM-RIGHT corner of the image
- X increases left to right
- Y increases top to bottom

## ENTITIES TO DETECT AND EXTRACT

1. **LINES**: Wall segments, boundaries, and structural elements
   - Extract start and end points for each wall segment
   - Identify exterior walls vs interior walls
   - Include door openings as gaps in wall lines

2. **RECTANGLES**: Rooms, windows, furniture outlines
   - Identify room boundaries as rectangles where possible
   - Extract window openings
   - Furniture can be simplified to rectangles

3. **ARCS**: Door swings, curved walls, rounded corners
   - Door swings typically show 90-degree arcs
   - Identify center, radius, start and end angles

4. **POLYLINES**: Irregular room shapes, stairs, complex outlines
   - For L-shaped or irregular rooms, use polylines
   - Stairs can be represented as a series of connected lines
   - Mark as closed if the shape forms a complete boundary

5. **TEXT**: Room labels, dimensions, annotations
   - Extract room names (BEDROOM, KITCHEN, BATHROOM, etc.)
   - Extract any dimension values shown
   - Include floor plan title if visible

6. **CIRCLES**: Columns, circular features, trees/landscaping
   - Structural columns
   - Circular furniture or fixtures

## LAYER ASSIGNMENT
- **GEOMETRY**: Walls, room boundaries, structural elements
- **TEXT**: Room labels, titles, annotations
- **DIMENSIONS**: Any measurement annotations
- **CONSTRUCTION**: Grid lines, reference lines

## FLOOR PLAN SPECIFIC GUIDELINES
1. WALLS are the most important - extract every wall segment as a LINE
2. ROOMS should be identifiable by their boundary lines
3. DOORS appear as gaps in walls with arc swings
4. WINDOWS appear as parallel lines or gaps in exterior walls
5. Extract ALL readable text labels for rooms
6. Ignore decorative elements like furniture details, plants, textures

## OUTPUT
Return a valid JSON object with:
- metadata: title, detected_units, confidence_score
- entities: list of all geometric entities (prioritize walls and room labels)
- warnings: any issues or unclear areas

Focus on extracting the structural layout - walls, doors, windows, and room labels are the priority."""


# Alternative prompt for simpler sketches (fewer instructions)
EXTRACTION_PROMPT_SIMPLE = """Analyze this engineering sketch and extract geometric primitives.

Coordinate system: 0-1000 normalized (0,0 = top-left, 1000,1000 = bottom-right)

Extract:
- Lines (start/end points)
- Circles (center, radius)
- Arcs (center, radius, start/end angles in degrees)
- Rectangles (top-left, width, height)
- Text (position, content)
- Dimensions (start, end, value)

Assign layers: GEOMETRY, TEXT, DIMENSIONS, CENTERLINES

Return JSON matching the provided schema."""


# Two-pass extraction prompts for noisy hand-drawn plans

EXTRACTION_PROMPT_WALLS = """You are analyzing a preprocessed floor plan image where colors and shading have been removed.
Focus ONLY on extracting structural geometry - walls and room boundaries.

## COORDINATE SYSTEM
- Normalized coordinates 0-1000
- (0, 0) = TOP-LEFT, (1000, 1000) = BOTTOM-RIGHT

## WHAT TO EXTRACT

1. **LINES** - Wall segments
   - Every BLACK LINE is likely a wall
   - Extract start point and end point for each wall segment
   - Walls form the boundaries of rooms
   - Include both exterior walls (building outline) and interior walls (room dividers)

2. **RECTANGLES** - Room boundaries
   - If a room is rectangular, extract it as a rectangle
   - top_left corner, width, height

3. **POLYLINES** - Complex room shapes
   - L-shaped rooms, irregular spaces
   - List vertices in order, mark as closed

4. **ARCS** - Door swings
   - Quarter-circle arcs indicate door positions
   - Extract center, radius, start_angle, end_angle

## WHAT TO IGNORE
- DO NOT extract text or labels (handled separately)
- IGNORE any remaining shading or texture artifacts
- IGNORE furniture outlines
- IGNORE decorative elements

## LAYER ASSIGNMENT
- All geometry goes to layer "GEOMETRY"

## OUTPUT
Return JSON with:
- metadata: confidence_score (0-1)
- entities: list of LINE, RECTANGLE, POLYLINE, ARC entities only
- warnings: any unclear areas

Extract EVERY wall segment you can identify."""


EXTRACTION_PROMPT_TEXT = """You are analyzing a floor plan image to extract ONLY text labels and annotations.
Do NOT extract any geometry - focus only on readable text.

## COORDINATE SYSTEM
- Normalized coordinates 0-1000
- (0, 0) = TOP-LEFT, (1000, 1000) = BOTTOM-RIGHT

## WHAT TO EXTRACT

1. **TEXT** - Room labels and annotations
   - Room names: BEDROOM, KITCHEN, BATHROOM, LIVING ROOM, DINING, OFFICE, etc.
   - Floor plan title (e.g., "GROUND FLOOR PLAN", "FIRST FLOOR PLAN")
   - Any other readable text labels
   - Note the position (center of text)
   - Estimate text height
   - If text is rotated, note the rotation angle (0-360)

2. **DIMENSIONS** - Measurement annotations (if visible)
   - Start and end points of what's being measured
   - The dimension value and unit

## WHAT TO IGNORE
- DO NOT extract walls, lines, or shapes
- IGNORE shading, colors, textures
- IGNORE illegible or unclear text

## LAYER ASSIGNMENT
- Text labels go to layer "TEXT"
- Dimensions go to layer "DIMENSIONS"

## OUTPUT
Return JSON with:
- metadata: title (if found), confidence_score
- entities: list of TEXT and DIMENSION entities only
- warnings: any text that was unclear

Extract ALL readable text labels."""
