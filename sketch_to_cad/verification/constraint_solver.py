"""
Constraint solver for pool dimensions.

Uses scipy SLSQP to find optimal scale factors satisfying
dimension constraints when simple proportional scaling is insufficient.
"""

import math
from typing import Optional

import numpy as np

from sketch_to_cad.cad_engine.pool_generator import PoolEdge


def solve_constraints(
    edges: list[PoolEdge],
    target_length: float,
    target_width: float,
    fixed_indices: Optional[list[int]] = None,
) -> list[PoolEdge]:
    """
    Use SLSQP to find vertex positions satisfying dimension constraints.

    For non-rectangular pools where independent X/Y scaling isn't sufficient.
    Minimizes total vertex displacement while satisfying:
    - Overall bbox matches target length/width
    - Fixed edges (if any) maintain their length

    Args:
        edges: Original pool edges
        target_length: Target bbox length in inches
        target_width: Target bbox width in inches
        fixed_indices: Edge indices whose lengths should be preserved

    Returns:
        Adjusted edges with optimized vertex positions
    """
    from scipy.optimize import minimize

    if not edges:
        return edges

    fixed_indices = fixed_indices or []

    # Collect unique vertices
    vertices = []
    for e in edges:
        vertices.append((e.x1, e.y1))
        vertices.append((e.x2, e.y2))

    # Deduplicate (round to avoid float comparison issues)
    unique = list({(round(x, 4), round(y, 4)) for x, y in vertices})
    unique.sort()
    vert_map = {v: i for i, v in enumerate(unique)}

    # Map edges to vertex indices
    edge_verts = []
    for e in edges:
        k1 = (round(e.x1, 4), round(e.y1, 4))
        k2 = (round(e.x2, 4), round(e.y2, 4))
        edge_verts.append((vert_map[k1], vert_map[k2]))

    n = len(unique)
    x0 = np.array([c for v in unique for c in v], dtype=float)

    # Compute original bbox
    xs = [v[0] for v in unique]
    ys = [v[1] for v in unique]
    orig_w = max(xs) - min(xs)
    orig_h = max(ys) - min(ys)

    if orig_w == 0 or orig_h == 0:
        return edges

    # Scale initial guess proportionally
    sx = target_length / orig_w
    sy = target_width / orig_h
    minx, miny = min(xs), min(ys)
    x_init = np.copy(x0)
    for i in range(n):
        x_init[2 * i] = (unique[i][0] - minx) * sx
        x_init[2 * i + 1] = (unique[i][1] - miny) * sy

    def objective(x):
        """Minimize total displacement from proportional scaling."""
        return np.sum((x - x_init) ** 2)

    constraints = []

    # Bbox constraints
    def bbox_length(x):
        xvals = x[::2]
        return (xvals.max() - xvals.min()) - target_length

    def bbox_width(x):
        yvals = x[1::2]
        return (yvals.max() - yvals.min()) - target_width

    constraints.append({"type": "eq", "fun": bbox_length})
    constraints.append({"type": "eq", "fun": bbox_width})

    # Fixed edge length constraints
    for idx in fixed_indices:
        if idx >= len(edge_verts):
            continue
        vi, vj = edge_verts[idx]
        orig_len = edges[idx].length

        def fixed_len(x, _vi=vi, _vj=vj, _ol=orig_len):
            dx = x[2 * _vi] - x[2 * _vj]
            dy = x[2 * _vi + 1] - x[2 * _vj + 1]
            return math.sqrt(dx ** 2 + dy ** 2) - _ol

        constraints.append({"type": "eq", "fun": fixed_len})

    result = minimize(
        objective, x_init,
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-8},
    )

    if not result.success:
        # Fall back to proportional scaling
        opt = x_init
    else:
        opt = result.x

    # Reconstruct edges
    new_edges = []
    for i, e in enumerate(edges):
        vi, vj = edge_verts[i]
        new_edges.append(PoolEdge(
            x1=opt[2 * vi],
            y1=opt[2 * vi + 1],
            x2=opt[2 * vj],
            y2=opt[2 * vj + 1],
            label=e.label,
        ))

    return new_edges
