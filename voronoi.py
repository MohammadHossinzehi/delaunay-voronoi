"""
voronoi.py

Derives the Voronoi diagram from a Delaunay triangulation. The Voronoi
diagram is the topological dual of the Delaunay triangulation: the
circumcenter of every Delaunay triangle is a Voronoi vertex, and the
Voronoi cell of an input point is the polygon formed by the circumcenters
of every triangle incident to that point, connected in angular order
around it.

Design decision -- closing unbounded cells: a point on the convex hull
of the input has a mathematically unbounded Voronoi cell (an open
region, not a polygon). Rather than implement general polygon clipping
against a bounding box, this module triangulates the input together
with eight distant "ghost" points arranged in a ring far outside the
input's bounding box. Every real point then has a full 360-degree fan of
incident triangles (it is topologically interior to the padded point
set), so its cell is already a closed polygon -- just a very large one
near the hull. Each cell vertex is then independently clamped into the
visible bounding box (expanded by `margin`). This is a deliberate
simplification: for a reasonably generous margin it produces clean,
closed cells for rendering, but it is not a substitute for exact
half-plane clipping, and cells right at the box edge can look slightly
flattened. Ghost points themselves never appear as keys in the result.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple

from delaunay import Point, Triangle, triangulate


def _clip_to_box(pt: Point, box: Tuple[float, float, float, float]) -> Point:
    min_x, min_y, max_x, max_y = box
    x = min(max(pt[0], min_x), max_x)
    y = min(max(pt[1], min_y), max_y)
    return (x, y)


def _angle(origin: Point, pt: Point) -> float:
    return math.atan2(pt[1] - origin[1], pt[0] - origin[0])


def _ghost_ring(min_x: float, min_y: float, max_x: float, max_y: float,
                 padding_factor: float = 10.0, count: int = 8) -> List[Point]:
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    diag = math.hypot(max_x - min_x, max_y - min_y) or 1.0
    radius = diag * padding_factor
    return [
        (cx + radius * math.cos(2 * math.pi * i / count),
         cy + radius * math.sin(2 * math.pi * i / count))
        for i in range(count)
    ]


def voronoi_cells(points: List[Point], margin: float = 0.0) -> Dict[Point, List[Point]]:
    """Returns {input_point: [ordered polygon vertices]} for each distinct
    point in `points`. Internally pads the point set with distant ghost
    points so every real point gets a closed cell -- see the module
    docstring. Returns {} if fewer than 3 distinct points are given."""
    real_points = list(dict.fromkeys(points))
    if len(real_points) < 3:
        return {}

    xs = [p[0] for p in real_points]
    ys = [p[1] for p in real_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    ghosts = _ghost_ring(min_x, min_y, max_x, max_y)
    triangles = triangulate(real_points + ghosts)

    box = (min_x - margin, min_y - margin, max_x + margin, max_y + margin)
    real_set = set(real_points)

    incident: Dict[Point, List[Point]] = defaultdict(list)
    for t in triangles:
        center, _ = t.circumcircle()
        for v in t.vertices():
            if v in real_set:
                incident[v].append(center)

    cells: Dict[Point, List[Point]] = {}
    for p, centers in incident.items():
        ordered = sorted(centers, key=lambda c: _angle(p, c))
        cells[p] = [_clip_to_box(c, box) for c in ordered]
    return cells
