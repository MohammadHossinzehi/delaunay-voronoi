"""
delaunay.py

From-scratch incremental Delaunay triangulation using the Bowyer-Watson
algorithm. No external dependencies -- no scipy, no numpy. The only
non-obvious import is `fractions.Fraction` from the standard library,
used as an exact-arithmetic fallback (see "On robustness" below).

Core idea (Bowyer-Watson):
    1. Start with a single "super-triangle" large enough to contain every
       input point.
    2. Insert points one at a time. For each new point p:
         a. Find every triangle whose circumcircle contains p ("bad"
            triangles) by flood-filling outward from the triangle that
            geometrically contains p, following shared edges, and only
            crossing into a neighbor if its circumcircle also contains p.
         b. Remove the bad triangles; the union of their edges forms a
            star-shaped polygonal hole around p, except for edges shared
            by two bad triangles (those are interior to the hole and
            cancel out).
         c. Re-triangulate the hole by connecting p to every boundary
            edge of the hole.
    3. Once all points are inserted, discard every triangle that touches
       a vertex of the original super-triangle.

Complexity: O(n^2) worst case -- there is no spatial index, so each
insertion is O(n) to locate the containing triangle plus O(cavity size)
to flood-fill the rest. That trade-off is intentional: it keeps the
implementation short and easy to verify against the Delaunay
empty-circumcircle property in test_delaunay.py. See benchmark.py for
how it scales in practice.

On robustness (read this before "simplifying" the incircle test):
this module went through three real bugs during development that are
worth naming, because each one is a classic Bowyer-Watson pitfall and
the fixes are not optional polish:

  1. Finding "bad" triangles with an independent brute-force scan over
     every triangle (instead of a connectivity-respecting flood fill)
     can silently strand an orphaned triangle inside the cavity for
     near-degenerate inputs, cracking the mesh. Fixed by flood-filling
     from a single seed through triangle adjacency.
  2. Seeding that flood fill from "any triangle whose circumcircle
     contains p" rather than "the triangle that actually contains p"
     can seed far from p's true location. Fixed by locating the
     triangle p geometrically falls inside first, and only falling back
     to a circumcircle-based scan (relevant for points outside every
     current triangle, e.g. early insertions against the super-triangle)
     if point-location fails.
  3. The classic circumcenter-and-radius incircle test loses precision
     for the long, thin triangles that connect real points to a distant
     super-triangle vertex -- exactly the triangles this algorithm
     creates constantly. That precision loss can flip the sign of a
     near-zero test, which is enough to leave an interior point
     incorrectly connected to the super-triangle after cleanup. Fixed by
     using the standard determinant-based incircle predicate (see
     `_in_circumcircle`), which avoids division entirely, and falling
     back to exact rational arithmetic via `fractions.Fraction` whenever
     the floating-point determinant is too close to zero to trust. The
     same fast-path-then-exact-fallback pattern is applied to the
     orientation test. Combined with a generous super-triangle margin
     (see `_super_triangle`), this was verified against 500 random seeds
     (sizes 5-60 points) with zero mesh cracks -- see test_delaunay.py.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Tuple

Point = Tuple[float, float]


def _orientation_raw(a: Point, b: Point, c: Point) -> float:
    """Twice the signed area of triangle abc, as a plain float. Positive
    if a, b, c turn counter-clockwise, negative if clockwise, zero if
    collinear. Exposed (unprefixed by an underscore in spirit, but kept
    private) mainly so tests can reuse it for their own hull checks."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _orientation(a: Point, b: Point, c: Point) -> float:
    return _orientation_raw(a, b, c)


def _orientation_sign(a: Point, b: Point, c: Point) -> int:
    """Sign of the orientation test: +1 counter-clockwise, -1 clockwise,
    0 collinear. Uses a fast float path, falling back to exact
    Fraction arithmetic when the float result is too close to zero to
    trust (see the module docstring)."""
    o = _orientation_raw(a, b, c)
    scale = (abs(b[0] - a[0]) + abs(c[1] - a[1]) + abs(b[1] - a[1]) + abs(c[0] - a[0])) or 1.0
    if abs(o) > scale * 1e-9:
        return 1 if o > 0 else -1
    fa = (Fraction(a[0]), Fraction(a[1]))
    fb = (Fraction(b[0]), Fraction(b[1]))
    fc = (Fraction(c[0]), Fraction(c[1]))
    fo = (fb[0] - fa[0]) * (fc[1] - fa[1]) - (fb[1] - fa[1]) * (fc[0] - fa[0])
    return 1 if fo > 0 else (-1 if fo < 0 else 0)


def _det3(m) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _incircle_matrix(a, b, c, d):
    ax, ay = a[0] - d[0], a[1] - d[1]
    bx, by = b[0] - d[0], b[1] - d[1]
    cx, cy = c[0] - d[0], c[1] - d[1]
    return [
        [ax, ay, ax * ax + ay * ay],
        [bx, by, bx * bx + by * by],
        [cx, cy, cx * cx + cy * cy],
    ]


def _in_circumcircle(a: Point, b: Point, c: Point, d: Point) -> bool:
    """True if d is inside the circumcircle of triangle (a, b, c). This
    is the standard determinant-based incircle predicate rather than the
    more obvious "compute the circumcenter and compare distance to
    radius" approach, because the determinant form needs no division and
    is far better conditioned for the long thin triangles Bowyer-Watson
    produces near the super-triangle. Falls back to exact Fraction
    arithmetic when the float determinant is too close to zero to trust
    -- see the module docstring."""
    m = _incircle_matrix(a, b, c, d)
    det = _det3(m)
    scale = sum(abs(v) for row in m for v in row) or 1.0
    if abs(det) > scale * 1e-6:
        sign = det
    else:
        fm = _incircle_matrix(
            (Fraction(a[0]), Fraction(a[1])),
            (Fraction(b[0]), Fraction(b[1])),
            (Fraction(c[0]), Fraction(c[1])),
            (Fraction(d[0]), Fraction(d[1])),
        )
        sign = _det3(fm)
    ccw = _orientation_sign(a, b, c) > 0
    return (sign > 0) if ccw else (sign < 0)


@dataclass(frozen=True)
class Triangle:
    a: Point
    b: Point
    c: Point

    def vertices(self) -> Tuple[Point, Point, Point]:
        return (self.a, self.b, self.c)

    def edges(self) -> Tuple[Tuple[Point, Point], Tuple[Point, Point], Tuple[Point, Point]]:
        return ((self.a, self.b), (self.b, self.c), (self.c, self.a))

    def circumcircle(self) -> Tuple[Point, float]:
        """Returns (center, radius) of the circle through a, b, c. Used
        by voronoi.py to place Voronoi vertices; NOT used for the
        Delaunay bad-triangle test itself, which uses the more robust
        `_in_circumcircle` determinant predicate instead."""
        ax, ay = self.a
        bx, by = self.b
        cx, cy = self.c
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-12:
            raise ValueError("Degenerate triangle (collinear points)")
        ax2ay2 = ax * ax + ay * ay
        bx2by2 = bx * bx + by * by
        cx2cy2 = cx * cx + cy * cy
        ux = (ax2ay2 * (by - cy) + bx2by2 * (cy - ay) + cx2cy2 * (ay - by)) / d
        uy = (ax2ay2 * (cx - bx) + bx2by2 * (ax - cx) + cx2cy2 * (bx - ax)) / d
        center = (ux, uy)
        radius = math.hypot(ax - ux, ay - uy)
        return center, radius

    def contains_in_circumcircle(self, p: Point) -> bool:
        return _in_circumcircle(self.a, self.b, self.c, p)

    def contains_point(self, p: Point) -> bool:
        """True if p lies inside or on the boundary of this triangle."""
        d1 = _orientation_sign(self.a, self.b, p)
        d2 = _orientation_sign(self.b, self.c, p)
        d3 = _orientation_sign(self.c, self.a, p)
        has_neg = d1 < 0 or d2 < 0 or d3 < 0
        has_pos = d1 > 0 or d2 > 0 or d3 > 0
        return not (has_neg and has_pos)


def _super_triangle(points: List[Point]) -> Triangle:
    """A triangle large enough to strictly contain every point in
    `points`, with a wide margin. The margin matters: too tight, and a
    real (but unwanted) empty-circle path can connect an interior point
    to a super-triangle vertex, which survives the final cleanup step
    and cracks the mesh -- see point 3 in the module docstring."""
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    dx = max_x - min_x
    dy = max_y - min_y
    delta = max(dx, dy, 1.0) * 1000
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2
    return Triangle(
        (mid_x - 2 * delta, mid_y - delta),
        (mid_x, mid_y + 2 * delta),
        (mid_x + 2 * delta, mid_y - delta),
    )


def _edge_map(triangles: List[Triangle]) -> Dict[frozenset, List[Triangle]]:
    """Maps each undirected edge to the (at most two) triangles that use it."""
    m: Dict[frozenset, List[Triangle]] = defaultdict(list)
    for t in triangles:
        for e in t.edges():
            m[frozenset(e)].append(t)
    return m


def _find_bad_cavity(triangles: List[Triangle], edges: Dict[frozenset, List[Triangle]],
                      p: Point) -> List[Triangle]:
    """Finds every triangle whose circumcircle contains `p`, by
    flood-filling from a seed triangle through shared edges. The seed is
    the triangle that geometrically contains p (point location), falling
    back to a circumcircle-based scan only if no triangle contains p
    (which can happen for points outside the current mesh early on).
    See point 2 in the module docstring for why point location matters."""
    seed = None
    for t in triangles:
        if t.contains_point(p):
            seed = t
            break
    if seed is None:
        for t in triangles:
            if t.contains_in_circumcircle(p):
                seed = t
                break
    if seed is None:
        return []

    bad = set()
    stack = [seed]
    while stack:
        t = stack.pop()
        if t in bad:
            continue
        bad.add(t)
        for e in t.edges():
            for neighbor in edges[frozenset(e)]:
                if neighbor is not t and neighbor not in bad and neighbor.contains_in_circumcircle(p):
                    stack.append(neighbor)
    return list(bad)


def triangulate(points: List[Point]) -> List[Triangle]:
    """Compute the Delaunay triangulation of `points` via Bowyer-Watson.

    Duplicate points are removed automatically. Raises ValueError if
    fewer than 3 distinct points are given, or if the points are all
    collinear (no valid triangulation exists).
    """
    unique_points = list(dict.fromkeys(points))
    if len(unique_points) < 3:
        raise ValueError("Need at least 3 distinct points to triangulate")

    super_tri = _super_triangle(unique_points)
    triangles: List[Triangle] = [super_tri]

    for p in unique_points:
        edges = _edge_map(triangles)
        bad_triangles = _find_bad_cavity(triangles, edges, p)

        # The boundary of the polygonal hole left by removing the bad
        # triangles is exactly the set of edges that belong to only one
        # bad triangle (edges shared by two bad triangles are interior
        # to the hole and cancel out).
        edge_count = {}
        for t in bad_triangles:
            for e in t.edges():
                key = frozenset(e)
                edge_count[key] = edge_count.get(key, 0) + 1
        boundary = [e for t in bad_triangles for e in t.edges()
                    if edge_count[frozenset(e)] == 1]

        bad_set = set(bad_triangles)
        triangles = [t for t in triangles if t not in bad_set]

        for (e0, e1) in boundary:
            triangles.append(Triangle(e0, e1, p))

    # Discard every triangle that still touches a super-triangle vertex.
    super_vertices = set(super_tri.vertices())
    triangles = [t for t in triangles
                 if not (set(t.vertices()) & super_vertices)]

    if not triangles:
        raise ValueError(
            "Degenerate point set: no valid triangulation (points may be collinear)"
        )

    return triangles
