"""
test_delaunay.py

Run with:  python -m unittest test_delaunay -v

Checks the mathematical property that defines a Delaunay triangulation
(no input point lies inside the circumcircle of any triangle that does
not already contain it as a vertex), plus structural sanity checks: the
triangle count matches the standard 2n - 2 - h formula (n points, h on
the convex hull), degenerate inputs are rejected, and the Voronoi dual
produces one cell per point.
"""

import math
import random
import unittest

from delaunay import triangulate, _orientation
from voronoi import voronoi_cells


def convex_hull_size(points):
    """Gift-wrapping (Jarvis march) convex hull size, used only to
    validate triangle counts against the 2n - 2 - h formula."""
    pts = list(set(points))
    if len(pts) <= 2:
        return len(pts)

    start = min(pts, key=lambda p: (p[1], p[0]))
    hull = []
    current = start
    while True:
        hull.append(current)
        candidate = pts[0] if pts[0] != current else pts[1]
        for p in pts:
            if p == current or p == candidate:
                continue
            cross = _orientation(current, candidate, p)
            if cross < 0:
                candidate = p
            elif cross == 0:
                # p is farther along the same ray: prefer it so
                # collinear hull-edge points don't get skipped.
                d_candidate = (candidate[0] - current[0]) ** 2 + (candidate[1] - current[1]) ** 2
                d_p = (p[0] - current[0]) ** 2 + (p[1] - current[1]) ** 2
                if d_p > d_candidate:
                    candidate = p
        current = candidate
        if current == start:
            break
    return len(hull)


class TestDelaunayProperty(unittest.TestCase):
    def _assert_empty_circumcircle_property(self, points, triangles):
        # Checks the defining Delaunay property directly against the
        # circumcenter/radius (not the determinant predicate the
        # algorithm itself uses) so this is an independent check. Points
        # exactly on the circumcircle (a tie, e.g. four corners of a
        # square) are allowed -- that's a legitimate non-unique
        # triangulation, not a violation -- so this checks for points
        # *strictly* inside, with a small tolerance for that boundary.
        for t in triangles:
            center, radius = t.circumcircle()
            for p in points:
                if p in t.vertices():
                    continue
                dist = math.hypot(p[0] - center[0], p[1] - center[1])
                self.assertGreaterEqual(
                    dist, radius - 1e-6,
                    f"point {p} is strictly inside the circumcircle of triangle "
                    f"{t.vertices()} (violates the Delaunay condition)",
                )

    def test_square_plus_center(self):
        points = [(0, 0), (0, 10), (10, 10), (10, 0), (5, 5)]
        triangles = triangulate(points)
        self.assertEqual(len(triangles), 4)
        self._assert_empty_circumcircle_property(points, triangles)

    def test_random_point_sets_satisfy_delaunay_property(self):
        rng = random.Random(42)
        for _ in range(10):
            n = rng.randint(5, 40)
            points = list({
                (round(rng.uniform(0, 100), 3), round(rng.uniform(0, 100), 3))
                for _ in range(n)
            })
            if len(points) < 3:
                continue
            triangles = triangulate(points)
            self._assert_empty_circumcircle_property(points, triangles)

    def test_triangle_count_matches_euler_formula(self):
        rng = random.Random(7)
        points = list({
            (round(rng.uniform(0, 50), 3), round(rng.uniform(0, 50), 3))
            for _ in range(30)
        })
        triangles = triangulate(points)
        h = convex_hull_size(points)
        expected = 2 * len(points) - 2 - h
        self.assertEqual(len(triangles), expected)

    def test_rejects_too_few_points(self):
        with self.assertRaises(ValueError):
            triangulate([(0, 0), (1, 1)])

    def test_collinear_points_raise(self):
        with self.assertRaises(ValueError):
            triangulate([(0, 0), (1, 1), (2, 2)])

    def test_duplicate_points_are_ignored(self):
        points = [(0, 0), (0, 0), (0, 10), (10, 10), (10, 0)]
        triangles = triangulate(points)
        self._assert_empty_circumcircle_property(list(set(points)), triangles)


class TestVoronoiDual(unittest.TestCase):
    def test_cell_per_point(self):
        points = [(0, 0), (0, 10), (10, 10), (10, 0), (5, 5), (2, 8)]
        cells = voronoi_cells(points, margin=5.0)
        self.assertEqual(set(cells.keys()), set(points))
        for cell in cells.values():
            self.assertGreaterEqual(len(cell), 3)

    def test_too_few_points_gives_empty_diagram(self):
        self.assertEqual(voronoi_cells([(0, 0), (1, 1)]), {})

    def test_cells_partition_a_grid_reasonably(self):
        # A regular grid is a good stress case for the ghost-point
        # closure trick: every cell, including the four corners, should
        # come back as a closed polygon with at least 3 vertices.
        points = [(x, y) for x in range(0, 50, 10) for y in range(0, 50, 10)]
        cells = voronoi_cells(points, margin=5.0)
        self.assertEqual(len(cells), len(points))
        for cell in cells.values():
            self.assertGreaterEqual(len(cell), 3)


if __name__ == "__main__":
    unittest.main()
