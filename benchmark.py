"""
benchmark.py

Times triangulate() over a range of input sizes so the O(n^2) cost of
the brute-force "find bad triangles" scan (see delaunay.py) is visible
and measurable rather than just asserted in a docstring.

Run with:  python benchmark.py
"""

from __future__ import annotations

import random
import time

from delaunay import triangulate


def run(sizes=(50, 100, 200, 400, 800, 1600)) -> None:
    rng = random.Random(0)
    print(f"{'n':>6} {'triangles':>10} {'seconds':>10}")
    for n in sizes:
        points = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(n)]
        start = time.perf_counter()
        triangles = triangulate(points)
        elapsed = time.perf_counter() - start
        print(f"{n:>6} {len(triangles):>10} {elapsed:>10.4f}")


if __name__ == "__main__":
    run()
