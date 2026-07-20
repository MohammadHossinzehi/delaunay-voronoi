"""
cli.py

Command line entry point:

    python cli.py --points 60 --seed 1 --out demo.svg --voronoi

Generates `points` random 2D points, computes their Delaunay
triangulation (and optionally its dual Voronoi diagram), and writes the
result to an SVG file that can be opened directly in a browser.
"""

from __future__ import annotations

import argparse
import random

from delaunay import triangulate
from svg_render import save_svg
from voronoi import voronoi_cells


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delaunay triangulation / Voronoi diagram generator"
    )
    parser.add_argument("--points", type=int, default=40, help="number of random points")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--out", type=str, default="output.svg", help="output SVG path")
    parser.add_argument("--voronoi", action="store_true", help="also render the Voronoi diagram")
    parser.add_argument("--width", type=float, default=100.0, help="width of the point sampling area")
    parser.add_argument("--height", type=float, default=100.0, help="height of the point sampling area")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    points = [
        (rng.uniform(0, args.width), rng.uniform(0, args.height))
        for _ in range(args.points)
    ]

    triangles = triangulate(points)
    cells = voronoi_cells(points, margin=5.0) if args.voronoi else None

    save_svg(args.out, points, triangles=triangles, voronoi=cells)
    print(f"{len(points)} points -> {len(triangles)} triangles")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
