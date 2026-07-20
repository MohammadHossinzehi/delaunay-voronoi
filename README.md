# delaunay-voronoi

A from-scratch Delaunay triangulation and Voronoi diagram library in pure Python -- zero dependencies, no NumPy, no SciPy. Given a set of 2D points, it computes the Delaunay triangulation via the incremental Bowyer-Watson algorithm, derives the dual Voronoi diagram, and renders both as SVG.

## Why this exists

Delaunay triangulation and Voronoi diagrams show up everywhere: mesh generation, procedural terrain, nearest-neighbor queries, interpolation, motion planning. Most people reach for `scipy.spatial.Delaunay` and never see what's inside it. This is what's inside it -- and, more usefully, this project is a write-up of what actually goes wrong when you implement it yourself.

Naive Bowyer-Watson implementations have a well-known failure mode: they crack. A point that should be fully interior to the mesh ends up with a stray edge running out to nowhere, because the algorithm's "find every triangle whose circumcircle contains this point" step is more numerically delicate than it looks. This implementation hit that exact bug during development (see `delaunay.py`'s module docstring for the full post-mortem) and fixes it properly rather than papering over it with a bigger epsilon:

- Bad triangles are found by **flood-filling** outward from the triangle that geometrically contains the new point, following shared edges -- not by independently scanning every triangle, which can strand an orphaned triangle inside the cavity.
- The circumcircle test is the standard **determinant-based incircle predicate**, which needs no division and is far better conditioned than computing a circumcenter and comparing distances -- especially for the long, thin triangles this algorithm constantly creates near its bounding "super-triangle."
- When even that determinant is too close to zero to trust, it falls back to **exact rational arithmetic** via `fractions.Fraction`, so the sign is always correct instead of a guess.

The result was verified against 800 randomized point sets (sizes 4-70) with zero mesh cracks -- see `test_delaunay.py`.

## What's in here

- `delaunay.py` -- the triangulation: `Triangle`, `triangulate(points)`.
- `voronoi.py` -- derives the Voronoi diagram from a triangulation via the ghost-point trick (see its module docstring): `voronoi_cells(points, margin)`.
- `svg_render.py` -- zero-dependency SVG renderer for both.
- `cli.py` -- command-line entry point.
- `benchmark.py` -- timing across input sizes, to make the documented O(n^2) complexity visible rather than just asserted.
- `test_delaunay.py` -- unit tests: the empty-circumcircle Delaunay property, the 2n-2-h triangle-count formula, degenerate-input handling, and the Voronoi dual.

## Running it

No dependencies to install -- everything here is the Python standard library.

```bash
python cli.py --points 60 --seed 1 --out demo.svg --voronoi
```

This generates 60 random points, triangulates them, derives the Voronoi diagram, and writes both to `demo.svg`, which you can open directly in a browser. Options:

```
--points N       number of random points (default 40)
--seed N         random seed, for reproducible output
--out PATH       output SVG path (default output.svg)
--voronoi        also render the Voronoi diagram
--width, --height    size of the point-sampling area (default 100x100)
```

Or use the library directly:

```python
from delaunay import triangulate
from voronoi import voronoi_cells

points = [(0, 0), (4, 0), (2, 3), (2, 1)]
triangles = triangulate(points)          # list[Triangle]
cells = voronoi_cells(points, margin=1)  # {point: [polygon vertices]}
```

Run the tests with:

```bash
python -m unittest test_delaunay -v
```

Run the benchmark with:

```bash
python benchmark.py
```

## Design decisions and known limitations

**Complexity is O(n^2) on purpose.** There's no spatial index (no quadtree, no kd-tree), so finding the triangle a new point lands in, and finding every triangle whose circumcircle contains it, are both linear scans in the worst case. A production triangulator would add one; this one trades that for an implementation short enough to read end to end and verify against the math it's implementing. `benchmark.py` shows exactly how that scales -- roughly quadratic, as expected, comfortable up to a couple thousand points.

**Voronoi cells are closed with a ghost-point trick, not exact clipping.** A point on the convex hull has a mathematically unbounded Voronoi cell. Rather than implement general polygon clipping against a bounding box, `voronoi.py` triangulates the input together with eight distant points arranged in a ring around it, so every real point becomes topologically interior and gets a genuinely closed cell -- just a very large one near the hull -- before each vertex is clamped into the visible bounding box. This is a deliberate simplification for clean rendering; it is not exact half-plane clipping, and cells right at the box edge can look slightly flattened. This is called out in `voronoi.py`'s docstring so it doesn't get mistaken for a bug later.

**Testing strategy.** Rather than trust a single check, `test_delaunay.py` verifies the triangulation three independent ways: the empty-circumcircle property that mathematically defines a Delaunay triangulation, the 2n-2-h triangle-count identity (n points, h on the convex hull) as a structural sanity check, and a stress test across a grid layout for the Voronoi dual. Degenerate inputs (too few points, all-collinear points) are tested explicitly rather than left to fail unpredictably.
