"""
svg_render.py

Zero-dependency SVG renderer for a Delaunay triangulation and/or its dual
Voronoi diagram. Writes plain SVG text directly -- no Pillow, no
matplotlib, no browser required to view the result.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from delaunay import Point, Triangle

_STYLE = """
  <style>
    .tri { fill: none; stroke: #2b6cb0; stroke-width: 0.6; }
    .voronoi { fill: none; stroke: #c05621; stroke-width: 0.8; }
    .pt { fill: #1a202c; }
  </style>
"""


def _bounds(points: List[Point], pad: float) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return (min_x - pad, min_y - pad, max_x - min_x + 2 * pad, max_y - min_y + 2 * pad)


def render_svg(
    points: List[Point],
    triangles: Optional[List[Triangle]] = None,
    voronoi: Optional[Dict[Point, List[Point]]] = None,
    size: int = 800,
    pad: float = 5.0,
) -> str:
    min_x, min_y, w, h = _bounds(points, pad)
    scale = size / max(w, h)

    def tx(p: Point) -> Tuple[float, float]:
        return ((p[0] - min_x) * scale, size - (p[1] - min_y) * scale)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" '
        f'height="{size}" viewBox="0 0 {size} {size}">',
        _STYLE,
    ]

    if voronoi:
        for cell in voronoi.values():
            if len(cell) < 3:
                continue
            pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in (tx(p) for p in cell))
            parts.append(f'<polygon class="voronoi" points="{pts}" />')

    if triangles:
        for t in triangles:
            (ax, ay), (bx, by), (cx, cy) = tx(t.a), tx(t.b), tx(t.c)
            parts.append(
                f'<polygon class="tri" points="{ax:.2f},{ay:.2f} '
                f'{bx:.2f},{by:.2f} {cx:.2f},{cy:.2f}" />'
            )

    for p in points:
        x, y = tx(p)
        parts.append(f'<circle class="pt" cx="{x:.2f}" cy="{y:.2f}" r="2.5" />')

    parts.append("</svg>")
    return "\n".join(parts)


def save_svg(path: str, *args, **kwargs) -> None:
    svg = render_svg(*args, **kwargs)
    with open(path, "w") as f:
        f.write(svg)
