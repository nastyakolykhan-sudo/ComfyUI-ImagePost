"""Scene-space masks: polygons and half-planes rasterised with supersampling (integer coords = pixel centres).

job['occluders']: [{"label": "", "side": "left|right|above|below", "polyline": [[x, y], ...]}]
                  the occluder occupies that side of the polyline (ends are extended along their segments)
              or  [{"label": "", "polygon": [[x, y], ...]}]
                  optional "soften": px (Gaussian sigma) for out-of-focus occluder edges
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from .raster import fill_polygon


def poly_cover(roi, pts, ss=8):
    """Area coverage (0..1) of a polygon given in scene px, over the work box (ss x ss samples per pixel)."""
    x0, y0, x1, y1 = roi
    w, h = x1 - x0, y1 - y0
    sub = fill_polygon(h * ss, w * ss, [((x - x0 + 0.5) * ss - 0.5, (y - y0 + 0.5) * ss - 0.5) for x, y in pts])
    return sub.reshape(h, ss, w, ss).mean(axis=(1, 3))


def _extended(P, axis, lo, hi):
    P = sorted((tuple(map(float, p)) for p in P), key=lambda p: p[axis])
    other = 1 - axis

    def at(a, b, t):
        o = a[other] if b[axis] == a[axis] else a[other] + (t - a[axis]) * (b[other] - a[other]) / (b[axis] - a[axis])
        return (t, o) if axis == 0 else (o, t)

    if P[0][axis] > lo:
        P.insert(0, at(P[0], P[1] if len(P) > 1 else P[0], lo))
    if P[-1][axis] < hi:
        P.append(at(P[-2] if len(P) > 1 else P[-1], P[-1], hi))
    return P


def halfplane(roi, polyline, side, margin=64):
    x0, y0, x1, y1 = roi
    if side in ("left", "right"):
        P = _extended(polyline, 1, y0 - margin, y1 + margin)
        xb = x0 - margin if side == "left" else x1 + margin
        return P + [(xb, P[-1][1]), (xb, P[0][1])]
    if side in ("above", "below"):
        P = _extended(polyline, 0, x0 - margin, x1 + margin)
        yb = y0 - margin if side == "above" else y1 + margin
        return P + [(P[-1][0], yb), (P[0][0], yb)]
    raise ValueError(f"occluder side must be left/right/above/below, got {side!r}")


def occluder_polygon(roi, occ):
    return [tuple(map(float, p)) for p in occ["polygon"]] if "polygon" in occ else halfplane(roi, occ["polyline"], occ["side"])


def visibility(roi, occluders, ss=8):
    """1 where the product can show, 0 under occluders (anti-aliased)."""
    x0, y0, x1, y1 = roi
    vis = np.ones((y1 - y0, x1 - x0))
    for occ in occluders:
        cover = poly_cover(roi, occluder_polygon(roi, occ), ss)
        if occ.get("soften"):  # out-of-focus foreground: blur the occluder's edge (px, Gaussian sigma)
            cover = gaussian_filter(cover, float(occ["soften"]))
        vis *= 1.0 - cover
    return vis
