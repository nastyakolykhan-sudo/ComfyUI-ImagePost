"""Exact even-odd polygon rasterisation at pixel centres.

Index coordinates: pixel (row j, col i) is centred at (x=i, y=j). PIL's ImageDraw.polygon is not used because
it draws the outline too, pushing right and bottom edges out by one pixel.
"""
import numpy as np


def fill_polygon(h, w, pts):
    """Boolean h x w mask of pixel centres inside the polygon (vertices in index coordinates)."""
    P = np.asarray(pts, float)
    xa, ya = P[:, 0], P[:, 1]
    xb, yb = np.roll(xa, -1), np.roll(ya, -1)
    out = np.zeros((h, w), bool)
    for j in range(max(int(np.ceil(ya.min())), 0), min(int(np.floor(ya.max())), h - 1) + 1):
        cross = ((ya <= j) & (yb > j)) | ((yb <= j) & (ya > j))
        if not cross.any():
            continue
        xs = np.sort(xa[cross] + (j - ya[cross]) * (xb[cross] - xa[cross]) / (yb[cross] - ya[cross]))
        for x0, x1 in zip(xs[0::2], xs[1::2]):
            i0, i1 = max(int(np.ceil(x0)), 0), min(int(np.ceil(x1)), w)
            if i1 > i0:
                out[j, i0:i1] = True
    return out
