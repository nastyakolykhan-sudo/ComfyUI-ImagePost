"""Render the reference into the scene's work box, supersampled, in linear light."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def warp(ref_lin, H, roi, refmask, ss=4, prefilter=0.6):
    """-> (rgb linear h x w x 3, product coverage h x w, (u, v) reference coords per scene pixel)"""
    x0, y0, x1, y1 = roi
    w, h = x1 - x0, y1 - y0
    xs = x0 + (np.arange(w * ss) + 0.5) / ss - 0.5
    ys = y0 + (np.arange(h * ss) + 0.5) / ss - 0.5
    X, Y = np.meshgrid(xs, ys)
    q = np.c_[X.ravel(), Y.ravel(), np.ones(X.size)] @ np.linalg.inv(H).T
    u = (q[:, 0] / q[:, 2]).reshape(X.shape)
    v = (q[:, 1] / q[:, 2]).reshape(X.shape)
    del q, X, Y
    us, vs = refmask.clamp(u, v)
    rgb = np.stack([map_coordinates(gaussian_filter(ref_lin[..., c], prefilter), [vs, us], order=3, mode="reflect")
                    for c in range(3)], -1)
    a = refmask.sample(u, v)

    def down(z):
        return z.reshape(h, ss, w, ss, *z.shape[2:]).mean(axis=(1, 3))

    return np.clip(down(rgb), 0, None), np.clip(down(a), 0, 1), (down(u), down(v))
