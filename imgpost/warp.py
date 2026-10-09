"""Render the reference into the scene's work box, supersampled, in linear light. H: a 3x3 matrix or a
geometry.Cylinder (anything with inverse(x, y) -> (u, v, valid))."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def warp(ref_lin, H, roi, refmask, ss=4, prefilter=0.6):
    """-> (rgb linear h x w x 3, product coverage h x w, (u, v) reference coords per scene pixel)"""
    x0, y0, x1, y1 = roi
    w, h = x1 - x0, y1 - y0
    xs = x0 + (np.arange(w * ss) + 0.5) / ss - 0.5
    ys = y0 + (np.arange(h * ss) + 0.5) / ss - 0.5
    X, Y = np.meshgrid(xs, ys)
    ok = None
    if hasattr(H, "inverse"):                    # a cylinder: rays that miss it, or meet its back half, show nothing
        u, v, ok = H.inverse(X, Y)
        del X, Y
    else:
        q = np.c_[X.ravel(), Y.ravel(), np.ones(X.size)] @ np.linalg.inv(H).T
        u = (q[:, 0] / q[:, 2]).reshape(X.shape)
        v = (q[:, 1] / q[:, 2]).reshape(X.shape)
        del q, X, Y
    us, vs = refmask.clamp(u, v)
    if ok is not None and hasattr(H, "axis_x"):  # colour from just inside the packshot's limbs: past them is backdrop
        m = 1.5
        us = np.clip(us, H.axis_x - H.r + m, H.axis_x + H.r - m)
    rgb = np.stack([map_coordinates(gaussian_filter(ref_lin[..., c], prefilter), [vs, us], order=3, mode="reflect")
                    for c in range(3)], -1)
    a = refmask.sample(u, v)
    if ok is not None:
        a = a * ok

    def down(z):
        return z.reshape(h, ss, w, ss, *z.shape[2:]).mean(axis=(1, 3))

    return np.clip(down(rgb), 0, None), np.clip(down(a), 0, 1), (down(u), down(v))


def bleed_cover(H, roi, refmask, ss=2):
    """Where the product's paper reaches in the work box when its outline bleeds past its real edge (0..1)."""
    x0, y0, x1, y1 = roi
    w, h = x1 - x0, y1 - y0
    xs = x0 + (np.arange(w * ss) + 0.5) / ss - 0.5
    ys = y0 + (np.arange(h * ss) + 0.5) / ss - 0.5
    X, Y = np.meshgrid(xs, ys)
    if hasattr(H, "inverse"):
        u, v, ok = H.inverse(X, Y)
    else:
        q = np.c_[X.ravel(), Y.ravel(), np.ones(X.size)] @ np.linalg.inv(H).T
        u, v, ok = (q[:, 0] / q[:, 2]).reshape(X.shape), (q[:, 1] / q[:, 2]).reshape(X.shape), None
    a = refmask.sample_bleed(u, v)
    if ok is not None:
        a = a * ok
    return np.clip(a.reshape(h, ss, w, ss).mean(axis=(1, 3)), 0, 1)
