"""Reference -> scene mapping: homography (or affine) fitted to point, single-axis and line constraints.

Constraints (job['align']):
  points : [{"ref": [x, y], "scene": [x, y], "w": 1, "label": ""}]        at least 3 (they seed the fit)
  y_only : [{"ref": [x, y], "scene_y": y, "w": 1, "label": ""}]            rows, shelves, repeated items
  x_only : [{"ref": [x, y], "scene_x": x, "w": 1, "label": ""}]
  lines  : [{"ref": [[x0, y0], [x1, y1]], "n": 12, "scene": [[x, y], ...], "w": 1, "label": ""}]
           the reference segment must land on the straight line through the scene points
"""
import numpy as np
from scipy.optimize import least_squares


def apply_H(H, pts):
    pts = np.atleast_2d(np.asarray(pts, float))
    ph = np.c_[pts, np.ones(len(pts))] @ H.T
    return ph[:, :2] / ph[:, 2:3]


def line_through(pts):
    """Total-least-squares line a*x + b*y + c = 0 with unit normal (a, b)."""
    P = np.asarray(pts, float)
    m = P.mean(0)
    n = np.linalg.svd(P - m)[2][-1]
    return np.array([n[0], n[1], -n @ m])


def segment(seg, n):
    (xa, ya), (xb, yb) = seg
    t = np.linspace(0, 1, n)
    return np.c_[xa + t * (xb - xa), ya + t * (yb - ya)]


def _H(p, model):
    if model == "affine":
        return np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [0, 0, 1.0]])
    return np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], 1.0]])


def fit(spec):
    """-> (H 3x3 mapping reference px to scene px, residual report in scene px)"""
    model = spec.get("model", "homography")
    P = spec.get("points", [])
    if len(P) < 3:
        raise ValueError("align.points needs at least 3 reference->scene correspondences")
    pr = np.array([q["ref"] for q in P], float)
    ps = np.array([q["scene"] for q in P], float)
    pw = np.array([q.get("w", 1.0) for q in P], float)[:, None]
    Yo, Xo = spec.get("y_only", []), spec.get("x_only", [])
    lines = [(segment(l["ref"], l.get("n", 12)), line_through(l["scene"]), float(l.get("w", 1.0)),
              l.get("label", f"line {i}")) for i, l in enumerate(spec.get("lines", []))]
    reg = float(spec.get("perspective_reg", 0.0))

    def res(p):
        H = _H(p, model)
        r = [((apply_H(H, pr) - ps) * pw).ravel()]
        if Yo:
            q = apply_H(H, [c["ref"] for c in Yo])
            r.append((q[:, 1] - np.array([c["scene_y"] for c in Yo])) * np.array([c.get("w", 1.0) for c in Yo]))
        if Xo:
            q = apply_H(H, [c["ref"] for c in Xo])
            r.append((q[:, 0] - np.array([c["scene_x"] for c in Xo])) * np.array([c.get("w", 1.0) for c in Xo]))
        for pts, (a, b, c), w, _ in lines:
            q = apply_H(H, pts)
            r.append(w * (a * q[:, 0] + b * q[:, 1] + c))
        if model != "affine" and reg:
            r.append(reg * np.asarray(p[6:8]))
        return np.concatenate(r)

    A = np.linalg.lstsq(np.c_[pr, np.ones(len(pr))], ps, rcond=None)[0].T
    p0 = A.ravel() if model == "affine" else np.r_[A.ravel(), 0.0, 0.0]
    sol = least_squares(res, p0, x_scale="jac")
    H = _H(sol.x, model)
    return H, _report(H, spec, pr, ps, lines)


def _report(H, spec, pr, ps, lines):
    rep = {"model": spec.get("model", "homography"), "points": [], "y_only": [], "x_only": [], "lines": []}
    for q, d in zip(spec["points"], apply_H(H, pr) - ps):
        rep["points"].append({"label": q.get("label", ""), "dx": round(float(d[0]), 2), "dy": round(float(d[1]), 2)})
    for key, coord, axis in (("y_only", "scene_y", 1), ("x_only", "scene_x", 0)):
        for c in spec.get(key, []):
            d = apply_H(H, [c["ref"]])[0, axis] - c[coord]
            rep[key].append({"label": c.get("label", ""), "d": round(float(d), 2)})
    for pts, (a, b, c), _, lab in lines:
        q = apply_H(H, pts)
        d = a * q[:, 0] + b * q[:, 1] + c
        rep["lines"].append({"label": lab, "max_px": round(float(np.abs(d).max()), 2), "mean_px": round(float(d.mean()), 2)})
    return rep
