"""Reference -> scene mapping: homography (or affine), or a cylinder for labels on bottles, fitted to point,
single-axis, line and curve constraints.

Constraints (job['align']):
  points : [{"ref": [x, y], "scene": [x, y], "w": 1, "label": ""}]        at least 3 (they seed the fit)
  y_only : [{"ref": [x, y], "scene_y": y, "w": 1, "label": ""}]            rows, shelves, repeated items
  x_only : [{"ref": [x, y], "scene_x": x, "w": 1, "label": ""}]
  lines  : [{"ref": [[x0, y0], [x1, y1]], "n": 12, "scene": [[x, y], ...], "w": 1, "label": ""}]
           the reference segment must land on the straight line through the scene points; with "curve": true
           on the smooth curve through them (a label's top edge on a bottle seen from above or below); with
           "one_sided": true it must reach the line but may run past it (away from the product's middle)
Models: "homography" (default), "affine", "cylinder" (align.cylinder {"axis_x", "radius"} in reference px,
optional align.perspective, and align.limbs [{"side": "left|right", "scene": [[x, y], ...], "rows": [v0, v1]}]: the
scene's silhouette of the bottle on that side, which a label wrapping past it ends at).
"""
import numpy as np
from scipy.optimize import least_squares


class Cylinder:
    """Reference px -> scene px for print on a cylinder: a bottle's label, a jar, a can.

    The packshot is read as an orthographic front view of a cylinder whose axis is vertical at x = axis_x, radius r
    (reference px): reference (u, v) is the surface point P = (u - axis_x, v, sqrt(r^2 - (u - axis_x)^2)), z toward
    the packshot's camera. The scene sees P through the camera p = (A P + a) / (1 + c.P): affine when c = 0,
    projective with align.perspective. Lines across the axis (a label's top and bottom, its text) become the arcs
    a bottle seen from above or below shows, which a homography can't bend."""

    def __init__(self, axis_x, r, A, a, c=(0.0, 0.0, 0.0), wrap=0.0):
        self.axis_x, self.r, self.wrap = float(axis_x), float(r), float(wrap)
        self.A = np.asarray(A, float).reshape(2, 3)
        self.a = np.asarray(a, float).reshape(2)
        self.c = np.asarray(c, float).reshape(3)

    def surface(self, pts):
        P = np.atleast_2d(np.asarray(pts, float))
        X = P[:, 0] - self.axis_x
        return np.c_[X, P[:, 1], np.sqrt(np.clip(self.r ** 2 - X ** 2, 0.0, None))]

    def forward(self, pts):
        S = self.surface(pts)
        return (S @ self.A.T + self.a) / (1.0 + S @ self.c)[:, None]

    def inverse(self, x, y):
        """Scene px -> reference px (u, v) and a validity mask: False where the ray misses the cylinder or meets
        its back half, which the packshot doesn't show. With wrap (degrees) the surface up to that far past the
        packshot's limb stays valid, mirrored from the packshot's side of the limb (plain label paper near a
        limb the scene sees further round than the packshot)."""
        shape = np.shape(x)
        x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
        A, a, c, r = self.A, self.a, self.c, self.r
        r1 = A[0][None, :] - x[:, None] * c[None, :]
        r2 = A[1][None, :] - y[:, None] * c[None, :]
        b1, b2 = x - a[0], y - a[1]
        d = np.cross(r1, r2)                                     # the viewing ray through the pixel
        g11, g12, g22 = (r1 * r1).sum(1), (r1 * r2).sum(1), (r2 * r2).sum(1)
        det = g11 * g22 - g12 * g12
        P0 = r1 * ((g22 * b1 - g12 * b2) / det)[:, None] + r2 * ((g11 * b2 - g12 * b1) / det)[:, None]
        qa = d[:, 0] ** 2 + d[:, 2] ** 2
        qb = 2.0 * (P0[:, 0] * d[:, 0] + P0[:, 2] * d[:, 2])
        qc = P0[:, 0] ** 2 + P0[:, 2] ** 2 - r * r
        disc = qb * qb - 4.0 * qa * qc
        hit = disc >= 0.0
        sq = np.sqrt(np.clip(disc, 0.0, None))
        lo, hi = (-qb - sq) / (2.0 * qa), (-qb + sq) / (2.0 * qa)
        cam = self._camera()
        if cam is None:                                          # affine: d points at the camera
            lam = hi
        else:                                                    # projective: the hit nearer the camera centre
            lc = ((cam[None, :] - P0) * d).sum(1) / np.maximum((d * d).sum(1), 1e-30)
            lam = np.where(np.abs(hi - lc) < np.abs(lo - lc), hi, lo)
        P = P0 + lam[:, None] * d
        ok = hit & (P[:, 2] >= -r * max(np.sin(np.radians(self.wrap)), 1e-6))
        u = self.axis_x + np.clip(P[:, 0], -r, r)
        return u.reshape(shape), P[:, 1].reshape(shape), ok.reshape(shape)

    def _camera(self):
        if np.abs(self.c).max() < 1e-12:
            return None
        M = np.vstack([self.A, self.c])
        if np.linalg.cond(M) > 1e12:
            return None
        return np.linalg.solve(M, -np.r_[self.a, 1.0])

    def limb_angle(self, side):
        """Angle (radians, 0 = facing the packshot's camera, + toward reference +x) of the scene's silhouette on
        `side` (right = the packshot's right half): where the view ray grazes the cylinder."""
        cam = self._camera()
        sgn = 1.0 if side == "right" else -1.0
        if cam is None:
            t = np.cross(self.A[0], self.A[1])
            th = np.arctan2(-t[2], t[0])                         # N(th) . t = 0, two opposite solutions
            return th if np.sin(th) * sgn > 0 else th + np.pi
        rho, phi = np.hypot(cam[0], cam[2]), np.arctan2(cam[0], cam[2])
        dth = np.arccos(np.clip(self.r / max(rho, self.r), -1.0, 1.0))
        return phi + sgn * dth

    def limb_points(self, side, v):
        """Scene points of the silhouette on `side` at reference rows v."""
        th = self.limb_angle(side)
        v = np.atleast_1d(np.asarray(v, float))
        S = np.c_[np.full(len(v), self.r * np.sin(th)), v, np.full(len(v), self.r * np.cos(th))]
        return (S @ self.A.T + self.a) / (1.0 + S @ self.c)[:, None]

    def to_json(self):
        return {"axis_x": self.axis_x, "radius": self.r, "wrap": self.wrap, "A": self.A.round(8).tolist(),
                "a": self.a.round(6).tolist(), "c": self.c.tolist()}


class Residual:
    """A smooth scene-space correction on top of a fitted mapping (homography or cylinder): it moves the mapped
    product onto the generated edges where the rigid model can't follow them (a bottle drawn with a bent
    silhouette, a label edge that wavers). Gaussian RBF through the edge residuals (sigma px, ridge lam), so it
    fades out away from the edges and the print inside stays as fitted. Evaluated on a grid, interpolated."""

    def __init__(self, base, ctrl, disp, sigma, lam=0.5):
        self.base, self.sigma = base, float(sigma)
        C, d = np.asarray(ctrl, float), np.asarray(disp, float)
        K = np.exp(-((C[:, None, :] - C[None, :, :]) ** 2).sum(-1) / (2 * self.sigma ** 2))
        W = np.linalg.solve(K + lam * np.eye(len(C)), d)
        lo, hi = C.min(0) - 3 * self.sigma, C.max(0) + 3 * self.sigma
        self.step = max(self.sigma / 8.0, 2.0)
        gx, gy = np.arange(lo[0], hi[0] + self.step, self.step), np.arange(lo[1], hi[1] + self.step, self.step)
        G = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
        D = np.zeros((len(G), 2))
        for i in range(0, len(G), 20000):
            g = G[i:i + 20000]
            D[i:i + 20000] = np.exp(-((g[:, None, :] - C[None]) ** 2).sum(-1) / (2 * self.sigma ** 2)) @ W
        self.origin, self.shape = lo, (len(gy), len(gx))
        self.grid = D.reshape(len(gy), len(gx), 2)
        self.max_shift = float(np.linalg.norm(self.grid, axis=-1).max())

    def __getattr__(self, name):                        # a cylinder's axis_x, r, limb_points ... pass through
        return getattr(self.__dict__["base"], name)

    def shift(self, x, y):
        from scipy.ndimage import map_coordinates
        gi, gj = (np.asarray(y, float) - self.origin[1]) / self.step, (np.asarray(x, float) - self.origin[0]) / self.step
        return [map_coordinates(self.grid[..., k], [gi, gj], order=1, mode="constant", cval=0.0) for k in (0, 1)]

    def forward(self, pts):
        q = apply_H(self.base, pts)
        dx, dy = self.shift(q[:, 0], q[:, 1])
        return q + np.c_[dx, dy]

    def inverse(self, x, y):
        bx, by = np.asarray(x, float), np.asarray(y, float)
        for _ in range(3):                               # y + D(y) = x by fixed point (D is small and smooth)
            dx, dy = self.shift(bx, by)
            bx, by = x - dx, y - dy
        if hasattr(self.base, "inverse"):
            return self.base.inverse(bx, by)
        q = np.c_[bx.ravel(), by.ravel(), np.ones(bx.size)] @ np.linalg.inv(self.base).T
        return (q[:, 0] / q[:, 2]).reshape(bx.shape), (q[:, 1] / q[:, 2]).reshape(bx.shape), None

    def to_json(self):
        out = self.base.to_json() if hasattr(self.base, "to_json") else {"H": np.asarray(self.base).tolist()}
        return dict(out, residual={"sigma": self.sigma, "max_shift_px": round(self.max_shift, 2)})


def apply_H(H, pts):
    if hasattr(H, "forward"):
        return H.forward(pts)
    pts = np.atleast_2d(np.asarray(pts, float))
    ph = np.c_[pts, np.ones(len(pts))] @ H.T
    return ph[:, :2] / ph[:, 2:3]


def as_matrix(H, at):
    """A 3 x 3 matrix for code that needs one: H itself, or the mapping's local affine at reference point `at`."""
    return _local_affine(H, at) if hasattr(H, "forward") else H


def _local_affine(H, p, e=0.5):
    p = np.asarray(p, float)
    q = H.forward([p, p + (e, 0.0), p + (0.0, e)])
    J = np.c_[(q[1] - q[0]) / e, (q[2] - q[0]) / e]
    M = np.eye(3)
    M[:2, :2], M[:2, 2] = J, q[0] - J @ p
    return M


def map_polygon(H, poly, step=2.0):
    """A closed reference polygon in scene px. A cylinder bends straight sides, so they are resampled first."""
    P = np.asarray(poly, float)
    if not hasattr(H, "forward"):
        return apply_H(H, P)
    Q = np.vstack([P, P[:1]])
    out = []
    for a, b in zip(Q[:-1], Q[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        out.append(a[None] + np.linspace(0.0, 1.0, n, endpoint=False)[:, None] * (b - a)[None])
    return H.forward(np.vstack(out))


def line_through(pts):
    """Total-least-squares line a*x + b*y + c = 0 with unit normal (a, b)."""
    P = np.asarray(pts, float)
    m = P.mean(0)
    n = np.linalg.svd(P - m)[2][-1]
    return np.array([n[0], n[1], -n @ m])


class Curve:
    """Scene points measured along a generated edge that may bend. The residual is each point's signed distance to
    the reference segment as the mapping draws it (sampled densely), so the measured stretch may be shorter than the
    segment (a corner out of reach, a part behind an occluder), never longer: the segment ends where it ends, so
    the fit can't slide the product along its own edges."""

    def __init__(self, pts):
        self.S = np.asarray(pts, float)

    def dist(self, q):
        """q: the reference segment's samples mapped into the scene, in order -> distance per scene point."""
        a, b = q[:-1], q[1:]
        ab = b - a
        L2 = np.maximum((ab * ab).sum(1), 1e-12)
        t = ((self.S[:, None, :] - a[None]) * ab[None]).sum(2) / L2[None]
        t = np.clip(t, 0.0, 1.0)
        e = self.S[:, None, :] - (a[None] + t[..., None] * ab[None])
        dd = np.sqrt((e * e).sum(2))
        k = dd.argmin(1)
        i = np.arange(len(self.S))
        side = np.sign(ab[k, 0] * e[i, k, 1] - ab[k, 1] * e[i, k, 0])
        return np.where(side == 0, 1.0, side) * dd[i, k]

    def nearest(self, q):
        """For each scene point, the nearest point on the mapped segment q."""
        a, b = q[:-1], q[1:]
        ab = b - a
        t = np.clip(((self.S[:, None, :] - a[None]) * ab[None]).sum(2) / np.maximum((ab * ab).sum(1), 1e-12)[None], 0.0, 1.0)
        f = a[None] + t[..., None] * ab[None]
        k = np.sqrt(((self.S[:, None, :] - f) ** 2).sum(2)).argmin(1)
        return f[np.arange(len(self.S)), k]


def _one_sided(d, sd, pull=0.15):
    """Falling short of the edge (inside the product) costs fully; running past it only a weak pull, so the edge
    stays put unless another constraint needs it to move out."""
    return np.where(d * sd > 0, d * sd, pull * d * sd)


def _dist(L, q):
    if isinstance(L, Curve):
        return L.dist(q)
    a, b, c = L
    return a * q[:, 0] + b * q[:, 1] + c


def segment(seg, n):
    """n points along a reference segment [[x0, y0], [x1, y1]], or along a measured polyline of 3+ points."""
    if len(seg) > 2:
        P = np.asarray(seg, float)
        cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        s = np.linspace(0.0, cum[-1], n)
        return np.c_[np.interp(s, cum, P[:, 0]), np.interp(s, cum, P[:, 1])]
    (xa, ya), (xb, yb) = seg
    t = np.linspace(0, 1, n)
    return np.c_[xa + t * (xb - xa), ya + t * (yb - ya)]


def _H(p, model):
    if model == "affine":
        return np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [0, 0, 1.0]])
    return np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], 1.0]])


def fit(spec):
    """-> (H mapping reference px to scene px: a 3x3 matrix, a Cylinder, or either under a Residual correction;
    residual report in scene px)"""
    H, rep = _fit_base(spec)
    if spec.get("residual") not in (None, False):
        H, rep = _add_residual(H, spec, rep)
    return H, rep


def _add_residual(H, spec, rep):
    """align.residual {"sigma": px | "auto", "lam": 0.1, "outlier": 2.5}: a smooth correction through the edge
    residuals (lines, curves, limbs) after the rigid fit; a snapped point whose residual differs from the median of
    its 6 neighbours along the same edge by more than outlier px is a stray and is left out."""
    cfg = spec["residual"] if isinstance(spec["residual"], dict) else {}
    S_all, D_all, src = [], [], []
    cen = np.vstack([np.asarray(l["scene"], float) for l in spec.get("lines", [])]).mean(0) if spec.get("lines") else None
    for i, l in enumerate(spec.get("lines", [])):
        C = Curve(l["scene"])
        D = C.S - C.nearest(apply_H(H, segment(l["ref"], max(int(l.get("n", 12)), 60))))
        if l.get("one_sided"):           # past the generated edge is allowed: correct only where the model falls short
            D = np.where((((C.S - cen) * D).sum(1) > 0)[:, None], D, 0.0)
        S_all.append(C.S)
        D_all.append(D)
        src += [l.get("label", f"line {i}")] * len(C.S)
    if isinstance(H, Cylinder):
        for l in spec.get("limbs", []):
            C = Curve(l["scene"])
            v = np.linspace(*l.get("rows", (0, 1)), 80)
            S_all.append(C.S)
            D_all.append(C.S - C.nearest(H.limb_points(l["side"], v)))
            src += [l.get("label", "limb")] * len(C.S)
    # a stray snapped point disagrees with its neighbours along the same edge; a real bend doesn't
    keeps = []
    for D in D_all:
        nb = np.array([np.median(D[max(0, j - 3):j + 4], axis=0) for j in range(len(D))])
        keeps.append(np.linalg.norm(D - nb, axis=1) <= float(cfg.get("outlier", 2.5)))
    S_all, D_all, keep = np.vstack(S_all), np.vstack(D_all), np.concatenate(keeps)
    sigma = cfg.get("sigma", "auto")
    if sigma == "auto":
        sigma = 0.2 * float(min(np.ptp(S_all[:, 0]), np.ptp(S_all[:, 1])))
    R = Residual(H, S_all[keep] - D_all[keep], D_all[keep], sigma, float(cfg.get("lam", 0.1)))
    after, k0 = [], 0
    for i, l in enumerate(spec.get("lines", [])):                 # on the points kept (strays don't count)
        d = Curve(l["scene"]).dist(apply_H(R, segment(l["ref"], max(int(l.get("n", 12)), 60))))
        if l.get("one_sided"):
            a, b, c = line_through(l["scene"])
            q = apply_H(R, segment(l["ref"], max(int(l.get("n", 12)), 60)))
            sd = float(np.sign(a * cen[0] + b * cen[1] + c)) or 1.0
            d = np.maximum((a * q[:, 0] + b * q[:, 1] + c) * sd, 0.0)
            d = np.full(len(Curve(l["scene"]).S), float(d.max()))
        kk = keep[k0:k0 + len(d)]
        k0 += len(d)
        after.append({"label": l.get("label", f"line {i}"), "max_px": round(float(np.abs(d[kk]).max()) if kk.any() else 0.0, 2)})
    if isinstance(H, Cylinder):                       # the silhouette after the correction, on the points kept
        for l in spec.get("limbs", []):
            C = Curve(l["scene"])
            q = H.limb_points(l["side"], np.linspace(*l.get("rows", (0, 1)), 80))
            dx, dy = R.shift(q[:, 0], q[:, 1])
            d = C.dist(q + np.c_[dx, dy])
            kk = keep[k0:k0 + len(d)]
            k0 += len(d)
            after.append({"label": l.get("label", f"{l['side']} limb"),
                          "max_px": round(float(np.abs(d[kk]).max()) if kk.any() else 0.0, 2)})
    rep["residual"] = {"sigma": round(float(sigma), 1), "max_shift_px": round(R.max_shift, 2),
                       "controls": int(keep.sum()), "dropped": sorted({src[j] for j in np.nonzero(~keep)[0]}),
                       "dropped_n": int((~keep).sum()), "lines_after": after}
    return R, rep


def _fit_base(spec):
    model = spec.get("model", "homography")
    P = spec.get("points", [])
    if len(P) < 3:
        raise ValueError("align.points needs at least 3 reference->scene correspondences")
    pr = np.array([q["ref"] for q in P], float)
    ps = np.array([q["scene"] for q in P], float)
    pw = np.array([q.get("w", 1.0) for q in P], float)[:, None]
    Yo, Xo = spec.get("y_only", []), spec.get("x_only", [])
    # one_sided lines (a label's edge the real one must cover, but may run past onto glass): the interior side is
    # the one holding the centroid of every scene point measured
    allS = [np.asarray(q["scene"], float)[None] for q in P] + [np.asarray(l["scene"], float) for l in spec.get("lines", [])]
    cen = np.vstack(allS).mean(0)

    def side(l):
        if not l.get("one_sided"):
            return None
        if l.get("curve"):
            raise ValueError(f"align.lines {l.get('label', '')!r}: one_sided works on straight lines only")
        a, b, c = line_through(l["scene"])
        return float(np.sign(a * cen[0] + b * cen[1] + c)) or 1.0

    lines = [(segment(l["ref"], l.get("n", 40 if l.get("curve") else 12)),
              Curve(l["scene"]) if l.get("curve") else line_through(l["scene"]),
              float(l.get("w", 1.0)), l.get("label", f"line {i}"), side(l)) for i, l in enumerate(spec.get("lines", []))]
    reg = float(spec.get("perspective_reg", 0.0))
    if model == "cylinder":
        return _fit_cylinder(spec, pr, ps, pw, Yo, Xo, lines, reg)

    def res(p):
        H = _H(p, model)
        r = [((apply_H(H, pr) - ps) * pw).ravel()]
        if Yo:
            q = apply_H(H, [c["ref"] for c in Yo])
            r.append((q[:, 1] - np.array([c["scene_y"] for c in Yo])) * np.array([c.get("w", 1.0) for c in Yo]))
        if Xo:
            q = apply_H(H, [c["ref"] for c in Xo])
            r.append((q[:, 0] - np.array([c["scene_x"] for c in Xo])) * np.array([c.get("w", 1.0) for c in Xo]))
        for pts, L, w, _, sd in lines:
            q = apply_H(H, pts)
            r.append(w * (_dist(L, q) if sd is None else _one_sided(_dist(L, q), sd)))
        if model != "affine" and reg:
            r.append(reg * np.asarray(p[6:8]))
        return np.concatenate(r)

    A = np.linalg.lstsq(np.c_[pr, np.ones(len(pr))], ps, rcond=None)[0].T
    p0 = A.ravel() if model == "affine" else np.r_[A.ravel(), 0.0, 0.0]
    sol = least_squares(res, p0, x_scale="jac")
    H = _H(sol.x, model)
    return H, _report(H, spec, pr, ps, lines)


def _fit_cylinder(spec, pr, ps, pw, Yo, Xo, lines, reg):
    cy = spec.get("cylinder") or {}
    if "axis_x" not in cy or "radius" not in cy:
        raise ValueError("align.model 'cylinder' needs align.cylinder {\"axis_x\", \"radius\"}: the bottle's axis and "
                         "radius in reference px (midway between its left and right edges, half the distance)")
    ax, rad = float(cy["axis_x"]), float(cy["radius"])
    persp = bool(spec.get("perspective", False))
    # limbs: the scene's silhouette (a label wrapping past it ends there), as rows of the reference
    limbs = [(l["side"], Curve(l["scene"]), float(l.get("w", 1.0)), np.linspace(*l.get("rows", (pr[:, 1].min(), pr[:, 1].max())), 40),
              l.get("label", f"{l['side']} limb")) for l in spec.get("limbs", [])]
    lmin = cy.get("limb_min")
    lmin = None if lmin is None else float(lmin)
    lw = float(cy.get("limb_w", 3.0))

    def make(p):
        return Cylinder(ax, rad, p[0:6], p[6:8], p[8:11] if persp else (0.0, 0.0, 0.0), cy.get("wrap", 0.0))

    def res(p):
        H = make(p)
        r = [((H.forward(pr) - ps) * pw).ravel()]
        if Yo:
            q = H.forward([c["ref"] for c in Yo])
            r.append((q[:, 1] - np.array([c["scene_y"] for c in Yo])) * np.array([c.get("w", 1.0) for c in Yo]))
        if Xo:
            q = H.forward([c["ref"] for c in Xo])
            r.append((q[:, 0] - np.array([c["scene_x"] for c in Xo])) * np.array([c.get("w", 1.0) for c in Xo]))
        for pts, L, w, _, sd in lines:
            d = _dist(L, H.forward(pts))
            r.append(w * (d if sd is None else _one_sided(d, sd)))
        for side, L, w, v, _ in limbs:
            r.append(w * L.dist(H.limb_points(side, v)))
        if lmin is not None:             # print may not be turned into the silhouette: |limb angle| >= limb_min
            for side in {l[0] for l in limbs} or {"right"}:
                deg = abs(float(np.degrees(H.limb_angle(side))))
                r.append(np.array([lw * max(0.0, lmin - deg)]))
        if persp and reg:
            r.append(reg * np.asarray(p[8:11]))
        return np.concatenate(r)

    # seed: a flat affine fit of the weighted points (no bend), then everything at once
    S = Cylinder(ax, rad, np.zeros(6), np.zeros(2)).surface(pr)
    sw = np.sqrt(np.clip(pw[:, 0], 0.0, None))[:, None]
    K = np.linalg.lstsq(np.c_[S[:, :2], np.ones(len(S))] * sw, ps * sw, rcond=None)[0]
    p0 = np.r_[K[0, 0], K[1, 0], 0.0, K[0, 1], K[1, 1], 0.0, K[2]]
    if persp:
        p0 = np.r_[p0, 0.0, 0.0, 0.0]
    sol = least_squares(res, p0, x_scale="jac")
    H = make(sol.x)
    rep = _report(H, spec, pr, ps, lines)
    for side, L, _, v, lab in limbs:
        d = L.dist(H.limb_points(side, v))
        rep["lines"].append({"label": lab, "max_px": round(float(np.abs(d).max()), 2), "mean_px": round(float(d.mean()), 2)})
        rep["cylinder"][f"{side}_limb_deg"] = round(float(np.degrees(H.limb_angle(side))), 1)
    if lmin is not None:
        rep["cylinder"]["limb_min"] = lmin
    return H, rep


def _report(H, spec, pr, ps, lines):
    rep = {"model": spec.get("model", "homography"), "points": [], "y_only": [], "x_only": [], "lines": []}
    for q, d in zip(spec["points"], apply_H(H, pr) - ps):
        rep["points"].append({"label": q.get("label", ""), "dx": round(float(d[0]), 2), "dy": round(float(d[1]), 2)})
    for key, coord, axis in (("y_only", "scene_y", 1), ("x_only", "scene_x", 0)):
        for c in spec.get(key, []):
            d = apply_H(H, [c["ref"]])[0, axis] - c[coord]
            rep[key].append({"label": c.get("label", ""), "d": round(float(d), 2)})
    for pts, L, _, lab, sd in lines:
        q = apply_H(H, pts)
        d = _dist(L, q)
        e = {"label": lab, "max_px": round(float(np.abs(d).max()), 2), "mean_px": round(float(d.mean()), 2)}
        if sd is not None:               # one-sided: only falling short of the generated edge counts
            e.update(one_sided=True, max_px=round(float(np.maximum(d * sd, 0.0).max()), 2),
                     past_px=round(float(np.maximum(-d * sd, 0.0).max()), 2))
        rep["lines"].append(e)
    if isinstance(H, Cylinder):
        rep["cylinder"] = H.to_json()
    return rep
