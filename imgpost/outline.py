"""The real product's outline in reference pixels, rasterised once and sampled at warped coordinates.

job['reference_outline'] types:
  rounded_rect : x0, y0, x1, y1, r_top, r_bottom
  profile      : x0, x1, y1, r_bottom, top_left [[x, y], ...] (left side top -> start of flat top), mirror_x
  polygon      : pts [[x, y], ...]
  key_white    : threshold (non-white pixels of a packshot on white, holes filled, largest component)
  alpha        : the reference's own transparency
Optional 'extend': [{"side": "left|right|top|bottom", "to": coord, "range": [lo, hi], "clamp": coord}]
  adds margin past the outline (from `to` to `clamp`, within `range` on the other axis) and samples colour
  there from the reference's own edge pixels (coordinate clamped at `clamp`). Use it when the real edge
  must run on behind an occluder.
"""
import numpy as np
from scipy.ndimage import binary_fill_holes, label, map_coordinates

from .raster import fill_polygon


def _arc(cx, cy, r, a0, a1, n=12):
    t = np.radians(np.linspace(a0, a1, n))
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in t]


def rounded_rect(x0, y0, x1, y1, r_top=0.0, r_bottom=0.0):
    pts = []
    pts += _arc(x0 + r_top, y0 + r_top, r_top, 180, 270) if r_top else [(x0, y0)]
    pts += _arc(x1 - r_top, y0 + r_top, r_top, 270, 360) if r_top else [(x1, y0)]
    pts += _arc(x1 - r_bottom, y1 - r_bottom, r_bottom, 0, 90) if r_bottom else [(x1, y1)]
    pts += _arc(x0 + r_bottom, y1 - r_bottom, r_bottom, 90, 180) if r_bottom else [(x0, y1)]
    return pts


def profile_shape(x0, x1, y1, top_left, r_bottom=0.0, mirror_x=None):
    """Symmetric outline: straight sides x0/x1, bottom y1, top from a measured left-half profile."""
    m = (x0 + x1) / 2.0 if mirror_x is None else float(mirror_x)
    left = [tuple(map(float, p)) for p in top_left]
    right = [(2 * m - x, y) for x, y in reversed(left)]
    pts = left + right
    if abs(right[-1][0] - x1) > 1e-6:
        pts.append((x1, right[-1][1]))
    pts += _arc(x1 - r_bottom, y1 - r_bottom, r_bottom, 0, 90) if r_bottom else [(x1, y1)]
    pts += _arc(x0 + r_bottom, y1 - r_bottom, r_bottom, 90, 180) if r_bottom else [(x0, y1)]
    if abs(left[0][0] - x0) > 1e-6:
        pts.append((x0, left[0][1]))
    return pts


def outline_polygon(spec):
    t = spec.get("type", "rounded_rect")
    if t == "polygon":
        return [tuple(map(float, p)) for p in spec["pts"]]
    if t == "rounded_rect":
        return rounded_rect(spec["x0"], spec["y0"], spec["x1"], spec["y1"], spec.get("r_top", 0), spec.get("r_bottom", 0))
    if t == "profile":
        return profile_shape(spec["x0"], spec["x1"], spec["y1"], spec["top_left"], spec.get("r_bottom", 0), spec.get("mirror_x"))
    return None


class RefMask:
    """Coverage of the real product in reference space, plus colour-sampling clamps for extended margins."""

    def __init__(self, spec, ref_u8, ref_alpha=None, k=2, pad=64):
        H, W = ref_u8.shape[:2]
        t = spec.get("type", "rounded_rect")
        self.pad = pad
        if t in ("alpha", "key_white"):
            if t == "alpha":
                if ref_alpha is None:
                    raise ValueError("reference_outline type 'alpha' needs a reference with transparency")
                m = ref_alpha > 127
            else:
                m = binary_fill_holes(ref_u8.min(axis=2) < spec.get("threshold", 245))
            lab, n = label(m)
            if n > 1:
                sizes = np.bincount(lab.ravel())
                sizes[0] = 0
                m = lab == sizes.argmax()
            self.k = 1
            mask = np.zeros((H + 2 * pad, W + 2 * pad), bool)
            mask[pad:pad + H, pad:pad + W] = m
            self.polygon = None
        else:
            self.k = k
            self.polygon = outline_polygon(spec)
            if self.polygon is None:
                raise ValueError(f"unknown reference_outline type {t!r}")
            mask = fill_polygon((H + 2 * pad) * k, (W + 2 * pad) * k, [(self._idx(x), self._idx(y)) for x, y in self.polygon])
        # bleed: the label's paper runs on past its real edge (each pixel there takes the colour of the nearest one
        # 'bleed_inset' px inside), so a job whose visible shape is the generated outline can fill it to the edge
        self.bleed = float(spec.get("bleed", 0.0))
        self.bleed_map = None
        if self.bleed > 0:
            from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
            inner = mask[::self.k, ::self.k][pad:pad + H, pad:pad + W] if self.k > 1 else mask[pad:pad + H, pad:pad + W]
            core = binary_erosion(inner, iterations=max(1, int(round(spec.get("bleed_inset", 2.0)))))
            if core.any():
                iy, ix = distance_transform_edt(~core, return_distances=False, return_indices=True)
                self.bleed_map = (inner, ix.astype(np.float32), iy.astype(np.float32))
            self.mask_bleed = binary_dilation(mask, iterations=int(round(self.bleed * self.k))).astype(np.float32)
        self.clamps = []
        for e in spec.get("extend", []):
            side, to, cl = e["side"], float(e["to"]), float(e["clamp"])
            lo, hi = map(float, e["range"])
            if side in ("left", "right"):
                (xa, xb), (ya, yb) = sorted((to, cl)), (lo, hi)
            else:
                (ya, yb), (xa, xb) = sorted((to, cl)), (lo, hi)
            mask |= fill_polygon(*mask.shape, [(self._idx(x), self._idx(y)) for x, y in ((xa, ya), (xb, ya), (xb, yb), (xa, yb))])
            self.clamps.append((side, cl, lo, hi))
        self.mask = mask.astype(np.float32)

    def _idx(self, c):
        # mask pixel i is centred on reference coord (i + 0.5) / k - 0.5 - pad
        return (c + 0.5 + self.pad) * self.k - 0.5

    def bbox(self, ref_shape, pad=0):
        """Product bounding box in reference px (x0, y0, x1, y1), clipped to the reference image."""
        ys, xs = np.nonzero(self.mask > 0.5)
        to_ref = lambda i: (i + 0.5) / self.k - 0.5 - self.pad
        H, W = ref_shape[:2]
        return (max(0, int(to_ref(xs.min())) - pad), max(0, int(to_ref(ys.min())) - pad),
                min(W, int(np.ceil(to_ref(xs.max()))) + 1 + pad), min(H, int(np.ceil(to_ref(ys.max()))) + 1 + pad))

    def sample(self, u, v):
        return map_coordinates(self.mask, [self._idx(v), self._idx(u)], order=1, mode="constant", cval=0.0)

    def sample_bleed(self, u, v):
        """Coverage of the outline grown by its bleed (the plain outline without one)."""
        m = self.mask_bleed if self.bleed > 0 else self.mask
        return map_coordinates(m, [self._idx(v), self._idx(u)], order=1, mode="constant", cval=0.0)

    def clamp(self, u, v):
        us, vs = u, v
        if self.bleed_map is not None:          # past the real edge: the colour of the nearest pixel well inside
            inner, ix, iy = self.bleed_map
            H, W = inner.shape
            cu = np.clip(np.round(us).astype(int), 0, W - 1)
            cv = np.clip(np.round(vs).astype(int), 0, H - 1)
            out = ~inner[cv, cu]
            us = np.where(out, ix[cv, cu], us)
            vs = np.where(out, iy[cv, cu], vs)
        for side, cl, lo, hi in self.clamps:
            if side in ("left", "right"):
                sel = (v >= lo) & (v <= hi)
                us = np.where(sel, np.maximum(us, cl) if side == "left" else np.minimum(us, cl), us)
            else:
                sel = (u >= lo) & (u <= hi)
                vs = np.where(sel, np.maximum(vs, cl) if side == "top" else np.minimum(vs, cl), vs)
        return us, vs
