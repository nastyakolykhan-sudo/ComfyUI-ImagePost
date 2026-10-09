"""Automatic QA after a run: the problems a reviewer circles, measured so they can't be skipped.

  covered   the product went over colour the generated product doesn't have: an occluder in front the job missed
            (a carrier's wall over a label's corner) or an outline past the generated edge (glass, background)
  leftover  a rim of the generated product left just outside the new matte
  fill      fill thick enough to show as a smear
  squeeze   (cylinder) print within a few degrees of the scene's silhouette, pressed into the edge
  edges     an edge line still more than 1.5 px off after the fit

Each finding: {"check", "level" (FAIL | WARN | INFO), "box" [x0, y0, x1, y1] scene px, "text"}. FAIL and WARN need a
look at qa/audit.png (before | after tiles of every finding) before delivery.
"""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, find_objects, label, uniform_filter

from . import qa


def colour_model(crop_u8, mask, q=16):
    """Share of the generated product's pixels per colour bin (q levels per channel), smoothed over neighbours."""
    idx = (crop_u8[mask] // (256 // q)).astype(int)
    h = np.zeros((q, q, q))
    np.add.at(h, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    h = uniform_filter(h, size=3, mode="constant") * 27.0
    return h / max(float(h.sum()), 1.0)


def share(model, crop_u8, q=16):
    i = (crop_u8 // (256 // q)).astype(int)
    return model[i[..., 0], i[..., 1], i[..., 2]]


def _components(mask, offset, min_px=1):
    lab, n = label(mask)
    out = []
    for i, sl in enumerate(find_objects(lab), 1):
        m = lab[sl] == i
        a = int(m.sum())
        if a < min_px:
            continue
        thick = 2.0 * float(distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1].max())
        ys, xs = sl
        out.append({"px": a, "thick": round(thick, 1),
                    "box": [xs.start + offset[0], ys.start + offset[1], xs.stop - 1 + offset[0], ys.stop - 1 + offset[1]]})
    return out


def _local_mean(img, mask, win):
    """Mean colour of the masked pixels in a win x win window around every pixel (normalized box filter)."""
    w = mask.astype(float)
    num = np.stack([uniform_filter(img[..., k].astype(float) * w, win) for k in range(3)], -1)
    den = uniform_filter(w, win)
    return num / np.maximum(den, 1e-6)[..., None], den


def limb_guard(H, ref_lin, inside):
    """A cylinder's limbs checked against the packshot's label paper: if the paper bottoms out and climbs onto a flat
    plateau before a measured limb, that "limb" is the edge of the packshot's drop shadow. -> warning lines"""
    from .grade import flatten_columns
    _, prof = flatten_columns(ref_lin, inside)
    Lp = prof.mean(1)
    out = []
    for side, sgn in (("right", 1), ("left", -1)):
        us = np.arange(int(H.axis_x), int(H.axis_x + sgn * (H.r - 2)), sgn)
        us = us[(us >= 0) & (us < len(Lp))]
        if len(us) > 20:
            k = int(np.argmin(Lp[us]))
            rise = float(Lp[us[k:]].max() / max(Lp[us[k]], 1e-6)) if k < len(us) - 3 else 1.0
            if rise > 1.15 and k >= len(us) // 2 and abs(us[-1] - us[k]) > 4:   # a dip near the limb, not a lit side
                out.append(f"limb       the packshot's label paper bottoms out at u {us[k]} and climbs {rise:.2f}x before "
                           f"the {side} limb ({H.axis_x + sgn * H.r:.1f}): the drop shadow? Measure the limb where the "
                           "label or glass meets the shadow, not the shadow's outer edge (pitfall 64)")
    return out


def squeeze_info(H, refmask, ref_shape, ref_lin):
    """How far round the packshot's print reaches against the scene's silhouette, per side, in degrees (cylinders).
    Bands running across the label are not text. -> {side: {"print", "limb"}} or None"""
    if not hasattr(H, "axis_x"):
        return None
    ry, rx = np.mgrid[0:ref_shape[0], 0:ref_shape[1]].astype(float)
    ins = (refmask.sample(rx, ry) > 0.5) & (np.abs(rx - H.axis_x) < H.r)
    ins = binary_erosion(ins, iterations=max(2, int(0.03 * H.r)))            # the label's own edge isn't print
    Lr = ref_lin @ np.array([0.2126, 0.7152, 0.0722])
    if not ins.any():
        return None
    ink = ins & (Lr < 0.45 * np.percentile(Lr[ins], 90))
    band = np.zeros(ink.shape[0], bool)                                     # unbroken runs over 12 % of the width
    for yy in np.nonzero(ink.any(1))[0]:
        e = np.diff(np.r_[0, ink[yy].astype(np.int8), 0])
        band[yy] = (np.nonzero(e < 0)[0] - np.nonzero(e > 0)[0]).max() > 0.12 * max(int(ins[yy].sum()), 1)
    ink &= ~band[:, None]
    if ink.sum() <= 20:
        return None
    th = np.degrees(np.arcsin(np.clip((rx[ink] - H.axis_x) / H.r, -1, 1)))
    out = {}
    for side, sel in (("right", th > 0), ("left", th < 0)):
        if sel.any():
            out[side] = {"print": float(np.percentile(np.abs(th[sel]), 99.0)),
                         "limb": abs(float(np.degrees(H.limb_angle(side))))}
    return out


def run(crop_u8, after_u8, alpha, old, vis, F, offset, align_rep=None, squeeze=None):
    """-> (findings, audit sheet PIL image or None). squeeze: {side: {"print": deg the print reaches, "limb": deg}}."""
    found = []
    prod = (alpha > 0.5) & (vis > 0.5)
    ys, xs = np.nonzero(prod)
    if not len(ys):
        return found, None
    size = float(min(np.ptp(ys), np.ptp(xs)) + 1)
    t_fail = max(3.0, 0.03 * size)                   # px of thickness that reads as a region, not an edge
    band = max(6.0, 0.06 * size)                     # occluders and overshoot happen near the edge
    win = int(2 * band + 1) | 1
    d_in = distance_transform_edt(prod)              # inside the new product, px from its edge
    d_out = distance_transform_edt(~prod)            # outside it
    gen = binary_erosion(old & (vis > 0.99), iterations=3)
    # local colours in the ORIGINAL frame: the generated product's interior and the background just outside the new one
    pm, pden = _local_mean(crop_u8, gen, win)
    bgm, bden = _local_mean(crop_u8, (d_out >= 3) & (d_out <= band + 3) & (vis > 0.5) & ~old, win)

    def dist(a, m):
        return np.abs(a.astype(float) - m).max(-1)

    # covered: near the new edge, pixels whose original colour is the background's, not the product's
    near = prod & (d_in <= band) & (pden > 0.02) & (bden > 0.02)
    db, dp = dist(crop_u8, bgm), dist(crop_u8, pm)
    cov = near & (db < 0.5 * dp) & (dp > 30)
    for c in sorted(_components(cov, offset, 4), key=lambda c: -c["px"])[:6]:
        lvl = "FAIL" if c["thick"] >= t_fail and c["px"] >= max(12, 0.004 * prod.sum()) else \
              "WARN" if c["thick"] >= 2.5 else "INFO"
        what = "an occluder in front the job missed, or the outline runs past the generated product" if lvl != "INFO" \
            else f"a strip up to {c['thick']:.0f} px wide past the generated edge"
        found.append({"check": "covered", "level": lvl, "box": c["box"],
                      "text": f"product over {c['px']} px of the background's colour ({what})"})

    # leftover: just outside the new edge, the result looks like the generated product, not the background beyond
    ring = (d_out >= 1) & (d_out <= 3) & (vis > 0.5) & ~F
    far_m, fden = _local_mean(after_u8, (d_out >= 6) & (d_out <= 10) & (vis > 0.5), win)
    left = ring & (fden > 0.02) & (pden > 0.02) & (dist(after_u8, pm) < 0.5 * dist(after_u8, far_m)) & \
        (dist(after_u8, far_m) > 30) & (np.abs(far_m - pm).max(-1) > 40)   # a rim only shows on a different background
    for c in sorted(_components(left, offset, 6), key=lambda c: -c["px"])[:4]:
        found.append({"check": "leftover", "level": "WARN" if c["px"] >= 12 else "INFO", "box": c["box"],
                      "text": f"{c['px']} px just outside the new edge look like the generated product, not the "
                              "background beyond: a rim may show"})

    # fill: thick fill regions smear at small scale
    for c in sorted(_components(F & (vis > 0.5), offset, 6), key=lambda c: -c["thick"])[:4]:
        if c["thick"] >= max(2.5, 0.02 * size):
            found.append({"check": "fill", "level": "WARN", "box": c["box"],
                          "text": f"fill {c['thick']:.0f} px thick over {c['px']} px: check it doesn't smear"})

    # squeeze: print pressed into a cylinder's silhouette. The scene sees the bottle turned by (limb - 90) deg; print at
    # angle a shows cos(a + turn) / cos(a) as wide as in the packshot (hidden past 90 deg)
    for side, v in (squeeze or {}).items():
        turn = 90.0 - v["limb"]
        a = v["print"]
        ratio = np.cos(np.radians(min(a + turn, 90.0))) / max(np.cos(np.radians(a)), 1e-6)
        if ratio < 0.72:                              # reviewed: 0.48-0.64 read as crushed, 0.75 passed
            found.append({"check": "squeeze", "level": "FAIL" if ratio < 0.66 else "WARN", "box": None,
                          "text": f"print reaching {a:.0f} deg on the {side} shows {ratio:.2f}x as wide as in the "
                                  f"packshot (silhouette at {v['limb']:.0f} deg): pressed into the edge. Set "
                                  "cylinder.limb_min (88) and make the label's printed edge one_sided"})

    # edges: residuals after the fit (after the residual correction when there is one)
    if align_rep:
        after = {l["label"]: l["max_px"] for l in (align_rep.get("residual") or {}).get("lines_after", [])}
        for l in align_rep.get("lines", []):
            m = after.get(l["label"], l["max_px"])
            if m > 1.5:
                found.append({"check": "edges", "level": "WARN", "box": None,
                              "text": f"{l['label']}: {m:.1f} px off the generated edge after the fit"})

    return found, sheet(crop_u8, after_u8, found, offset)


def sheet(before_u8, after_u8, found, offset, tile=220):
    tiles = []
    h, w = before_u8.shape[:2]
    for f in found:
        if not f.get("box"):
            continue
        x0, y0, x1, y1 = (f["box"][0] - offset[0], f["box"][1] - offset[1], f["box"][2] - offset[0], f["box"][3] - offset[1])
        half = int(max(x1 - x0, y1 - y0) / 2 + 10)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        a, b = max(cx - half, 0), max(cy - half, 0)
        c, d = min(cx + half + 1, w), min(cy + half + 1, h)
        pair = []
        for arr in (before_u8, after_u8):
            im = qa.Image.fromarray(np.ascontiguousarray(arr[b:d, a:c]))
            s = tile / max(im.width, im.height)
            pair.append(im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))),
                                  qa.Image.NEAREST if s >= 3 else qa.Image.LANCZOS))
        tiles.append(qa.labelled(qa.side_by_side(pair, gap=4), f"{f['level']} {f['check']} at x {f['box'][0]}, y {f['box'][1]}",
                                 h=22, size=14))
    return qa.montage(tiles, max_w=1400) if tiles else None
