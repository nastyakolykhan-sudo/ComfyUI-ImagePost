"""Measuring helpers: rough input in, precise measurements out. Integer coordinates are pixel centres. Images are
float sRGB 0..1, H x W x 3.

The agent (or a person) points roughly at things; these functions do the precise part:
  snap_points / snap_polyline  move rough points along their normals onto a colour edge (sub-pixel)
  propose_lines                per straight side of the real outline: the generated edge as an align.lines entry,
                               kept only where it is straight and near where the rough fit puts it
  snap_silhouette              the generated product's outline, from the real outline under the rough fit
  refine_points                rough landmark pairs -> scene points refined by colour NCC, with a confidence score;
                               hallucinated print scores low and is reported, not trusted
  suggest_landmarks            distinct, well-matching packshot corners under a fit, for the agent to pick from
  trace_outline / white_box    the packshot's outline polygon and a white patch for grade.ref_white_box
Fully automatic fitting (a free homography onto the nearest edges, or RANSAC over correlation matches) was tried
on 17 production jobs and drifted onto neighbours, occluders and print, so it is deliberately not here.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import (binary_erosion, binary_fill_holes, gaussian_filter, gaussian_filter1d, label,
                           map_coordinates, maximum_filter, sobel)

from . import qa
from .geometry import apply_H, as_matrix, line_through

LUMA = np.array([0.2126, 0.7152, 0.0722])


# ---- geometry ----

def _norm_T(P):
    m = P.mean(0)
    s = np.sqrt(2) / max(float(np.mean(np.linalg.norm(P - m, axis=1))), 1e-9)
    return np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1.0]])


def fit_points(src, dst, w=None, model="homography"):
    """Least-squares homography (normalised DLT, 4+ points) or affine (3+), mapping src -> dst."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    w = np.ones(len(src)) if w is None else np.asarray(w, float)
    if model == "affine" or len(src) < 4:
        X = np.linalg.lstsq(np.c_[src, np.ones(len(src))] * w[:, None], dst * w[:, None], rcond=None)[0].T
        return np.vstack([X, [0, 0, 1.0]])
    Ts, Td = _norm_T(src), _norm_T(dst)
    s = (np.c_[src, np.ones(len(src))] @ Ts.T)[:, :2]
    d = (np.c_[dst, np.ones(len(dst))] @ Td.T)[:, :2]
    A = np.zeros((2 * len(s), 9))
    A[0::2, 0:2], A[0::2, 2] = -s, -1
    A[0::2, 6:8], A[0::2, 8] = s * d[:, :1], d[:, 0]
    A[1::2, 3:5], A[1::2, 5] = -s, -1
    A[1::2, 6:8], A[1::2, 8] = s * d[:, 1:], d[:, 1]
    A *= np.repeat(w, 2)[:, None]
    H = np.linalg.inv(Td) @ np.linalg.svd(A)[2][-1].reshape(3, 3) @ Ts
    return H / H[2, 2]


def local_scale(H, p):
    """Scene px per reference px at reference point p."""
    q = apply_H(H, [p, (p[0] + 1.0, p[1]), (p[0], p[1] + 1.0)])
    return float(np.sqrt(abs(np.linalg.det(np.c_[q[1] - q[0], q[2] - q[0]]))))


def resample(P, step, closed=False):
    """Points every `step` px along a polyline (or closed polygon)."""
    P = np.asarray(P, float)
    if closed:
        P = np.vstack([P, P[:1]])
    L = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(L)]
    n = max(int(np.ceil(cum[-1] / step)), 1)
    s = np.linspace(0.0, cum[-1], n + 1)
    if closed:
        s = s[:-1]
    return np.c_[np.interp(s, cum, P[:, 0]), np.interp(s, cum, P[:, 1])]


def inside_polygon(pts, poly):
    """Even-odd test of points against a polygon."""
    x, y = np.atleast_2d(np.asarray(pts, float)).T
    P = np.asarray(poly, float)
    res = np.zeros(len(x), bool)
    for (xa, ya), (xb, yb) in zip(P, np.roll(P, -1, axis=0)):
        cross = (ya > y) != (yb > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = xa + (y - ya) * (xb - xa) / (yb - ya)
        res ^= cross & (x < xi)
    return res


def normals_for(P, start, closed=False, poly=None):
    """Unit normals of a sampled polyline, pointing away from `start`: left/right/above/below (the side the search
    starts on), or for a polygon outside (normals point in) / inside (normals point out)."""
    P = np.asarray(P, float)
    nb = np.roll(P, -1, 0) - np.roll(P, 1, 0) if closed else np.gradient(P, axis=0)
    tg = nb / np.maximum(np.linalg.norm(nb, axis=1, keepdims=True), 1e-9)
    n = np.c_[-tg[:, 1], tg[:, 0]]
    if start in ("left", "right", "above", "below"):
        ax, sgn = {"left": (0, 1), "right": (0, -1), "above": (1, 1), "below": (1, -1)}[start]
        return n * np.where(n[:, ax] * sgn < 0, -1.0, 1.0)[:, None]
    inward = inside_polygon(P + 1.5 * n, poly if poly is not None else P)
    return n * np.where(inward == (start == "outside"), 1.0, -1.0)[:, None]


def _sample(img, x, y):
    if img.ndim == 2:
        return map_coordinates(img, [y, x], order=1, mode="nearest")
    return np.stack([map_coordinates(img[..., c], [y, x], order=1, mode="nearest") for c in range(img.shape[2])], -1)


# ---- edges ----

def snap_points(img, pts, normals, search=6.0, pick="nearest", sigma=0.8, res=0.25, spread=1.0,
                min_contrast=0.04, rel=0.3, lo=None, hi=None):
    """Move each point along its normal onto a colour edge.

    normals point away from the side the search starts on. pick "nearest": the significant edge closest to the
    rough point; "first": the first one walking from -search along the normal, for an occluder's edge measured
    from the occluder's side (inside the product the strongest edge is often hallucinated print).
    An edge is a local maximum of the colour gradient (RGB, smoothed by sigma px, averaged over +-spread px along
    the edge) at least `rel` of the window's strongest, with a colour step of at least min_contrast across 3 px.
    lo / hi (px along the normal) replace the symmetric +-search window.
    -> (N x 2 points, NaN where no edge was found; info dicts with offset px and contrast)
    """
    pts, normals = np.atleast_2d(np.asarray(pts, float)), np.atleast_2d(np.asarray(normals, float))
    lo = -search if lo is None else lo
    hi = search if hi is None else hi
    t = np.arange(lo, hi + res / 2, res)
    tang = np.c_[-normals[:, 1], normals[:, 0]]
    offs = (-spread, 0.0, spread) if spread else (0.0,)
    out, info = np.full_like(pts, np.nan), []
    k3 = int(round(1.5 / res))
    for i, (p, n, tg) in enumerate(zip(pts, normals, tang)):
        prof = 0.0
        for o in offs:
            xy = p[None] + t[:, None] * n[None] + o * tg[None]
            prof = prof + _sample(img, xy[:, 0], xy[:, 1])
        prof = np.atleast_2d((prof / len(offs)).T).T
        sm = gaussian_filter1d(prof, sigma / res, axis=0, mode="nearest")
        g = np.linalg.norm(np.gradient(sm, res, axis=0), axis=1)
        if g.max() <= 0:
            info.append({"ok": False, "why": "flat"})
            continue
        cand = np.nonzero((g[1:-1] >= g[:-2]) & (g[1:-1] > g[2:]) & (g[1:-1] >= rel * g.max()))[0] + 1
        step = lambda j: float(np.linalg.norm(sm[min(j + k3, len(sm) - 1)] - sm[max(j - k3, 0)]))
        cand = [j for j in cand if step(j) >= min_contrast]
        if not cand:
            info.append({"ok": False, "why": "no edge"})
            continue
        j = min(cand, key=lambda j: abs(t[j])) if pick == "nearest" else cand[0]
        den = g[j - 1] - 2 * g[j] + g[j + 1]
        off = 0.5 * (g[j - 1] - g[j + 1]) / den if den else 0.0
        tj = float(t[j] + np.clip(off, -0.5, 0.5) * res)
        out[i] = p + tj * n
        info.append({"ok": True, "offset": round(tj, 2), "edges": len(cand), "contrast": round(step(j), 3)})
    return out, info


def snap_polyline(img, poly, start, step=8.0, closed=False, **kw):
    """Resample a rough polyline every `step` px and snap each sample (see snap_points).
    -> (samples, snapped points with NaN where none, info)"""
    P = resample(poly, step, closed)
    n = normals_for(P, start, closed, poly if closed else None)
    out, info = snap_points(img, P, n, **kw)
    return P, out, info


def fit_line(pts, k=2.5, iters=3):
    """Total-least-squares line through the finite points, dropping outliers beyond k x their robust spread.
    -> (line (a, b, c) with unit normal or None, kept mask, residuals px)"""
    P = np.asarray(pts, float)
    fin = np.isfinite(P).all(1)
    keep = fin.copy()
    if keep.sum() < 2:
        return None, keep, np.full(len(P), np.nan)
    for _ in range(iters):
        L = line_through(P[keep])
        r = np.where(fin, np.nan_to_num(P) @ L[:2] + L[2], np.nan)
        spread = max(1.4826 * float(np.median(np.abs(r[keep]))), 0.15)
        new = fin & (np.abs(np.nan_to_num(r, nan=1e9)) <= k * spread)
        if new.sum() < 2 or np.array_equal(new, keep):
            break
        keep = new
    L = line_through(P[keep])
    return L, keep, np.where(fin, np.nan_to_num(P) @ L[:2] + L[2], np.nan)


def straight_sides(poly, min_len, angle=12.0):
    """Long straight runs of a closed polygon -> [(start point, end point)] in its own coordinates. Consecutive
    edges that turn by less than `angle` degrees are merged; runs shorter than min_len are dropped."""
    P = np.asarray(poly, float)
    n = len(P)
    d = np.roll(P, -1, 0) - P
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    turn = lambda a, b: abs((a - b + 180) % 360 - 180)
    runs, cur = [], [0]
    for i in range(1, n):
        if turn(ang[i], ang[cur[0]]) < angle and turn(ang[i], ang[i - 1]) < angle:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    if len(runs) > 1 and turn(ang[runs[0][0]], ang[runs[-1][-1]]) < angle:
        runs[0] = runs.pop() + runs[0]
    out = []
    for r in runs:
        a, b = P[r[0]], P[(r[-1] + 1) % n]
        if np.linalg.norm(b - a) >= min_len:
            out.append((a, b))
    return out


def propose_lines(img, poly_ref, H, search=6.0, shrink=0.12, n=14, max_offset=3.0, max_resid=1.2, min_found=0.6,
                  min_len=None):
    """align.lines candidates: each long straight side of the real outline, mapped by the rough fit H, snapped to
    the nearest edge. Kept when enough samples find an edge, the edge is straight (TLS residual) and it sits near
    the side's predicted position; a generated outline that differs (a taller top, a kink) is left to the fill.
    -> [{"ref": [[x, y], [x, y]], "scene": [...], "kept": bool, "why": str, "offset": px, "found", "resid"}]"""
    P = np.asarray(poly_ref, float)
    min_len = min_len or 0.25 * float(min(np.ptp(P, 0)))
    S_poly = apply_H(H, P)
    out = []
    for a, b in straight_sides(P, min_len):
        t = np.linspace(shrink, 1 - shrink, n)
        R = a[None] + t[:, None] * (b - a)[None]
        S = apply_H(H, R)
        d = (S[-1] - S[0]) / max(float(np.linalg.norm(S[-1] - S[0])), 1e-9)
        nrm = np.repeat(np.array([[-d[1], d[0]]]), n, 0)
        nrm *= 1.0 if inside_polygon(S[n // 2] + 1.5 * nrm[0], S_poly)[0] else -1.0   # pointing into the product
        T, info = snap_points(img, S, nrm, search=search, pick="nearest")
        off = np.array([i.get("offset", np.nan) for i in info])
        ok = np.isfinite(T).all(1) & (np.abs(np.nan_to_num(off, nan=99.0)) <= max_offset)
        L, keep, r = fit_line(np.where(ok[:, None], T, np.nan))
        good = ok & keep
        entry = {"ref": [[round(float(x), 2) for x in R[0]], [round(float(x), 2) for x in R[-1]]],
                 "scene": [[round(float(x), 2), round(float(y), 2)] for x, y in T[good]],
                 "found": round(float(good.mean()), 2),
                 "offset": round(float(np.median(off[good])), 2) if good.any() else None,
                 "resid": round(float(np.max(np.abs(r[good]))), 2) if good.any() else None}
        why = "" if good.mean() >= min_found else f"edge found on {good.mean():.0%} of the side"
        if not why and entry["resid"] > max_resid:
            why = f"generated edge not straight ({entry['resid']} px)"
        if not why and abs(entry["offset"]) > 0.7 * max_offset:
            why = f"generated edge {entry['offset']:+.1f} px from the real one under the rough fit"
        entry["kept"], entry["why"] = not why, why or "straight, near the prediction"
        out.append(entry)
    return out


def _robust_curve(pos, offs, tol=1.0, deg=2, tries=400):
    """A smooth side through per-sample candidate offsets (N x K, NaN = none): RANSAC over polynomials of `deg`, then
    least squares on the samples with a candidate within tol. The paper's own edge is the one curve most samples
    agree on: print inside and edges beyond (a bottle's base, a carrier) each fool one walk, on part of the side.
    -> (coefficients, share of samples on the curve, rms px) or None"""
    N = len(pos)
    have = np.isfinite(offs)
    if have.any(1).sum() < deg + 2:
        return None
    rng = np.random.default_rng(0)
    parts = [q for q in np.array_split(np.nonzero(have.any(1))[0], deg + 1) if len(q)]
    if len(parts) < deg + 1:
        return None

    def support(c):
        r = np.abs(offs - np.polyval(c, pos)[:, None])
        r = np.where(have, r, np.inf).min(1)
        return r <= tol, r

    best, best_key = None, None
    for _ in range(tries):
        ii = [int(rng.choice(q)) for q in parts]
        oo = [offs[i][have[i]][int(rng.integers(have[i].sum()))] for i in ii]
        if len(set(ii)) < deg + 1:
            continue
        c = np.polyfit(pos[ii], oo, deg)
        ok, r = support(c)
        key = (int(ok.sum()), -float(np.sum(r[ok] ** 2)))
        if best_key is None or key > best_key:
            best, best_key = c, key
    c = best
    for _ in range(3):                                   # least squares on the agreeing samples, nearest candidate each
        ok, _ = support(c)
        if ok.sum() < deg + 2:
            break
        rr = np.abs(offs - np.polyval(c, pos)[:, None])
        j = np.where(have, rr, np.inf).argmin(1)
        y = offs[np.arange(N), j]
        c = np.polyfit(pos[ok], y[ok], deg)
    ok, r = support(c)
    return c, float(ok.mean()), float(np.sqrt(np.mean(r[ok] ** 2))) if ok.any() else 99.0


def _local_curve(pos, offs, c0, tol, win):
    """A side's offset per sample: the candidate nearest the robust curve c0 (within tol), then a local linear fit
    (tricube weights over win px, refitted twice without hits more than 1 px off it). -> (knots, offsets): the
    positions of the samples where the fit stands on hits and the smoothed offsets there"""
    have = np.isfinite(offs)
    r = np.where(have, np.abs(offs - np.polyval(c0, pos)[:, None]), np.inf)
    y = offs[np.arange(len(pos)), r.argmin(1)]
    ok = r.min(1) <= tol
    w_ok = ok.astype(float)
    out = np.polyval(c0, pos).astype(float)
    for _ in range(3):
        for k, p in enumerate(pos):
            w = np.clip(1.0 - (np.abs(pos - p) / win) ** 3, 0.0, None) ** 3 * w_ok
            if (w > 0).sum() < 4:
                continue
            A = np.c_[np.ones_like(pos), pos - p] * np.sqrt(w)[:, None]
            out[k] = np.linalg.lstsq(A, np.where(ok, y, 0.0) * np.sqrt(w), rcond=None)[0][0]
        w_ok = (ok & (np.abs(y - out) <= 1.0)).astype(float)
    keep = np.convolve(ok.astype(float), np.ones(5), "same") > 0          # where hits stand within 2 samples
    keep[[0, -1]] = True
    return pos[keep], out[keep]


def label_outline(img, corners, inset=3.0, search=6.0, step=2.0, skip=0.08, tol=1.0, sharp=(), r_max=None, passes=2):
    """See _label_outline. A second pass starts from the first pass's corners when one moved more than 1.5 px:
    the walks then start where the edge really is (a rough corner 4 px off put a side's start on the far side of a
    shadow line under the label)."""
    r = _label_outline(img, corners, inset, search, step, skip, tol, sharp, r_max)
    for _ in range(passes - 1):
        moved = float(np.max(np.linalg.norm(np.asarray(r["sharp"]) - np.asarray(corners, float), axis=1)))
        if moved <= 1.5:
            break
        corners = np.asarray(r["sharp"], float)
        r = _label_outline(img, corners, inset, search, step, skip, tol, sharp, r_max)
    return r


def _label_outline(img, corners, inset=3.0, search=6.0, step=2.0, skip=0.08, tol=1.0, sharp=(), r_max=None):
    """A generated label's own outline from 4 rough corners in order around it (within ~2 px of the true edges).

    Sides: every sample walks out from `inset` px inside to the first colour edge and in from `search` px outside to
    the first one; a robust quadratic through both sets of hits finds the curve they agree on (the paper's edge:
    print fools the inside walk, a carrier or a bottle's base the outside one, never both on the same stretch).
    Corners: the two sides' curves meet at the sharp corner; the paper's edge along the bisector gives the radius
    r = d / (1 / sin(angle / 2) - 1) and an arc tangent to both sides replaces it. Corners listed in `sharp` (indices
    into `corners`: an occluder hides them) stay sharp.
    -> dict(polygon K x 2, sides [4 x M x 2], corners [4 x arc points], radii [4], support [4] share of samples on
       each side's curve, rms [4] px)"""
    C0 = np.asarray(corners, float)
    cen = C0.mean(0)
    img_s = gaussian_filter(img, (0.7, 0.7, 0) if img.ndim == 3 else 0.7)      # for the double-edge band test
    kw = dict(pick="first", rel=0.05, min_contrast=0.03)
    S = []
    for i in range(4):
        a, b = C0[i], C0[(i + 1) % 4]
        L = float(np.linalg.norm(b - a))
        t = (b - a) / L
        n = np.array([t[1], -t[0]])
        if np.dot(n, (a + b) / 2 - cen) < 0:
            n = -n
        pos = np.arange(skip * L, (1 - skip) * L + 1e-6, step)
        P = a[None] + pos[:, None] * t[None]
        N = np.repeat(n[None], len(P), 0)
        Tin = snap_points(img, P - inset * n[None], N, lo=-1.0, hi=inset + search, **kw)[0]
        Tout = snap_points(img, P + search * n[None], -N, lo=-1.0, hi=search + inset, **kw)[0]
        oi, oo = (Tin - P) @ n, (Tout - P) @ n
        # a double edge: the two walks stop a few px apart (a light rim on the paper's cut edge; a dark line of
        # shadow under the label). The band between belongs to the label where it looks like the paper inside,
        # not like what lies beyond: keep the outer hit there, else the inner one
        edge = "single"
        dual = np.isfinite(oi) & np.isfinite(oo) & (oo - oi > 1.0) & (oo - oi < 8.0)
        if dual.any():
            def col(off):
                q = P[dual] + off[:, None] * n[None]
                return np.stack([_sample(img_s, q[:, 0] + dx, q[:, 1] + dy) for dx, dy in
                                 ((0, 0), (t[0], t[1]), (-t[0], -t[1]))], 0).mean(0).reshape(len(q), -1)
            paper, beyond = col(oi[dual] - 3.0), col(oo[dual] + 3.0)
            # the band's least paper-like point decides (a 1 px dark line between two hits 3 px apart)
            bands = [col(oi[dual] + f * (oo - oi)[dual]) for f in (0.25, 0.5, 0.75)]
            far = np.argmax(np.stack([np.abs(b_ - paper).max(-1) for b_ in bands]), axis=0)
            band = np.stack(bands)[far, np.arange(len(far))]
            outer = np.abs(band - paper).max(-1) < np.abs(band - beyond).max(-1)
            k = np.nonzero(dual)[0]
            oi, oo = oi.copy(), oo.copy()
            oi[k[outer]] = np.nan
            oo[k[~outer]] = np.nan
            edge = f"double on {dual.sum()} of {len(pos)} samples: outer kept on {outer.sum()} (the band is paper), inner on {(~outer).sum()}"
        fit = _robust_curve(pos, np.c_[oi, oo], tol)
        if fit is None:
            raise ValueError(f"side {i + 1} ({a.round(1).tolist()} -> {b.round(1).tolist()}): too few edge hits, "
                             "check the corners")
        # the quadratic picks the edge out of the clutter; the side itself follows the hits on it locally (a
        # generated edge bends unevenly: steeper toward a bottle's silhouette, a kink an AI left), straight past its ends
        loc = _local_curve(pos, np.c_[oi, oo], fit[0], 1.5 * tol, max(30.0, 0.12 * L))
        S.append({"a": a, "t": t, "n": n, "L": L, "c": fit[0], "loc": loc, "support": fit[1], "rms": fit[2], "edge": edge})

    def off(s, p):
        P_, F_ = s["loc"]
        if p <= P_[0]:                   # straight past the ends, along the last stretch's slope
            return F_[0] + (p - P_[0]) * (F_[1] - F_[0]) / max(P_[1] - P_[0], 1e-9)
        if p >= P_[-1]:
            return F_[-1] + (p - P_[-1]) * (F_[-1] - F_[-2]) / max(P_[-1] - P_[-2], 1e-9)
        return float(np.interp(p, P_, F_))

    def at(s, p):
        return s["a"] + p * s["t"] + off(s, p) * s["n"]

    def tangent(s, p):
        d = s["t"] + (off(s, p + 1.0) - off(s, p - 1.0)) / 2.0 * s["n"]
        return d / np.linalg.norm(d)

    corner_info = []
    for j in range(4):                                   # corner j: side j-1 arrives, side j leaves
        s1, s2 = S[j - 1], S[j]
        p1, p2 = s1["L"], 0.0
        for _ in range(5):
            A, B = at(s1, p1), at(s2, p2)
            T1, T2 = tangent(s1, p1), tangent(s2, p2)
            M = np.array([T1, -T2]).T
            if abs(np.linalg.det(M)) < 1e-6:
                break
            k1, _ = np.linalg.solve(M, B - A)
            C = A + k1 * T1
            p1, p2 = float((C - s1["a"]) @ s1["t"]), float((C - s2["a"]) @ s2["t"])
        t_in, t_out = tangent(s1, p1), tangent(s2, p2)
        cos_th = float(np.clip(-t_in @ t_out, -0.999, 0.999))
        half = 0.5 * np.arccos(cos_th)
        bis = -t_in + t_out
        bis = bis / np.linalg.norm(bis)
        if bis @ (cen - C) < 0:
            bis = -bis
        info = {"C": C, "r": 0.0, "arc": np.array([C]), "p1": p1, "p2": p2}
        if j not in sharp:
            k_r = 1.0 / np.sin(half) - 1.0
            rm = r_max if r_max else 0.2 * min(s1["L"], s2["L"])
            dmax = rm * k_r + 2.0
            side = np.array([bis[1], -bis[0]])
            hits = []
            for e in (-1.0, 0.0, 1.0):
                q = C + e * side
                ti = snap_points(img, q[None] + dmax * bis[None], -bis[None], lo=-1.0, hi=dmax + 2.0, **kw)[0][0]
                to = snap_points(img, q[None] - 3.0 * bis[None], bis[None], lo=-1.0, hi=dmax + 4.0, **kw)[0][0]
                hits += [float((h - q) @ bis) for h in (ti, to) if np.isfinite(h).all()]
            hits = [h for h in hits if -1.0 <= h <= dmax]
            if hits:
                d = max(0.0, float(np.median(hits)))
                r = min(d / k_r, rm)
                if r >= 0.5:
                    ctr = C + bis * r / np.sin(half)
                    q1 = C - t_in * r / np.tan(half)
                    q2 = C + t_out * r / np.tan(half)
                    a1 = np.arctan2(*(q1 - ctr)[::-1])
                    a2 = np.arctan2(*(q2 - ctr)[::-1])
                    da = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
                    m = max(2, int(np.ceil(abs(da) * r / 1.0)))
                    ang = a1 + da * np.linspace(0, 1, m + 1)
                    arc = ctr[None] + r * np.c_[np.cos(ang), np.sin(ang)]
                    info.update(r=r, arc=arc, p1=float((q1 - s1["a"]) @ s1["t"]), p2=float((q2 - s2["a"]) @ s2["t"]))
        corner_info.append(info)
    sides_pts, poly = [], []
    for i in range(4):
        s = S[i]
        pa, pb = corner_info[i]["p2"], corner_info[(i + 1) % 4]["p1"]
        m = max(2, int(np.ceil(abs(pb - pa) / 1.5)))
        side = np.array([at(s, p) for p in np.linspace(pa, pb, m + 1)])
        sides_pts.append(side)
        poly.append(side[1:-1])
        poly.append(corner_info[(i + 1) % 4]["arc"])
    return {"polygon": np.vstack(poly), "sides": sides_pts, "corners": [c["arc"] for c in corner_info],
            "sharp": [c["C"] for c in corner_info], "radii": [round(float(c["r"]), 2) for c in corner_info],
            "support": [round(s["support"], 2) for s in S], "rms": [round(s["rms"], 2) for s in S],
            "edges": [s["edge"] for s in S]}


def snap_silhouette(img, poly_ref, H, out=10.0, back=2.0, step=1.0, push=1.5, margin=1.0, corner_reach=12.0):
    """The generated product's outline: from just inside the real outline (mapped by H) walk outward and stop at
    the first edge, where the generated product's colour ends: taller tops of a hallucinated outline are covered,
    print further inside isn't reached. Found edges move `margin` px further out to take in the edge's blur
    fringe; where nothing is found within `out` px, the real outline pushed `push` px outward. Corners between
    two straight sides extend to where the sides' fitted lines meet (up to corner_reach px out), which covers
    square generated corners on a rounded real product. -> (polygon scene px, share found)"""
    S = resample(apply_H(H, poly_ref), step, closed=True)
    nrm = normals_for(S, "inside", closed=True, poly=S)          # pointing out of the product
    T, _ = snap_points(img, S, nrm, pick="first", lo=-back, hi=out)
    ok = np.isfinite(T).all(1)
    T = np.where(ok[:, None], T, S)
    poly = np.where((ok & (np.sum((T - S) * nrm, 1) > -push))[:, None], T + margin * nrm, S + push * nrm)

    P = np.asarray(poly_ref, float)
    sides = straight_sides(P, 0.25 * float(min(np.ptp(P, 0))))
    runs = []                                                    # (first, last sample index, fitted line) per side
    for a, b in (apply_H(H, [a, b]) for a, b in sides):
        ab = b - a
        t = ((S - a) @ ab) / max(float(ab @ ab), 1e-9)
        d = np.abs((S - a) @ np.array([-ab[1], ab[0]]) / max(float(np.linalg.norm(ab)), 1e-9))
        idx = np.nonzero((t > 0.15) & (t < 0.85) & (d < 3.0) & ok)[0]
        if len(idx) >= 5:
            L, keep, _ = fit_line(poly[idx])
            if L is not None:
                runs.append((idx, L))
    if len(runs) >= 2:
        n = len(S)
        runs.sort(key=lambda r: r[0][0])
        cut = np.zeros(n, bool)
        extra = []
        for k in range(len(runs)):
            (ia, La), (ib, Lb) = runs[k], runs[(k + 1) % len(runs)]
            A = np.array([La[:2], Lb[:2]])
            if abs(np.linalg.det(A)) < 0.2:                         # nearly parallel sides: no corner
                continue
            X = np.linalg.solve(A, -np.array([La[2], Lb[2]]))
            i0, i1 = int(ia.max()), int(ib.min())
            span = np.arange(i0 + 1, i1 if i1 > i0 else i1 + n) % n
            if not len(span):
                continue
            reach = float(np.min(np.linalg.norm(poly[span] - X, axis=1)))
            if reach <= corner_reach and not inside_polygon(X[None], poly)[0]:
                cut[span] = True
                extra.append((int(span[len(span) // 2]), X))
        if extra:
            keep_idx = [i for i in range(n) if not cut[i]] + [i for i, _ in extra]
            pts = {i: poly[i] for i in range(n) if not cut[i]}
            pts.update({i: X for i, X in extra})
            poly = np.array([pts[i] for i in sorted(set(keep_idx))])
    return _dp(np.vstack([poly, poly[:1]]), 0.5)[:-1], float(ok.mean())


# ---- fitting to a mask ----

def mask_from_background(img, box, bg=None, thresh=28.0, open_it=3, close_it=6):
    """The product inside `box` (x0, y0, x1, y1) of a scene on a fairly plain background: pixels whose colour (sRGB
    0..255, lightly smoothed) differs from the background colour `bg` by more than `thresh`; holes filled, largest
    component. bg defaults to the median colour of the box's border. -> bool mask of the whole image."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    a = np.stack([gaussian_filter(img[y0:y1, x0:x1, c] * 255.0, 2.0) for c in range(3)], -1)
    if bg is None:
        bg = np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]), 0)
    m = np.linalg.norm(a - np.asarray(bg, float), axis=2) > thresh
    from scipy.ndimage import binary_closing, binary_opening
    m = binary_fill_holes(binary_closing(binary_opening(m, iterations=open_it), iterations=close_it))
    lab, n = label(m)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        m = lab == sizes.argmax()
    out = np.zeros(img.shape[:2], bool)
    out[y0:y1, x0:x1] = m
    return out


def _similarity(src, dst):
    """Least-squares rotation + uniform scale + translation (Umeyama) mapping src -> dst, as a 3 x 3 matrix."""
    ms, md = src.mean(0), dst.mean(0)
    A, B = src - ms, dst - md
    U, S, Vt = np.linalg.svd(B.T @ A)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, d]) @ Vt
    sc = float(np.sum(S * np.array([1.0, d])) / max(float(np.sum(A * A)), 1e-12))
    M = np.eye(3)
    M[:2, :2], M[:2, 2] = sc * R, md - sc * R @ ms
    return M


def fit_to_mask(mask, poly_ref, H0, model="homography", step=3.0, skip=None, iters=40, refine=True):
    """Fit the real outline onto a product mask's contour, starting from a rough homography H0.

    Iterative closest point: every sampled outline point pairs with the nearest contour pixel (no search window,
    so a side that starts far off still pulls), pairs beyond 2.5 x their median distance are dropped (leaks), and
    the transform is refitted: rotation + scale first, then affine, then a homography. A mask has no clutter next
    to its edge, so unlike fitting to image edges this can't latch onto a neighbour or print. A last pass snaps to
    the mask's 0.5 crossing for sub-pixel accuracy. `skip(ref_pts) -> bool array` leaves parts of the outline out
    (a cap the job keeps). -> dict(H, residual px per sample, used share, notes)"""
    from scipy.spatial import cKDTree
    P = np.asarray(poly_ref, float)
    D = resample(P, step / max(local_scale(H0, P.mean(0)), 1e-6), closed=True)
    use = np.ones(len(D), bool) if skip is None else ~np.asarray(skip(D), bool)
    edge = mask & ~binary_erosion(mask)
    by, bx = np.nonzero(edge)
    tree = cKDTree(np.c_[bx, by].astype(float))
    H, notes = H0 / H0[2, 2], []
    stages = [("similarity", iters // 3), ("affine", iters // 3), (model, iters - 2 * (iters // 3))]
    for kind, n_it in stages:
        if kind == "homography" and model == "affine":
            continue
        for _ in range(n_it):
            S = apply_H(H, D[use])
            dist, idx = tree.query(S)
            keep = dist <= max(2.5 * float(np.median(dist)), 2.0)
            src, dst = D[use][keep], tree.data[idx[keep]]
            if kind == "similarity":
                Hn = _similarity(apply_H(H, src), dst) @ H
            else:
                Hn = fit_points(src, dst, model="affine" if kind == "affine" else "homography")
            moved = float(np.max(np.linalg.norm(apply_H(Hn, D) - apply_H(H, D), axis=1)))
            H = Hn / Hn[2, 2]
            if moved < 0.05:
                break
        notes.append(f"{kind}: median distance to the contour {np.median(dist):.2f} px, {keep.mean():.0%} of points kept")
    resid = np.full(len(D), np.nan)
    if refine:
        m = gaussian_filter(mask.astype(float), 1.0)
        S = apply_H(H, D)
        nrm = normals_for(S, "outside", closed=True, poly=S)
        T, _ = snap_points(m, S, nrm, search=3.0, pick="nearest", sigma=0.5, spread=0.0, min_contrast=0.3)
        ok = use & np.isfinite(T).all(1)
        if ok.sum() >= 12:
            r = np.sum((S - T) * nrm, 1)
            good = ok & (np.abs(np.nan_to_num(r, nan=99.0)) <= max(2.5 * float(np.median(np.abs(r[ok]))), 0.75))
            H = fit_points(D[good], T[good], model=model)
            H = H / H[2, 2]
            S = apply_H(H, D)
            resid[ok] = np.sum((S[ok] - T[ok]) * nrm[ok], 1)
            notes.append(f"sub-pixel pass: {int(good.sum())} points, median |residual| {np.median(np.abs(resid[good])):.2f} px")
    return {"H": H, "ref": D, "residual": resid, "used": float(np.mean(np.isfinite(resid))), "notes": notes}


# ---- landmarks ----

def _ncc_surface(win, patch):
    """NCC of `patch` (h x w x C) against every placement in `win` -> (ny, nx) scores."""
    ph, pw = patch.shape[:2]
    W = sliding_window_view(win, (ph, pw), axis=(0, 1))          # ny, nx, C, ph, pw
    P = np.moveaxis(patch, -1, 0)
    Wz = W - W.mean(axis=(-2, -1), keepdims=True)
    Pz = P - P.mean(axis=(-2, -1), keepdims=True)
    num = np.einsum("yxcij,cij->yx", Wz, Pz)
    den = np.sqrt(np.einsum("yxcij,yxcij->yx", Wz, Wz) * float(np.sum(Pz * Pz)))
    return np.where(den > 1e-12, num / np.maximum(den, 1e-12), -1.0)


def _peak(S):
    """Best placement (sub-pixel) and the runner-up outside its 5 x 5 neighbourhood."""
    iy, ix = np.unravel_index(int(np.argmax(S)), S.shape)
    best = float(S[iy, ix])
    sub = []
    for a, b, c in ((S[iy, ix - 1] if ix > 0 else best, best, S[iy, ix + 1] if ix < S.shape[1] - 1 else best),
                    (S[iy - 1, ix] if iy > 0 else best, best, S[iy + 1, ix] if iy < S.shape[0] - 1 else best)):
        den = a - 2 * b + c
        sub.append(float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den < 0 else 0.0)
    T = S.copy()
    T[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3] = -1.0
    return ix + sub[0], iy + sub[1], best, float(T.max())


def prefiltered(ref, H, at):
    """The packshot blurred for sampling at the scene's scale around reference point `at`."""
    f = 1.0 / max(local_scale(H, at), 1e-6)
    return np.stack([gaussian_filter(ref[..., c], max(0.5 * f, 0.3)) for c in range(3)], -1)


def render_patch(ref_s, H, c, half):
    """The packshot around scene point c as the scene would show it under H: (2h+1)^2 x 3 on the grid c + d."""
    d = np.arange(-half, half + 1, dtype=float)
    X, Y = np.meshgrid(c[0] + d, c[1] + d)
    q = np.c_[X.ravel(), Y.ravel(), np.ones(X.size)] @ np.linalg.inv(H).T
    return _sample(ref_s, q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]).reshape(X.shape + (ref_s.shape[2],))


def _shift(H, d):
    return np.array([[1, 0, d[0]], [0, 1, d[1]], [0, 0, 1.0]]) @ H


def match_at(scene, ref_s, H, p_ref, guess, radius=5, half=8):
    """Find reference point p_ref in the scene near `guess`: the packshot patch rendered under H (moved so p_ref
    lands on guess), searched within +-radius px by colour NCC. -> dict(scene, ncc, gap), None near the frame edge"""
    guess = np.asarray(guess, float)
    Hs = _shift(H, guess - apply_H(H, [p_ref])[0])
    q = np.round(guess).astype(int)
    x0, y0 = q[0] - radius - half, q[1] - radius - half
    x1, y1 = q[0] + radius + half + 1, q[1] + radius + half + 1
    if x0 < 0 or y0 < 0 or x1 > scene.shape[1] or y1 > scene.shape[0]:
        return None
    patch = render_patch(ref_s, Hs, guess, half)
    if patch.std(axis=(0, 1)).max() < 0.02:
        return {"scene": guess.tolist(), "ncc": 0.0, "gap": 0.0, "flat": True}
    px, py, best, second = _peak(_ncc_surface(scene[y0:y1, x0:x1], patch))
    return {"scene": [float(q[0] - radius + px), float(q[1] - radius + py)], "ncc": round(best, 3),
            "gap": round(best - second, 3)}


def refine_points(scene, ref, H, pairs, radius=5, half=None, min_ncc=0.85, min_gap=0.03):
    """Refine rough landmark pairs [(ref xy, rough scene xy)] by colour NCC. H is the current rough fit; its local
    perspective shapes the patch. A result is trusted when ncc >= min_ncc and the match is distinct; untrusted
    ones keep the rough scene point. -> dicts (ref, rough, scene, ncc, gap, trusted, moved, why)"""
    out = []
    for p_ref, rough in pairs:
        p_ref, rough = np.asarray(p_ref, float), np.asarray(rough, float)
        h = half or int(np.clip(round(10 * max(local_scale(H, p_ref), 0.4)), 6, 12))
        m = match_at(scene, prefiltered(ref, H, p_ref), as_matrix(H, p_ref), p_ref, rough, radius, h)
        if m is None:
            out.append({"ref": p_ref.tolist(), "rough": rough.tolist(), "scene": rough.tolist(), "ncc": None,
                        "gap": None, "trusted": False, "moved": 0.0, "why": "too close to the frame edge"})
            continue
        ok = m["ncc"] >= min_ncc and m["gap"] >= min_gap and not m.get("flat")
        sc = m["scene"] if ok else rough.tolist()
        out.append({"ref": p_ref.tolist(), "rough": rough.tolist(), "scene": [round(v, 2) for v in sc],
                    "ncc": m["ncc"], "gap": m["gap"], "trusted": bool(ok),
                    "moved": round(float(np.linalg.norm(np.asarray(sc) - rough)), 2),
                    "why": "matched" if ok else ("flat patch" if m.get("flat") else
                                                 "weak match: the generated print differs here")})
    return out


def corners(L, mask, sigma=1.5, n=80, min_dist=6, rel=0.02):
    """Shi-Tomasi corners inside mask -> (N x 2 (x, y), strength), strongest first."""
    gx, gy = sobel(L, 1), sobel(L, 0)
    Sxx, Syy, Sxy = (gaussian_filter(a, sigma) for a in (gx * gx, gy * gy, gx * gy))
    lam = np.where(mask, 0.5 * (Sxx + Syy - np.sqrt((Sxx - Syy) ** 2 + 4 * Sxy ** 2)), 0.0)
    if lam.max() <= 0:
        return np.zeros((0, 2)), np.zeros(0)
    pk = (lam == maximum_filter(lam, size=2 * int(min_dist) + 1)) & (lam > rel * lam.max())
    ys, xs = np.nonzero(pk)
    s = lam[ys, xs]
    o = np.argsort(-s)[:n]
    return np.c_[xs[o], ys[o]].astype(float), s[o]


def suggest_landmarks(scene, ref, ref_mask, H, k=8, radius=4, min_ncc=0.9, min_gap=0.05, n=120, half=8):
    """Packshot corners that match the scene well under the fit H (within +-radius px): candidates for
    align.points, spread over the product. -> dicts (ref, scene, ncc, gap), best first, at most k."""
    ys, xs = np.nonzero(ref_mask)
    centre = (float(xs.mean()), float(ys.mean()))
    f = 1.0 / max(local_scale(H, centre), 1e-6)
    ref_s = prefiltered(ref, H, centre)
    hs, ws = int(ref.shape[0] / f) + 1, int(ref.shape[1] / f) + 1
    GX, GY = np.meshgrid((np.arange(ws) + 0.5) * f - 0.5, (np.arange(hs) + 0.5) * f - 0.5)
    small = _sample(ref_s, GX.ravel(), GY.ravel()).reshape(hs, ws, 3)
    ms = map_coordinates(ref_mask.astype(float), [GY.ravel(), GX.ravel()], order=0).reshape(hs, ws) > 0.5
    pts, _ = corners(small @ LUMA, binary_erosion(ms, iterations=half + 2), n=n, min_dist=6)
    found = []
    for p in np.c_[(pts[:, 0] + 0.5) * f - 0.5, (pts[:, 1] + 0.5) * f - 0.5]:
        m = match_at(scene, ref_s, as_matrix(H, p), p, apply_H(H, [p])[0], radius, half)
        if m and m["ncc"] >= min_ncc and m["gap"] >= min_gap:
            found.append({"ref": [round(float(p[0]), 1), round(float(p[1]), 1)],
                          "scene": [round(v, 2) for v in m["scene"]], "ncc": m["ncc"], "gap": m["gap"]})
    found.sort(key=lambda m: -m["ncc"])
    spread = max(np.ptp(xs), np.ptp(ys)) / 6.0
    picked = []
    for m in found:
        if all(np.hypot(m["ref"][0] - q["ref"][0], m["ref"][1] - q["ref"][1]) >= spread for q in picked):
            picked.append(m)
        if len(picked) == k:
            break
    return picked


def match_sheet(scene, ref, H, points, half=8, scale=5):
    """Review tiles: the packshot rendered under H around each point | the scene there."""
    tiles = []
    for i, m in enumerate(points):
        c = np.asarray(m["scene"], float)
        Hs = _shift(as_matrix(H, m["ref"]), c - apply_H(H, [m["ref"]])[0])
        a = render_patch(prefiltered(ref, H, m["ref"]), Hs, c, half)
        d = np.arange(-half, half + 1, dtype=float)
        X, Y = np.meshgrid(c[0] + d, c[1] + d)
        b = _sample(scene, X.ravel(), Y.ravel()).reshape(a.shape)
        pair = np.concatenate([a, np.full((a.shape[0], 1, 3), 0.15), b], axis=1)
        im = qa.Image.fromarray(qa.to_u8(pair)).resize((pair.shape[1] * scale, pair.shape[0] * scale), qa.Image.NEAREST)
        tag = f"ncc {m['ncc']}" if m.get("ncc") is not None else "rough"
        tiles.append(qa.labelled(im, f"{i}: {m.get('label', '')[:22]} {tag}", h=22, size=14))
    return qa.montage(tiles, max_w=1800) if tiles else None


# ---- checks ----

def orientation(H, box):
    """How the fit leans the product: lean (dx/dy) of its mapped left and right sides, slope (dy/dx) of its top
    and bottom. Verticals that fan (different leans) read as a skewed product (user review, round 4)."""
    x0, y0, x1, y1 = box
    q = apply_H(H, [(x0, y0), (x0, y1), (x1, y0), (x1, y1)])
    lean = lambda a, b: float((b[0] - a[0]) / (b[1] - a[1])) if b[1] != a[1] else float("inf")
    slope = lambda a, b: float((b[1] - a[1]) / (b[0] - a[0])) if b[0] != a[0] else float("inf")
    o = {"lean_left": lean(q[0], q[1]), "lean_right": lean(q[2], q[3]),
         "slope_top": slope(q[0], q[2]), "slope_bottom": slope(q[1], q[3])}
    o["fan"] = o["lean_right"] - o["lean_left"]
    o["converge"] = o["slope_bottom"] - o["slope_top"]
    return {k: round(v, 4) for k, v in o.items()}


# ---- packshot ----

def background(ref_u8):
    """The packshot's border colour (median) and each border pixel's distance from it."""
    b = np.concatenate([ref_u8[0], ref_u8[-1], ref_u8[:, 0], ref_u8[:, -1]]).astype(float)
    c = np.median(b, 0)
    return c, np.linalg.norm(b - c, axis=1)


def _bg_tol(d):
    return max(6.0, 3.0 * 1.4826 * float(np.median(d)))


def on_plain_background(ref_u8, frac=0.9):
    """True when the packshot's border is one plain colour (white, grey, any), so its outline can be traced."""
    c, d = background(ref_u8)
    return float(np.mean(d <= _bg_tol(d))) >= frac


def reference_mask(ref_u8, ref_alpha=None, tol=None):
    """The product in a packshot: alpha > 127, or what differs from the plain border colour by more than tol
    (holes filled); largest component. A white product edge on white needs a measured outline instead."""
    if ref_alpha is not None:
        m = ref_alpha > 127
    else:
        c, d = background(ref_u8)
        m = binary_fill_holes(np.linalg.norm(ref_u8.astype(float) - c, axis=2) > (tol or _bg_tol(d)))
    lab, n = label(m)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        m = lab == sizes.argmax()
    return m


def outline_fringe(ref_u8, poly, near=1.0, step=1.0, reach=6.0):
    """Where a measured outline runs into the packshot's plain backdrop. Along each outline sample's outward normal,
    the backdrop distance (colour distance from the border colour) falls from the product's level to the backdrop's;
    the 50% crossing is the product's edge. A sample nearer that edge than `near` px (or past it) takes in
    anti-aliased backdrop, which composites as a light rim. Samples with no backdrop beyond them (another item
    in the packshot) can't be judged and are skipped. -> list of stretches dict(from, to, px, worst), worst = the
    sample's distance inside the edge (negative: past it), worst first."""
    from scipy.ndimage import map_coordinates
    c, d = background(ref_u8)
    tol = _bg_tol(d)
    dist = np.linalg.norm(ref_u8.astype(float) - c, axis=2)
    P = np.asarray(poly, float)
    S = resample(P, step, closed=True)
    N = normals_for(S, "inside", closed=True, poly=S)       # outward
    ts = np.arange(-reach, reach + 0.01, 0.25)
    Q = S[:, None, :] + ts[None, :, None] * N[:, None, :]
    v = map_coordinates(dist, [Q[..., 1].ravel(), Q[..., 0].ravel()], order=1, mode="nearest").reshape(Q.shape[:2])
    inner = np.median(v[:, ts <= -3], 1)
    outer = np.median(v[:, ts >= 3], 1)
    judged = (outer <= 1.5 * tol) & (inner > 3 * tol)
    inside = np.full(len(S), np.nan)
    for i in np.nonzero(judged)[0]:
        half = 0.5 * (inner[i] + outer[i])
        above = np.nonzero(v[i] >= half)[0]          # from the backdrop end inward: the last product sample
        if len(above) and above[-1] < len(ts) - 1:
            k = above[-1]
            t0, t1, v0, v1 = ts[k], ts[k + 1], v[i, k], v[i, k + 1]
            inside[i] = t0 + (v0 - half) / (v0 - v1) * (t1 - t0)
    bad = np.isfinite(inside) & (inside < near)
    out, i, n = [], 0, len(S)
    while i < n:
        if not bad[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and bad[j + 1]:
            j += 1
        seg = inside[i:j + 1]
        out.append({"from": [round(float(v), 1) for v in S[i]], "to": [round(float(v), 1) for v in S[j]],
                    "px": round(float((j - i + 1) * step), 1), "worst": round(float(seg.min()), 2)})
        i = j + 1
    return sorted(out, key=lambda s: s["worst"])


def _dp(P, tol):
    """Douglas-Peucker simplification of an open polyline."""
    if len(P) < 3:
        return P
    a, b = P[0], P[-1]
    ab = b - a
    nrm = float(np.hypot(*ab))
    d = np.abs(ab[0] * (P[:, 1] - a[1]) - ab[1] * (P[:, 0] - a[0])) / nrm if nrm else np.linalg.norm(P - a, axis=1)
    i = int(np.argmax(d))
    if d[i] <= tol:
        return np.array([a, b])
    return np.vstack([_dp(P[:i + 1], tol)[:-1], _dp(P[i:], tol)])


def trace_outline(mask, step=None, tol=0.75):
    """Outline polygon (reference px) of a star-shaped mask: boundary crossings along rows and columns, ordered by
    angle around the centroid, simplified to within tol px."""
    ys, xs = np.nonzero(mask)
    step = step or max(1, int(min(np.ptp(ys), np.ptp(xs))) // 150)
    pts = []
    for y in range(ys.min(), ys.max() + 1, step):
        r = np.nonzero(mask[y])[0]
        if len(r):
            pts += [(r[0] - 0.5, y), (r[-1] + 0.5, y)]
    for x in range(xs.min(), xs.max() + 1, step):
        c = np.nonzero(mask[:, x])[0]
        if len(c):
            pts += [(x, c[0] - 0.5), (x, c[-1] + 0.5)]
    P = np.unique(np.round(np.array(pts, float), 2), axis=0)
    P = P[np.argsort(np.arctan2(P[:, 1] - ys.mean(), P[:, 0] - xs.mean()))]
    S = _dp(np.vstack([P, P[:1]]), tol)[:-1]
    return [[round(float(x), 2), round(float(y), 2)] for x, y in S]


def white_box(ref_u8, mask, size=None, min_l=0.9, max_sat=0.08):
    """A near-white, flat box inside the product (reference px) for grade.ref_white_box; None if there is none."""
    f = ref_u8.astype(float) / 255.0
    L = f @ LUMA
    ok = mask & (L >= min_l) & ((f.max(2) - f.min(2)) <= max_sat)
    ys, xs = np.nonzero(mask)
    size = size or int(np.clip(min(np.ptp(xs), np.ptp(ys)) / 12, 8, 60))
    core = binary_erosion(ok, iterations=size // 2 + 1)
    if not core.any():
        return None
    score = np.where(core, gaussian_filter(L, size / 2), 0.0)
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    hb = size // 2
    return [int(x - hb), int(y - hb), int(x + hb), int(y + hb)]
