"""Light and colour transfer from the generated product to the real one.

Robust to hallucinated content: the light comes from a smooth field (no pixel matching), the colour from
per-channel quantile matching (no pixel alignment). A free 3x3 colour-matrix regression overfits here.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, percentile_filter
from scipy.optimize import least_squares

from .images import lum


def _basis(X, Y, frame):
    cx, cy, sx, sy = frame
    xs, ys = (X - cx) / sx, (Y - cy) / sy
    return np.stack([np.ones_like(xs), xs, ys, xs * xs, xs * ys, ys * ys], -1)


def _frame(mask, X, Y):
    xs, ys = X[mask], Y[mask]
    return xs.mean(), ys.mean(), max(np.ptp(xs) / 2, 1.0), max(np.ptp(ys) / 2, 1.0)


def robust_quadratic(B, z, iters=3, scale=0.1, init_step=7):
    c = np.linalg.lstsq(B[::init_step], z[::init_step], rcond=None)[0]
    for _ in range(iters):
        r = B @ c - z
        w = 1 / (1 + (r / scale) ** 2)
        c = np.linalg.lstsq(B * w[:, None] ** 0.5, z * w ** 0.5, rcond=None)[0]
    return c


def white_level_gain(S_lin, inner, deep, X, Y, pct=92, window=41):
    """Light falloff = smooth fit to the generated product's local white level (its brightest material)."""
    wl = percentile_filter(np.where(inner, lum(S_lin), 0.0), pct, size=window)
    sel = deep & (wl > 0.05)
    if sel.sum() < 50:
        raise ValueError("white-level gain: too few interior pixels; lower grade.deep_erode or use gain 'luma_ratio'")
    fr = _frame(sel, X, Y)
    c = robust_quadratic(_basis(X[sel], Y[sel], fr), np.log(wl[sel]))
    return np.exp(_basis(X, Y, fr) @ c)


def luma_ratio_gain(S_lin, R_lin, mask, X, Y, sigma=6.0, match_hue=None, flat=0.08):
    """Light falloff = smooth fit to blurred scene/reference luminance ratio (for products without light areas).
    match_hue (deg): read the ratio only where both images show the same saturated ink, on flat areas (no edge in
    either), unblurred, so a generated print laid out differently can't read as light (a printed box: its
    background ink). The white level of such a box is set by its print, not by the light."""
    if not match_hue:
        Ls, Lr = gaussian_filter(lum(S_lin), sigma), gaussian_filter(lum(R_lin), sigma)
        sel = mask & (Ls > 1e-3) & (Lr > 1e-3)
        fr = _frame(sel, X, Y)
        c = robust_quadratic(_basis(X[sel], Y[sel], fr), np.log(Ls[sel] / Lr[sel]))
        g = np.exp(_basis(X, Y, fr) @ c)
        return g / np.median(g[sel])
    Ls, Lr = lum(S_lin), lum(R_lin)
    hs, ss = _hue_sat(S_lin)
    hr, sr = _hue_sat(R_lin)
    dh = np.abs((hs - hr + 180) % 360 - 180)

    def edge(L):
        gy, gx = np.gradient(np.log(gaussian_filter(L, 0.8) + 1e-3))
        return np.hypot(gx, gy)

    sel = mask & (ss > 0.2) & (sr > 0.2) & (dh < match_hue) & (edge(Ls) < flat) & (edge(Lr) < flat)
    if sel.sum() < 500:
        raise ValueError(f"luma_ratio gain: only {int(sel.sum())} px show the same ink in both images; "
                         "raise grade.match_hue or drop it")
    fr = _frame(sel, X, Y)
    c = robust_quadratic(_basis(X[sel], Y[sel], fr), np.log(Ls[sel] + 1e-3) - np.log(Lr[sel] + 1e-3))
    g = np.exp(_basis(X, Y, fr) @ c)
    return g / np.median(g[mask])


def tone_curves(S_lin, R_lin, g, mask, nq=50):
    """Per channel: scene/gain ~ a * ref^gamma + b, fitted on quantiles (alignment-free)."""
    qs = np.linspace(1, 99, nq)
    Rw, Sn = R_lin[mask], (S_lin / g[..., None])[mask]
    out = []
    for ch in range(3):
        rq, sq = np.percentile(Rw[:, ch], qs), np.percentile(Sn[:, ch], qs)
        sol = least_squares(lambda p: p[0] * np.power(np.clip(rq, 1e-5, None), p[1]) + p[2] - sq,
                            [1.0, 1.0, 0.0], bounds=([0.05, 0.3, -0.2], [3, 3, 0.5]))
        out.append({"a": round(float(sol.x[0]), 4), "gamma": round(float(sol.x[1]), 4), "b": round(float(sol.x[2]), 4),
                    "max_quantile_err": round(float(np.abs(sol.fun).max()), 4), "_p": sol.x})
    return out


def clip_weight(R_lin, lo=0.88, hi=0.98):
    """0..1 weight of near-clipped reference pixels: smoothstep on the sRGB max channel between lo and hi."""
    from .images import lin_to_srgb
    t = np.clip((lin_to_srgb(R_lin).max(-1) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def apply_grade(R_lin, g, curves):
    return np.stack([g * (c["_p"][0] * np.power(np.clip(R_lin[..., i], 1e-5, None), c["_p"][1]) + c["_p"][2])
                     for i, c in enumerate(curves)], -1)


def white_chroma(S_lin, mask, g, frac=0.9):
    """Colour of the generated product's near-white pixels (RGB / luminance, median)."""
    L = lum(S_lin)
    sel = mask & (L >= frac * g)
    if sel.sum() < 30:
        sel = mask & (L >= np.percentile(L[mask], 95))
    return np.median(S_lin[sel] / np.maximum(L[sel], 1e-6)[:, None], 0)


def keep_reference_colour(R_lin, white, black, ref_white):
    """Keep the real product's own colours: reference black -> scene black point, reference white -> scene
    white point (per pixel, so the light falloff and white balance come from the scene)."""
    return black + (white - black) * (R_lin / np.asarray(ref_white, float))


def _hue_sat(lin):
    from .images import lin_to_srgb
    c = lin_to_srgb(lin)
    mx, mn = c.max(-1), c.min(-1)
    d = np.maximum(mx - mn, 1e-6)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h, (mx - mn) / np.maximum(mx, 1e-6)


def shading_profile(S_lin, G_lin, U, mask, bins=32, smooth=1.0, match_hue=None, eps=1e-3):
    """Residual shading across the product along reference x (cylinders: cans, bottles).
    Median per column bin of log(scene luminance / graded luminance) inside mask, smoothed, mean-free.
    match_hue (deg): only use pixels where scene and reference show the same saturated colour (same
    material, e.g. blue ink on both), so differences in the print don't read as light."""
    from scipy.ndimage import gaussian_filter1d
    if match_hue:
        hs, ss = _hue_sat(S_lin)
        hg, sg = _hue_sat(G_lin)
        dh = np.abs((hs - hg + 180) % 360 - 180)
        same = mask & (ss > 0.25) & (sg > 0.25) & (dh < match_hue)
        if same.sum() > 200:
            mask = same
    r = np.log(lum(S_lin)[mask] + eps) - np.log(lum(G_lin)[mask] + eps)
    u = U[mask]
    lo, hi = np.percentile(u, 1), np.percentile(u, 99)
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(u, edges) - 1, 0, bins - 1)
    prof, cnt = np.full(bins, np.nan), np.bincount(idx, minlength=bins)
    for b in range(bins):
        if cnt[b] >= 20:
            prof[b] = np.median(r[idx == b])
    ok = np.isfinite(prof)
    centers = 0.5 * (edges[:-1] + edges[1:])
    prof = np.interp(centers, centers[ok], prof[ok])
    if smooth > 0:
        prof = gaussian_filter1d(prof, smooth, mode="nearest")
    prof -= np.average(prof, weights=np.maximum(cnt, 1))
    return centers, prof


def apply_profile(G_lin, U, centers, prof, strength=1.0, clip=(0.4, 2.5)):
    f = np.clip(np.exp(strength * np.interp(U, centers, prof)), *clip)
    return G_lin * f[..., None]


def cylinder_light(G_lin, U, x_range, limb=0.0, spec=None, white=0.9, shadow=None):
    """Cylinder cues the packshot's studio light removed: darken toward both edges (limb, 0..1) and add a
    soft specular streak (spec = {"t": 0..1 across the body, "width": fraction, "gain": 0..1}, screen-blended)."""
    t = (U - x_range[0]) / float(x_range[1] - x_range[0])
    x = np.clip(2 * t - 1, -1, 1)
    out = G_lin * (1 - limb * (1 - np.sqrt(1 - x * x)))[..., None]
    if shadow:   # core shadow on the side away from the light: 1 - gain at the edge, fading out by t = to (or from)
        if shadow.get("side", "left") == "left":
            u = np.clip(t / shadow.get("to", 0.4), 0, 1)
        else:
            u = np.clip((1 - t) / (1 - shadow.get("from", 0.6)), 0, 1)
        out = out * (1 - shadow.get("gain", 0.4) * (1 - u * u * (3 - 2 * u)))[..., None]
    for sp in ([spec] if isinstance(spec, dict) else (spec or [])):
        sb = sp.get("gain", 0.3) * np.exp(-((t - sp.get("t", 0.85)) / sp.get("width", 0.05)) ** 2)
        out = out + sb[..., None] * (white - out)
    return out


def _block_stat(v, valid, block, fn, min_count):
    """Per-block statistic of v over valid pixels -> (hb, wb) grid, NaN where too few pixels."""
    import warnings
    h, w = v.shape
    hb, wb = -(-h // block), -(-w // block)
    pad = np.full((hb * block, wb * block), np.nan)
    pad[:h, :w] = np.where(valid, v, np.nan)
    b = pad.reshape(hb, block, wb, block).transpose(0, 2, 1, 3).reshape(hb, wb, -1)
    cnt = np.isfinite(b).sum(-1)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        out = fn(np.where(cnt[..., None] >= min_count, b, np.nan), axis=-1)
    return out, cnt


def _fill_smooth(grid, sb, min_conf):
    """Normalized convolution on the block grid: fills NaN blocks, smooths, returns (field, confidence)."""
    from scipy.ndimage import gaussian_filter
    ok = np.isfinite(grid)
    num = gaussian_filter(np.where(ok, grid, 0.0), sb)
    den = gaussian_filter(ok.astype(float), sb)
    return num / np.maximum(den, 1e-6), np.clip(den / min_conf, 0, 1)


def _to_pixels(grid, shape, block, sigma):
    from scipy.ndimage import gaussian_filter, zoom
    h, w = shape
    up = zoom(grid, block, order=1)[:h, :w]
    return gaussian_filter(up, sigma / 2)


def _local_white(L, mask, block):
    g, _ = _block_stat(L, mask, block, lambda a, axis: np.nanpercentile(a, 90, axis=axis), max(4, block * block // 10))
    f, _ = _fill_smooth(g, 1.0, 0.05)
    return np.maximum(_to_pixels(f, L.shape, block, block), 1e-4)


def paper_field(img_lin, mask, size=31, sigma=3.0, pre=1.0):
    """Colour of a label's paper under the light, per pixel: the scene's white with whatever tint and falloff its
    light gives it (warm bounce from amber glass or fruit near a bottle's edge, the darkening toward its limb). Per
    channel, a grey-scale closing (size px, square) lifts away everything dark and narrower than size (text, a
    wordmark's strokes, a printed band) and keeps smooth ramps of light; then a light Gaussian. Outside mask the
    nearest inside pixel stands in, so neighbours (glass, an occluder) don't leak in. size must exceed the widest
    dark print. -> h x w x 3 linear."""
    from scipy.ndimage import distance_transform_edt, grey_closing
    idx = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    # near the edge a large closing overshoots light that falls off faster and faster (toward a bottle's limb):
    # there a small one takes over (print near a label's edge is small), blended by the distance from the edge
    small = max(5, size // 4) | 1
    w = np.clip(distance_transform_edt(mask) / float(size), 0.0, 1.0)[..., None]
    out = []
    for c in range(3):
        ch = gaussian_filter(img_lin[..., c][tuple(idx)], pre)          # the envelope of paper, not of its noise
        out.append(np.stack([grey_closing(ch, size=(size, size)), grey_closing(ch, size=(small, small))], -1))
    out = np.stack(out, -2)                                              # h x w x 3 x (big, small)
    env = w * out[..., 0] + (1 - w) * out[..., 1]
    # the envelope sits on the bright side of the paper's grain and mottling: the field is the mean of the pixels
    # it marks as paper (within 8 % of it), spread over the print it lifted away (normalized convolution)
    filled = np.stack([gaussian_filter(img_lin[..., c][tuple(idx)], pre) for c in range(3)], -1)
    paper = (lum(filled) >= 0.92 * lum(env)).astype(float)
    sp = max(sigma, size / 4.0)
    den = gaussian_filter(paper, sp)
    f = np.stack([gaussian_filter(filled[..., c] * paper, sp) for c in range(3)], -1) / np.maximum(den, 1e-6)[..., None]
    f = np.where((den > 0.05)[..., None], f, env)
    return np.maximum(np.stack([gaussian_filter(f[..., c], sigma) for c in range(3)], -1), 1e-4)


def flatten(ref_lin, inside, mode="field", columns=True):
    """The packshot's own light on its label paper taken out, so the paper reads 1.0 everywhere. "columns" divides
    by a per-column paper level (exact for a cylinder's steep falloff toward its limbs); "field" does that when
    columns is on (cylinders), then divides by the 2D paper field (paper_field: the rest, e.g. light from top to
    bottom). A flat product skips the column step: its per-column noise would print as stripes. -> flattened"""
    flat = flatten_columns(ref_lin, inside)[0] if (columns or mode == "columns") else ref_lin
    if mode == "columns":
        return flat
    ys, xs = np.nonzero(inside)
    size = 2 * max(2, round(min(np.ptp(ys), np.ptp(xs)) / 24)) + 1
    return flat / paper_field(flat, inside, size, max(1.0, size / 10))


def flatten_columns(ref_lin, inside, pct=92, sigma=2.0):
    """The packshot's own light on a cylinder's label (its studio falloff toward the limbs) taken out: each column's
    paper level (pct-th percentile over the label's rows, per channel, lightly smoothed along x) divided out, so the
    paper reads 1.0 everywhere. -> (flattened reference, profile per column, n columns x 3)"""
    from scipy.ndimage import gaussian_filter1d
    cols = np.nonzero(inside.sum(0) >= 8)[0]
    prof = np.full((ref_lin.shape[1], 3), np.nan)
    for x in cols:
        prof[x] = np.percentile(ref_lin[inside[:, x], x], pct, axis=0)
    ok = np.isfinite(prof[:, 0])
    xs = np.arange(ref_lin.shape[1])
    for c in range(3):
        prof[:, c] = np.interp(xs, xs[ok], prof[ok, c])
    prof = np.maximum(gaussian_filter1d(prof, sigma, axis=0, mode="nearest"), 1e-3)
    return ref_lin / prof[None, :, :], prof


def materials(W_lin, mask, k=10, iters=12, seed=3):
    """Material ids from the real packshot's own colours (k-means on log-luminance + chromaticity of the warped
    reference, which carries no scene light). Each ink or paper is a tight cluster, so one id = one material."""
    L = lum(W_lin)
    tot = W_lin.sum(-1) + 1e-6
    feat = np.stack([np.log(L + 1e-3), 4 * (W_lin[..., 0] - W_lin[..., 1]) / tot, 4 * (W_lin[..., 0] + W_lin[..., 1] - 2 * W_lin[..., 2]) / tot], -1)
    X = feat[mask]
    if len(X) < k * 20:
        return np.where(mask, 0, -1)
    rng = np.random.default_rng(seed)
    sub = X[rng.choice(len(X), min(len(X), 30000), replace=False)]
    C = sub[np.argsort(sub[:, 0])[np.linspace(0, len(sub) - 1, k).astype(int)]]
    for _ in range(iters):
        lab = np.argmin(((sub[:, None, :] - C[None]) ** 2).sum(-1), 1)
        C = np.array([sub[lab == i].mean(0) if (lab == i).any() else C[i] for i in range(k)])
    ids = np.full(mask.shape, -1)
    ids[mask] = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), 1)
    return ids


def _veil_fit(S_lin, G_lin, sel):
    """Per pixel S = k*G + s on 3 channels (2 unknowns, closed-form least squares) -> (k, s) arrays over sel."""
    G3, S3 = G_lin[sel], S_lin[sel]
    n = 3.0
    sc, scc = G3.sum(1), (G3 * G3).sum(1)
    sv, scs = S3.sum(1), (G3 * S3).sum(1)
    det = n * scc - sc ** 2
    det = np.where(np.abs(det) < 1e-6, np.nan, det)
    return (n * scs - sc * sv) / det, (scc * sv - sc * scs) / det


def relight_map(S_lin, G_lin, mask, sigma=14.0, match_hue=25.0, neutral=True, edge=0.12, min_conf=0.15, eps=1e-3,
                block=16, inlier=0.35, tier_tol=0.5, iters=2, sheen=None, W_lin=None, n_materials=10, light_cut=0.12,
                per_material=False, min_share=0.04, detail=None, anchor_white=True, **_):
    """Light of the source (generated) image that the global grade missed: cast shadows from neighbours,
    falloff, glows, and optionally the glossy veil of wrapped or shiny packs.
      1. Veil (sheen on): fit S = k*G + s on inks (3 channels, 2 unknowns), keep a smooth field of s/k (the veil
         relative to the diffuse light) and add it to the composite: light is then read against graded + veil on ALL
         materials, and a cast shadow darkens the veil with everything else.
      2. Materials come from the real packshot (k-means on the warped reference's colours). A source pixel
         counts when it shows the material's colour family and the same lightness relative to its local white
         in both images (a cast shadow darkens a neighbourhood together; a mismatched print stroke does not).
      3. Per material the shade offset is removed. Light-coloured materials drive the estimate, dark inks
         fill in where nothing lighter is near. Block medians with outlier rejection give a smooth field.
      4. detail (on by default): a fine pass on light materials restores crisp shadow and highlight edges
         the smooth field softens; only spatially coherent structure is kept, so print-shaped blobs are not.
    Returns (log light map, veil map in graded units or None, stats); relit = (graded + veil) * exp(light).""" 
    from scipy.ndimage import gaussian_filter, binary_opening
    stats = {}
    S_map = None
    hs0, ss0 = _hue_sat(S_lin)
    hg, sg = _hue_sat(G_lin)
    lg = np.log(lum(G_lin) + eps)
    gg = np.hypot(*np.gradient(gaussian_filter(lg, 0.8)))
    sb = max(sigma / block, 0.5)
    mc = max(4, block * block // 8)
    mid = materials(W_lin, mask, n_materials) if W_lin is not None else None
    if sheen is not None and sheen is not False:
        dh0 = np.abs((hs0 - hg + 180) % 360 - 180)
        ls0 = np.log(lum(S_lin) + eps)
        gs0 = np.hypot(*np.gradient(gaussian_filter(ls0, 0.8)))
        if W_lin is not None:
            # the real ink is saturated (packshot); the source shows either the same hue or a washed-out grey (haze)
            _hw0, sw0 = _hue_sat(W_lin)
            ink = mask & (sw0 > 0.3) & ((dh0 < match_hue) | (ss0 < 0.15)) & (gs0 < edge) & (gg < edge)
        else:
            ink = mask & (sg > 0.25) & (ss0 > 0.05) & (dh0 < match_hue) & (gs0 < edge) & (gg < edge)
        vcfg = sheen if isinstance(sheen, dict) else {}
        lit = bool(vcfg.get("lit", True))
        kk, sh = _veil_fit(S_lin, G_lin, ink)
        Sp = np.full(lg.shape, np.nan)
        # lit (gloss on the surface): veil relative to the diffuse light, so a cast shadow dims it too.
        # unlit (reflection haze of a wrap or window): absolute veil, added after the light
        Sp[ink] = np.where(np.isfinite(kk) & (kk > 0.05), sh / np.maximum(kk, 0.05) if lit else sh, np.nan)
        okp = np.isfinite(Sp)
        # a wrap's haze or a gloss veil varies smoothly: large blocks, iterated rejection of pixels that disagree
        # (a generated ink lighter than the real one would otherwise read as a local veil patch)
        vb, vsig, vin = int(vcfg.get("block", 24)), float(vcfg.get("sigma", 40.0)), float(vcfg.get("inlier", 0.12))
        okv = okp.copy()
        for _ in range(3):
            gS, _c = _block_stat(Sp, okv, vb, np.nanmedian, max(4, vb * vb // 8))
            fS, cS = _fill_smooth(gS, max(vsig / vb, 0.5), min_conf)
            S_map = np.maximum(_to_pixels(fS * cS, lg.shape, vb, vsig), 0.0)
            okv = okp & (np.abs(Sp - S_map) < vin)
        stats["veil_inliers"] = int(okv.sum())
        # the veil scales with the light that reaches the surface (a cast shadow darkens the gloss too), so it is
        # part of the material's look: light is read against graded + veil, and relit = (graded + veil) * light
        if lit:
            G_lin = G_lin + S_map[..., None]
            hg, sg = _hue_sat(G_lin)
            lg = np.log(lum(G_lin) + eps)
            gg = np.hypot(*np.gradient(gaussian_filter(lg, 0.8)))
        else:
            S_lin = np.maximum(S_lin - S_map[..., None], eps)   # light is read on the de-hazed source
        stats["ink_px"] = int(okp.sum())
        stats["veil"] = "lit" if lit else "unlit"
    Ls, Lg = lum(S_lin), lum(G_lin)
    hs, ss = _hue_sat(S_lin)
    dh = np.abs((hs - hg + 180) % 360 - 180)
    ls = np.log(Ls + eps)
    gs = np.hypot(*np.gradient(gaussian_filter(ls, 0.8)))
    flat = (gs < edge) & (gg < edge)
    rel = (ls - np.log(_local_white(Ls, mask, block))) - (lg - np.log(_local_white(Lg, mask, block)))
    tier = np.abs(rel) < tier_tol
    if W_lin is not None:   # coloured or neutral by the REAL material (graded paper carries the scene's warm cast)
        _hw, sw = _hue_sat(W_lin)
        chrom = sw > 0.15
        neu = (mask & ~chrom & (ss < 0.3)) if neutral else np.zeros_like(mask)
    else:
        sw = None
        chrom = sg > 0.12
        neu = (mask & ~chrom & (ss < 0.25)) if neutral else np.zeros_like(mask)
    inks = mask & chrom & (ss > 0.05) & (dh < match_hue)
    r = ls - lg
    if mid is not None:   # one class per real material (ink or paper), so ink-shade differences never read as light
        ok = (inks | neu) & flat & tier
        classes = [ok & (mid == i) for i in range(n_materials)]
    else:
        classes = [neu & flat & tier] + [inks & flat & tier & (((hg - h0) % 360) < 30) for h0 in range(0, 360, 30)]
    offs = {i: float(np.median(r[c])) for i, c in enumerate(classes) if c.sum() >= 100}
    # absolute level: the real white (paper, card) is white in both images, so its offset is light, not shade.
    # Every material is normalised to its own median, then shifted by the white's offset.
    base, anchor = 0.0, None
    if anchor_white and mid is not None and sw is not None:
        Lw = lum(W_lin)
        npx = max(int(mask.sum()), 1)
        cand = [(int(classes[i].sum()), i) for i in offs
                if classes[i].sum() >= 0.05 * npx and np.median(Lw[mid == i]) > 0.45 and np.median(sw[mid == i]) < 0.15]
        if cand:
            anchor = max(cand)[1]
            base = offs[anchor]
    stats["anchor"] = None if anchor is None else {"material": int(anchor), "offset": round(base, 3)}
    resid = np.full(r.shape, np.nan)
    light = np.zeros(r.shape, bool)
    for i, c in enumerate(classes):
        if i in offs:
            resid[c] = r[c] - offs[i] + base
            if np.median(Lg[c]) >= light_cut:   # paper, tan, white, light inks: light shows on them reliably
                light |= c
    used = np.isfinite(resid)
    valid = used.copy()
    for _ in range(iters + 1):
        # light-coloured materials first; dark inks only where no light material is near (a generated image
        # often renders one dark ink lighter in one place than another, which must not read as light)
        gL, _c = _block_stat(resid, valid & light, block, np.nanmedian, mc)
        gA, _c = _block_stat(resid, valid, block, np.nanmedian, mc)
        g = np.where(np.isfinite(gL), gL, gA)
        f, conf = _fill_smooth(g, sb, min_conf)
        F = _to_pixels(f * conf, r.shape, block, sigma)
        valid = used & (np.abs(resid - F) < inlier)
    stats.update({"matched_px": int(used.sum()), "inliers": int(valid.sum()),
                  "of_mask": round(float(valid.sum() / max(mask.sum(), 1)), 3)})
    if per_material and mid is not None:
        # surfaces at different angles get their own field where they have data, blended in by confidence
        npx = max(int(mask.sum()), 1)
        own = 0
        for i in range(n_materials):
            ci = valid & (mid == i)
            if ci.sum() < min_share * npx:
                continue
            gi, _c = _block_stat(resid, ci, block, np.nanmedian, mc)
            fi, confi = _fill_smooth(gi, sb, 0.5)
            wi = _to_pixels(confi, r.shape, block, sigma)
            F = np.where(mask & (mid == i), wi * _to_pixels(fi, r.shape, block, sigma) + (1 - wi) * F, F)
            own += 1
        stats["own_light_materials"] = own
    if detail is not False:
        # fine pass: crisp shadow / highlight edges. Light materials only, small blocks, a wider inlier band
        # around the smooth field, and only spatially coherent structure survives (print blobs don't)
        dc = detail if isinstance(detail, dict) else {}
        db, dmin, coh = int(dc.get("block", 4)), float(dc.get("min", 0.04)), int(dc.get("coherence", 5))
        dres = resid - F
        dvalid = used & light & (np.abs(dres) < float(dc.get("inlier", 0.8)))
        gD, _c = _block_stat(dres, dvalid, db, np.nanmedian, max(3, db * db // 3))
        fD, cD = _fill_smooth(gD, float(dc.get("smooth", 1.0)), 0.3)
        Dp = _to_pixels(fD * cD, r.shape, db, db)
        yy, xx = np.ogrid[-coh:coh + 1, -coh:coh + 1]
        disk = (xx * xx + yy * yy) <= coh * coh
        keep = binary_opening(Dp > dmin, structure=disk) | binary_opening(Dp < -dmin, structure=disk)
        keep = gaussian_filter(keep.astype(float), db / 2)
        F = F + Dp * keep * mask
        stats["detail_px"] = int((keep > 0.5).sum())
    return F, S_map, stats


def rolloff(x, k=0.85):
    """Soft shoulder for linear values above k (avoids hard-clipped whites)."""
    return np.where(x > k, k + (1 - k) * np.tanh((x - k) / (1 - k)), x)
