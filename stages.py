"""run.py's cmd_run, split at the node boundaries.

Each function is a line-for-line port of one section of the image-post skill's scripts/run.py (same names,
same order, same defaults) and works on the work box (roi) in float64, exactly like run.py. Keep it that way:
when run.py changes, port the diff here. `tools/sync_imgpost.py --check` says when run.py or imgpost/ moved on.

run.py's sys.exit() conditions raise StageError with the same message.
"""
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

from .imgpost import autofit as af
from .imgpost import fill as fl
from .imgpost import finish as fn
from .imgpost import geometry as geo
from .imgpost import grade as gr
from .imgpost import images as im
from .imgpost import masks as mk
from .imgpost import outline as ol
from .imgpost import psd
from .imgpost import qa
from .imgpost import warp as wp


class StageError(RuntimeError):
    """The job needs fixing; the message says how (the same text run.py exits with)."""


def check_job(job):
    todo = [k for k in ("roi", "old_silhouette") if not job.get(k)]
    if len(job.get("align", {}).get("points", [])) < 3:
        todo.append("align.points (3 or more)")
    if todo:
        raise StageError("job.json is missing: " + ", ".join(todo))


def work_box(job, Hs, Ws):
    return [max(0, int(round(job["roi"][0]))), max(0, int(round(job["roi"][1]))),
            min(Ws, int(round(job["roi"][2]))), min(Hs, int(round(job["roi"][3])))]


# 1. geometry: warp the reference into the work box

def fit_warp(job):
    """-> (H, align report)"""
    return geo.fit(job["align"])


def align_table(rep):
    """The residual table run.py prints after the fit."""
    lines = [f"align ({rep['model']}{' + residual' if 'residual' in rep else ''}), scene px"]
    for p in rep["points"]:
        lines.append(f"  point   {p['label'][:34]:34s} dx {p['dx']:6.2f}  dy {p['dy']:6.2f}")
    for key in ("y_only", "x_only"):
        for p in rep[key]:
            lines.append(f"  {key:7s} {p['label'][:34]:34s} d  {p['d']:6.2f}")
    for l in rep["lines"]:
        lines.append(f"  line    {l['label'][:34]:34s} max {l['max_px']:5.2f}  mean {l['mean_px']:6.2f}")
    r = rep.get("residual")
    if r:
        lines.append(f"residual  sigma {r['sigma']} px, max shift {r['max_shift_px']} px, {r['controls']} edge points"
                     + (f", {r['dropped_n']} stray left out ({', '.join(r['dropped'])})" if r["dropped_n"] else ""))
        lines.append("  after   " + ", ".join(f"{l['label'][:24]} {l['max_px']}" for l in r["lines_after"]))
    return "\n".join(lines)


def reference_mask(job, ref_u8, ref_a):
    return ol.RefMask(job["reference_outline"], ref_u8, ref_a)


def warp(job, H, refmask, scene_u8, ref_u8, roi):
    """-> (W_lin, a_new, (U, V) reference coords per work-box pixel, align overlay PIL image). H: a 3 x 3 matrix
    or a geometry mapping (cylinder, residual)."""
    x0, y0, x1, y1 = roi
    crop = im.to_float(scene_u8[y0:y1, x0:x1])
    ref_lin = im.srgb_to_lin(im.to_float(ref_u8))
    ref_flat = job.get("grade", {}).get("ref_flatten")
    if ref_flat:
        # a cylinder's label: the packshot's studio falloff toward its limbs comes out, the scene's light goes in
        ry, rx = np.mgrid[0:ref_u8.shape[0], 0:ref_u8.shape[1]].astype(float)
        inside = refmask.sample(rx, ry) > 0.5
        if hasattr(H, "axis_x"):                 # a cylinder's paper level is read between its limbs, not past them
            inside &= np.abs(rx - H.axis_x) <= H.r - 1.0
        ref_lin = gr.flatten(ref_lin, inside, "columns" if ref_flat == "columns" else "field", columns=hasattr(H, "axis_x"))
        del ry, rx
    pre = job.get("prefilter", 0.6)
    if pre == "auto":
        # anti-alias only when the scene shows the packshot smaller; enlarging needs no blur before sampling
        P = np.asarray(refmask.polygon, float) if refmask.polygon else np.array(refmask.bbox(ref_u8.shape)).reshape(2, 2)
        pre = 0.6 if af.local_scale(H, P.mean(0)) < 1.0 else 0.0
    W_lin, a_new, (U, V) = wp.warp(ref_lin, H, roi, refmask, ss=job.get("supersample", 4), prefilter=pre)
    occ = job.get("occluders", [])
    outline_scene = geo.map_polygon(H, refmask.polygon).tolist() if refmask.polygon else None
    overlay = qa.overlay_align(crop, im.lin_to_srgb(W_lin), a_new, roi, outline_scene,
                               [mk.occluder_polygon(roi, o) for o in occ])
    return W_lin, a_new, (U, V), overlay


def fringe(refmask, ref_u8, H=None):
    """-> (stretches, warning line or ""): where a measured outline takes in the packshot's plain backdrop, which
    composites as a light rim (run.py prints the same line after the fit)."""
    if not refmask.polygon or not af.on_plain_background(ref_u8):
        return [], ""
    fr = [f for f in af.outline_fringe(ref_u8, refmask.polygon, near=0.5) if f["px"] >= 5]
    if hasattr(H, "axis_x"):                      # a cylinder's outline runs past its limb on purpose (wrap): not a rim
        fr = [f for f in fr if min(abs(f["from"][0] - H.axis_x), abs(f["to"][0] - H.axis_x)) < H.r - 1.0]
    if not fr:
        return fr, ""
    return fr, (f"fringe     the outline takes in the packshot's backdrop on {len(fr)} stretch(es), worst "
                f"{fr[0]['worst']:+.2f} px from {fr[0]['from']} to {fr[0]['to']}: a light rim. "
                "measure.py fringe JOB, re-measure those sides (pitfall 45)")


def scene_masks(job, roi):
    """-> (old silhouette bool, visibility 0..1) over the work box"""
    vis = mk.visibility(roi, job.get("occluders", []))
    old = mk.poly_cover(roi, job["old_silhouette"], ss=4) > 0.5   # pixel centres inside, PIL edge bleed < 1/4 px
    return old, vis


# 2. light and colour from the generated product

def fit_masks(gs, a_new, old, vis):
    er = gs.get("erode", 4)
    inner = binary_erosion(old & (vis > 0.99), iterations=er)
    both = binary_erosion((a_new > 0.99) & old & (vis > 0.99), iterations=er)
    if both.sum() < 200:
        raise StageError("the real and generated products barely overlap: check align and old_silhouette")
    return inner, both


def ink_mode(matte):
    return matte == "ink" or (isinstance(matte, dict) and matte.get("type") == "ink")


def grade(gs, scene_u8, ref_u8, roi, W_lin, a_new, old, vis, U=None, matte=None):
    """-> (graded linear rgb, grade report, grade preview PIL image, relight map PIL image or None, ink or None).
    ref_u8 is only read for grade.ref_white_box, U (reference x per work-box pixel) only for grade.shading.
    grade.relight's `debug` (run.py saves the relight inputs to work/) is ignored here. With the job's matte 'ink',
    `ink` carries what run.py's composite reads: the white field (linear) and the ink's colour (linear)."""
    x0, y0, x1, y1 = roi
    crop = im.to_float(scene_u8[y0:y1, x0:x1])
    S_lin = im.srgb_to_lin(crop)
    Y, X = np.mgrid[y0:y1, x0:x1].astype(float)
    inner, both = fit_masks(gs, a_new, old, vis)
    gain = gs.get("gain", "white_level")
    if gain == "white_level":
        g = gr.white_level_gain(S_lin, inner, binary_erosion(inner, iterations=gs.get("deep_erode", 20)), X, Y,
                                gs.get("white_percentile", 92), gs.get("window", 41))
    elif gain == "luma_ratio":
        g = gr.luma_ratio_gain(S_lin, W_lin, both, X, Y, gs.get("sigma", 6.0), gs.get("match_hue"), gs.get("flat", 0.08))
    else:
        g = np.ones(S_lin.shape[:2])
    rep = {"gain": gain, "gain_range": [round(float(g[both].min()), 3), round(float(g[both].max()), 3)]}
    if gs.get("colour", "scene") == "reference":
        # keep the real product's colours; only light, white balance and black level come from the scene
        def box_lin(u8, box):
            bx0, by0, bx1, by1 = [int(round(v)) for v in box]
            return np.median(im.srgb_to_lin(im.to_float(u8[by0:by1, bx0:bx1])).reshape(-1, 3), 0)
        if gs.get("white_field") not in (None, False):
            # the generated label's own paper, per pixel: the scene's light with its falloff and tint (bounce light)
            wf = gs["white_field"] if isinstance(gs["white_field"], dict) else {}
            ys_, xs_ = np.nonzero(old)
            size = int(wf.get("size", 2 * max(2, round(min(np.ptp(ys_), np.ptp(xs_)) / 24)) + 1))
            white = gr.paper_field(S_lin, inner, size, wf.get("sigma", max(1.0, size / 10)))
            white = white * wf.get("scale", 1.0)
            rep["white_field"] = {"size": size}
        elif gs.get("white_box"):
            shape = {"luma_ratio": lambda: gr.luma_ratio_gain(S_lin, W_lin, both, X, Y, gs.get("sigma", 6.0),
                                                              gs.get("match_hue"), gs.get("flat", 0.08)),
                     "white_level": lambda: g / np.median(g[both]),
                     "none": lambda: np.ones(S_lin.shape[:2])}[gs.get("falloff", "luma_ratio")]()
            white = box_lin(scene_u8, gs["white_box"])[None, None, :] * shape[..., None] * gs.get("white_scale", 1.0)
        elif gain == "white_level":
            white = g[..., None] * gr.white_chroma(S_lin, inner, g)[None, None, :]
        else:
            raise StageError("grade.colour 'reference' needs grade.white_box, or gain 'white_level' on a product with real white areas")
        black = box_lin(scene_u8, gs["black_box"]) if gs.get("black_box") else \
            np.minimum(np.percentile(S_lin[both], 1, axis=0), gs.get("black_cap", 0.02))
        ref_flat = gs.get("ref_flatten")
        if gs.get("ref_white_box") and not ref_flat and ref_u8 is None:
            raise StageError("grade.ref_white_box reads the reference image: connect `reference` on the grade node")
        ref_white = np.ones(3) if ref_flat else box_lin(ref_u8, gs["ref_white_box"]) if gs.get("ref_white_box") else np.ones(3)
        graded = gr.keep_reference_colour(W_lin, white, black, ref_white)
        rep.update({"colour": "reference", "white_point_lin": [round(float(v), 4) for v in np.median(white[both], 0)],
                    "black_point_lin": [round(float(v), 4) for v in black],
                    "ref_white_lin": [round(float(v), 4) for v in ref_white]})
    else:
        curves = gr.tone_curves(S_lin, W_lin, g, both)
        graded = gr.apply_grade(W_lin, g, curves)
        rep.update({"colour": "scene", "curves_rgb": [{k: v for k, v in c.items() if k != "_p"} for c in curves]})
        pw = gs.get("protect_white")
        if pw:
            # near-clipped packshot whites carry none of its light falloff: never dim them below the median light
            lo, hi = pw if isinstance(pw, (list, tuple)) else (0.88, 0.98)
            w = gr.clip_weight(W_lin, lo, hi)
            gm = float(np.median(g[both]))
            graded = graded * (1.0 + w * (np.maximum(g, gm) / g - 1.0))[..., None]
            rep["protect_white"] = {"range": [lo, hi], "px": int((w[both] > 0.5).sum())}

    sh = gs.get("shading")
    if sh:
        if U is None:
            raise StageError("grade.shading needs the warp's reference coordinates: connect Warp Reference's coords to the grade node")
        # residual shading across the product (cylinders): keeps the generated can's round light, not its print
        centers, prof = gr.shading_profile(S_lin, graded, U, both, sh.get("bins", 32), sh.get("smooth", 1.0), sh.get("match_hue"))
        graded = gr.apply_profile(graded, U, centers, prof, sh.get("strength", 1.0), tuple(sh.get("clip", (0.4, 2.5))))
        rep["shading_profile"] = {"range": [round(float(np.exp(prof.min())), 3), round(float(np.exp(prof.max())), 3)],
                                  "bins": len(prof)}
        if sh.get("x_range") and (sh.get("limb") or sh.get("spec") or sh.get("shadow")):
            graded = gr.cylinder_light(graded, U, sh["x_range"], sh.get("limb", 0.0), sh.get("spec"), sh.get("white", 0.9), sh.get("shadow"))

    rl = gs.get("relight")
    sheet = None
    if rl:
        # the source's light the smooth grade missed: cast shadows, falloff, glows (+ optional glossy sheen)
        R, SH, rinfo = gr.relight_map(S_lin, graded, both, W_lin=W_lin,
                                      **{k: v for k, v in rl.items() if k not in ("about", "strength", "clip", "debug")})
        fmul = np.clip(np.exp(R), *tuple(rl.get("clip", (0.25, 3.0))))
        # veil (white, equal on all channels): lit gloss is part of the material's look and takes the light;
        # unlit haze is a reflection and goes on top
        if SH is not None and rinfo.get("veil") == "unlit":
            relit = graded * fmul[..., None] + SH[..., None]
        else:
            relit = (graded + (SH[..., None] if SH is not None else 0.0)) * fmul[..., None]
        st = float(rl.get("strength", 1.0))
        graded = graded + st * (relit - graded)
        rep["relight"] = dict(rinfo, range=[round(float(fmul[both].min()), 3), round(float(fmul[both].max()), 3)],
                              sheen_max=round(float(SH[both].max()), 3) if SH is not None else None)
        sheet = qa.relight_sheet(crop, fmul, SH, both)

    def med(img):
        return [int(v) for v in np.round(np.median(im.lin_to_srgb(img[both]), 0) * 255)]

    rep["median_srgb_overlap"] = {"scene": med(S_lin), "graded_reference": med(graded)}
    prev = im.lin_to_srgb(graded) * a_new[..., None] + crop * (1 - a_new[..., None])
    ink = None
    if ink_mode(matte):
        if gs.get("colour") != "reference" or not gs.get("ref_flatten"):
            raise StageError("matte 'ink' needs grade.colour 'reference' and grade.ref_flatten")
        mcfg = matte if isinstance(matte, dict) else {}
        Lw = im.lum(W_lin)
        ink_level = float(mcfg.get("ink", np.percentile(Lw[a_new > 0.5], 0.5)))
        ink = {"white": white, "ink_lin": gr.keep_reference_colour(np.full_like(W_lin, ink_level), white, black, ref_white),
               "ink_level": ink_level, "floor": float(mcfg.get("floor", 0.06))}
    return graded, rep, qa.before_after(scene_u8[y0:y1, x0:x1], im.to_u8(prev)), sheet, ink


# 3. finish: highlight roll-off, softness and grain matched to the frame

def finish(fs, gs, scene_u8, roi, graded, a_new, old, vis):
    """-> (finished product, sRGB float, work box; finish report)"""
    x0, y0, x1, y1 = roi
    crop = im.to_float(scene_u8[y0:y1, x0:x1])
    _, both = fit_masks(gs, a_new, old, vis)
    card = im.lin_to_srgb(gr.rolloff(graded, fs.get("rolloff", 0.85)))
    Ls = im.lum(crop)
    probe = fs.get("blur_probe")
    pb = (probe[0] - x0, probe[1] - y0, probe[2] - x0, probe[3] - y0) if probe else fn.auto_probe(Ls, both)
    s_scene, s_card = fn.edge_sigma(Ls, pb), fn.edge_sigma(im.lum(card), pb)
    sharp = None
    if fs.get("sharpen") == "auto" and np.isfinite(s_scene + s_card) and s_card > 1.05 * s_scene:
        # a packshot softer than the frame (a small screenshot enlarged): unsharp mask to the scene's edge blur
        card, r_sh, amt = fn.sharpen_to(card, pb, s_scene, im.lum, a_new)
        sharp = {"radius": round(r_sh, 2), "amount": round(amt, 3), "edge_sigma_before": round(s_card, 3)}
        s_card = fn.edge_sigma(im.lum(card), pb)
    auto_blur = float(np.sqrt(max(s_scene ** 2 - s_card ** 2, 0.0))) if np.isfinite(s_scene + s_card) else 0.0
    blur = auto_blur if fs.get("blur", "auto") == "auto" else float(fs["blur"])
    card = fn.blur_rgb(card, blur)
    grain = fs.get("grain", "auto")
    if grain == "auto":
        flat = fn.flat_mask(Ls, both)
        add = np.sqrt(np.clip(fn.mad_std(fn.highpass(crop)[flat]) ** 2 - fn.mad_std(fn.highpass(card)[flat]) ** 2, 0, None))
    else:
        add = np.full(3, float(grain))
    if add.max() > 0.002:
        card = card + fn.make_grain(card.shape[:2], add, a_new > 0.5, seed=11)
    rep = {"blur_px": round(blur, 3), "blur_auto_estimate": round(auto_blur, 3),
           "edge_sigma_scene": round(s_scene, 3), "edge_sigma_reference": round(s_card, 3),
           "blur_probe": [pb[0] + x0, pb[1] + y0, pb[2] + x0, pb[3] + y0],
           "grain_added": [round(float(v), 4) for v in add]}
    if sharp:
        rep["sharpen"] = sharp
    return card, rep


# 4. fill the background where the generated product showed and the real one doesn't

def fill(fls, scene_u8, roi, a_new, old, vis, matte=None):
    """-> (background sRGB float, work box; fill mask F; fill report). Raises StageError (FILL STOPPED)."""
    x0, y0, x1, y1 = roi
    crop = im.to_float(scene_u8[y0:y1, x0:x1])
    dil = int(fls.get("dilate", 2))
    old_dil = binary_dilation(old, iterations=dil) if dil > 0 else old.copy()   # scipy: iterations 0 = until it fills everything
    visb = vis > 0.5
    F = old_dil & visb & ((a_new < 0.98) | ink_mode(matte))    # an ink matte covers only its ink: clear all generated print
    D = binary_dilation(F) & ~old_dil & visb
    # specks boxed in between the product and an occluder (a few px) take their colour from whatever borders them
    orphans = fl.anchor_orphans(F, D, int(fls.get("orphan_px", 16)))
    if orphans:
        D = D | orphans["ring"]
    try:
        base, nfill = fl.harmonic_fill(crop, F, D, offset=(x0, y0))
    except fl.FillError as e:
        raise StageError(f"FILL STOPPED, leftover regions with no background around them: {e}\n"
                         "Fix: extend reference_outline under the occluder ('extend') or correct the occluder polyline, then re-run.")
    ring = binary_dilation(D, iterations=3) & ~old_dil & visb
    gstd = fn.mad_std(fn.highpass(crop)[ring]) if ring.any() else np.zeros(3)
    if nfill:
        base[F] = np.clip(base[F] + fn.make_grain(F.shape, gstd, F, seed=7)[F], 0, 1)
    rep = {"pixels": int(nfill), "grain_std": [round(float(v), 4) for v in gstd]}
    if orphans:
        rep["orphans"] = {"regions": orphans["n"], "px": orphans["px"]}
    return base, F, rep


# 5. composite behind the occluders

def composite(fs, base, card, a_new, vis, ink=None, matte=None):
    """-> (work box uint8, matte). With the job's matte 'ink', `ink` is the grade stage's ink output."""
    alpha = np.clip(gaussian_filter(a_new, fs.get("edge_softness", 0.45)), 0, 1) * vis
    if ink_mode(matte):
        # print only: the packshot's ink as coverage in the ink's colour, over the scene's own surface (the fill has
        # taken the generated print away), so a patch of packshot background never shows its edge
        if ink is None:
            raise StageError("the job's matte 'ink' needs the grade's ink: connect Grade To Scene's ink to Composite Behind")
        wL, kL = im.lum(ink["white"]), im.lum(ink["ink_lin"])
        cov = (wL - im.lum(im.srgb_to_lin(np.clip(card, 0, 1)))) / np.maximum(wL - kL, 1e-4)
        t0 = ink["floor"]                                    # paper grain and grain added stay out of the matte
        cov = np.clip((cov - t0) / (1 - t0), 0, 1)
        card = im.lin_to_srgb(ink["ink_lin"])
        alpha = alpha * cov
    out = base * (1 - alpha[..., None]) + card * alpha[..., None]
    if not np.isfinite(out).all():
        raise StageError("composite contains NaN or inf: nothing saved")
    return im.to_u8(out), alpha


# 6. checks and QA sheets

def change_report(res, scene_u8, roi):
    x0, y0, x1, y1 = roi
    chg = np.abs(res.astype(np.int16) - scene_u8.astype(np.int16)).max(2) > 0
    inside = np.zeros_like(chg)
    inside[y0:y1, x0:x1] = True
    ys, xs = np.nonzero(chg)
    return {"changed_px": int(chg.sum()), "pct_of_frame": round(100 * float(chg.mean()), 3),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None,
            "outside_roi_changed": int((chg & ~inside).sum())}


def qa_sheets(res, scene_u8, ref_u8, refmask, alpha, F, roi):
    """-> (before_after, compare, edges or None) PIL images"""
    x0, y0, x1, y1 = roi
    before, after = scene_u8[y0:y1, x0:x1], res[y0:y1, x0:x1]
    rx0, ry0, rx1, ry1 = refmask.bbox(ref_u8.shape, pad=10)
    tiles = qa.edge_tiles(before, after, ((alpha > 0.03) & (alpha < 0.97)) | F, (x0, y0))
    return qa.before_after(before, after), qa.compare_sheet(before, after, ref_u8[ry0:ry1, rx0:rx1]), tiles


def write_psd(path, scene_u8, res, base, F, card, alpha, roi):
    """run.py --psd: the layered PSD (original scene, background fill, product + matte x occluders as its mask).
    No ICC profile, dpi or alpha: ComfyUI's IMAGE doesn't carry them. -> report.json's psd entry."""
    return psd.write_refined(path, scene_u8, None, {}, res, base, F, card, alpha, roi)
