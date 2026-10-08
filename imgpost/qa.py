"""QA sheets: gridded zooms for measuring, the alignment overlay, before/after, comparison and edge tiles."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .images import to_u8


def font(size=12):
    return ImageFont.load_default(size=size)


def grid_tile(img_u8, box, scale, step, label=None):
    """Crop box (x0, y0, x1, y1), zoom by scale, draw a labelled grid whose lines sit on pixel centres."""
    H, W = img_u8.shape[:2]
    x0, y0, x1, y1 = max(int(box[0]), 0), max(int(box[1]), 0), min(int(box[2]), W), min(int(box[3]), H)
    im = Image.fromarray(np.ascontiguousarray(img_u8[y0:y1, x0:x1]))
    im = im.resize((int((x1 - x0) * scale), int((y1 - y0) * scale)), Image.NEAREST if scale >= 4 else Image.LANCZOS)
    d, f = ImageDraw.Draw(im), font(12)
    for gx in range((x0 // step + 1) * step, x1, step):
        X = (gx - x0 + 0.5) * scale
        d.line([(X, 0), (X, im.height)], fill=(0, 255, 255))
        d.text((X + 2, 14), str(gx), fill=(0, 220, 255), font=f)
    for gy in range((y0 // step + 1) * step, y1, step):
        Y = (gy - y0 + 0.5) * scale
        d.line([(0, Y), (im.width, Y)], fill=(255, 0, 255))
        d.text((2, Y + 2), str(gy), fill=(255, 0, 255), font=f)
    if label:
        d.rectangle([0, 0, im.width - 1, 13], fill=(0, 0, 0))
        d.text((3, 0), label, fill=(255, 255, 0), font=f)
    return im


def montage(tiles, max_w=2000, gap=8, bg=(40, 40, 40)):
    rows, cur, w = [], [], 0
    for t in tiles:
        if cur and w + t.width > max_w:
            rows.append(cur)
            cur, w = [], 0
        cur.append(t)
        w += t.width + gap
    if cur:
        rows.append(cur)
    W = max(sum(t.width + gap for t in r) for r in rows)
    H = sum(max(t.height for t in r) + gap for r in rows)
    M, y = Image.new("RGB", (W, H), bg), 0
    for r in rows:
        x = 0
        for t in r:
            M.paste(t, (x, y))
            x += t.width + gap
        y += max(t.height for t in r) + gap
    return M


def stretch(img_u8, box, mode="faint", lo=238.0, gain=5.0):
    """Reveal faint structure inside box. faint/contrast: luminance [lo, 255] -> full range. chroma: colour x gain."""
    x0, y0, x1, y1 = box
    c = img_u8[y0:y1, x0:x1].astype(float)
    if mode in ("faint", "contrast"):
        v = np.clip((c @ np.array([0.2126, 0.7152, 0.0722]) - lo) / (255 - lo), 0, 1) * 255
        v = np.repeat(v[..., None], 3, 2)
    elif mode == "chroma":
        L = c.mean(2, keepdims=True)
        s = L + (c - L) * gain
        a, b = np.percentile(s, 1), np.percentile(s, 99)
        v = np.clip((s - a) / (b - a + 1e-9), 0, 1) * 255
    else:
        raise ValueError(f"stretch mode must be faint, contrast or chroma, got {mode!r}")
    out = img_u8.copy()
    out[y0:y1, x0:x1] = v.astype(np.uint8)
    return out


def labelled(im, text, h=30, size=18):
    out = Image.new("RGB", (im.width, im.height + h), (24, 24, 24))
    out.paste(im, (0, h))
    ImageDraw.Draw(out).text((6, (h - size) // 2), text, fill=(235, 235, 235), font=font(size))
    return out


def side_by_side(images, gap=10, bg=(24, 24, 24)):
    W = sum(i.width for i in images) + gap * (len(images) - 1)
    M, x = Image.new("RGB", (W, max(i.height for i in images)), bg), 0
    for i in images:
        M.paste(i, (x, 0))
        x += i.width + gap
    return M


def overlay_align(crop_f, warped_srgb, alpha, roi, outline_scene, occluder_polys, scale=2):
    """Reference at 50% over the scene; real outline in green, occluder boundaries in red."""
    x0, y0 = roi[0], roi[1]
    blend = crop_f * (1 - 0.5 * alpha[..., None]) + warped_srgb * 0.5 * alpha[..., None]
    im = Image.fromarray(to_u8(blend))
    im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    d = ImageDraw.Draw(im)

    def P(pts):
        return [((x - x0 + 0.5) * scale, (y - y0 + 0.5) * scale) for x, y in pts]

    if outline_scene:
        p = P(outline_scene)
        d.line(p + [p[0]], fill=(0, 255, 0), width=1)
    for poly in occluder_polys:
        p = P(poly)
        d.line(p + [p[0]], fill=(255, 40, 40), width=1)
    return im


def before_after(before_u8, after_u8, scale=2):
    def z(a):
        im = Image.fromarray(np.ascontiguousarray(a))
        return im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    return side_by_side([labelled(z(before_u8), "before"), labelled(z(after_u8), "after")])


def compare_sheet(before_u8, after_u8, ref_u8, h=1000):
    imgs = []
    for arr, t in ((before_u8, "Generated (before)"), (after_u8, "Refined (after)"), (ref_u8, "Reference")):
        im = Image.fromarray(np.ascontiguousarray(arr))
        imgs.append(labelled(im.resize((max(1, int(im.width * h / im.height)), h), Image.LANCZOS), t))
    return side_by_side(imgs, gap=16)


def edge_tiles(before_u8, after_u8, boundary, offset, n=8, win=40, scale=4):
    """n before|after tiles at 4x spread along the boundary pixels (product edge, occluder edges, fill)."""
    ys, xs = np.nonzero(boundary)
    if not len(ys):
        return None
    h, w = boundary.shape
    win = min(win, h, w)
    order = np.argsort(np.arctan2(ys - ys.mean(), xs - xs.mean()))
    tiles = []
    for k in order[np.linspace(0, len(order), n, endpoint=False).astype(int)]:
        ya = int(np.clip(ys[k] - win // 2, 0, h - win))
        xa = int(np.clip(xs[k] - win // 2, 0, w - win))
        pair = [Image.fromarray(np.ascontiguousarray(a[ya:ya + win, xa:xa + win])).resize((win * scale, win * scale), Image.NEAREST)
                for a in (before_u8, after_u8)]
        tiles.append(labelled(side_by_side(pair, gap=4), f"x {xa + offset[0]}, y {ya + offset[1]}  before | after", h=20, size=13))
    return montage(tiles, max_w=1400)


def relight_sheet(crop_f, fmul, sheen, mask, scale=1):
    """Where the relight changed the light: blue = darker, red = brighter (0.5x..2x), sheen in white."""
    lg = np.clip(np.log2(np.maximum(fmul, 1e-3)), -1, 1)
    rgb = np.ones(fmul.shape + (3,)) * 0.5
    rgb[..., 0] = 0.5 + 0.5 * np.clip(lg, 0, 1) - 0.25 * np.clip(-lg, 0, 1)
    rgb[..., 2] = 0.5 + 0.5 * np.clip(-lg, 0, 1) - 0.25 * np.clip(lg, 0, 1)
    rgb[..., 1] = 0.5 - 0.25 * np.abs(lg)
    rgb = np.where(mask[..., None], rgb, crop_f * 0.35)
    tiles = [labelled(Image.fromarray(to_u8(crop_f)), "source"), labelled(Image.fromarray(to_u8(rgb)), "light change: blue darker, red brighter")]
    if sheen is not None:
        sv = np.where(mask, np.clip(sheen / max(float(sheen[mask].max()), 1e-3), 0, 1), 0.0)
        tiles.append(labelled(Image.fromarray(to_u8(np.repeat(sv[..., None], 3, 2))), "sheen (white = most)"))
    return side_by_side(tiles)
