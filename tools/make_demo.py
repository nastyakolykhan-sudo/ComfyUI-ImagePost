#!/usr/bin/env python3
"""Build the demo: a made-up product card, a generated-looking scene showing a garbled copy of it, and the job.

  python3 -P tools/make_demo.py [--out examples/demo]

Writes three files:
  imagepost_demo_packshot.png  the real product: the card's clean print on a plain background
  imagepost_demo_scene.png     the "generated" still: the same card with garbled text, another emblem and grid,
                               square corners and a taller top, lit from the right, a mug in front of its lower
                               right corner casting a shadow across it, softened and grained
  job.json                     the measurements that fix it
Everything is drawn from code with fixed seeds, so the demo shows no real product or brand. The job's landmarks
and outlines come from the geometry that placed the card; on a real image they are measured on the pixels.
Needs numpy, scipy and Pillow 10.1 or later.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, map_coordinates

PACK = Path(__file__).resolve().parent.parent
SS = 4                                   # supersampling for everything drawn
REF_W, REF_H = 600, 800                  # packshot
SCENE_W, SCENE_H = 1024, 768             # generated still
CARD = (40, 40, 560, 760)                # the real card, packshot px
GEN_TOP = 28                             # the generated card's top (taller than the real one)
RADIUS = 28                              # the real card's corner radius; the generated one has square corners
HEADER_Y1, BOTTOM_Y0 = 230, 690          # header band bottom, bottom band top (same in both prints)
EMBLEM = (300, 430)
CORNERS_SCENE = [(372, 170), (612, 186), (610, 566), (370, 578)]   # real card corners TL, TR, BR, BL in the scene
TABLE_Y = 560
MUG = {"x0": 592, "x1": 706, "top": 438, "bottom": 652, "ry": 13}
ROI = [330, 120, 660, 620]


# ---- colour and geometry helpers ----

def srgb_to_lin(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(c, 0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def rgb(*v):
    return np.array(v, float) / 255.0


def homography(src, dst):
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b += [u, v]
    h = np.linalg.solve(np.array(A, float), np.array(b, float))
    return np.append(h, 1.0).reshape(3, 3)


def apply_h(H, pts):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:]


def r2(p):
    return [round(float(p[0]), 2), round(float(p[1]), 2)]


def star(cx, cy, r_out, r_in, n, rot=-90.0):
    t = np.radians(rot + np.arange(2 * n) * 180.0 / n)
    r = np.where(np.arange(2 * n) % 2 == 0, r_out, r_in)
    return list(zip(cx + r * np.cos(t), cy + r * np.sin(t)))


def font(size):
    return ImageFont.load_default(size=size * SS)


# ---- the two prints ----

def draw_print(version):
    """-> (print sRGB float REF_H x REF_W x 3 on its background, card coverage 0..1)"""
    real = version == "real"
    big = (REF_W * SS, REF_H * SS)
    S = lambda *v: [int(round(c * SS)) for c in v]
    top = CARD[1] if real else GEN_TOP
    content = Image.new("RGB", big, (250, 250, 247))
    d = ImageDraw.Draw(content)
    teal, orange, navy = ((31, 122, 140), (232, 116, 59), (35, 57, 91)) if real else \
        ((36, 128, 146), (238, 122, 62), (40, 62, 96))
    d.rectangle(S(CARD[0], top, CARD[2], HEADER_Y1), fill=teal)
    d.rectangle(S(CARD[0], BOTTOM_Y0, CARD[2], CARD[3]), fill=navy)
    word, wsize, wpos = ("PRODUCT", 92, (300, 139)) if real else ("PRDOUCT", 86, (296, 137))
    d.text(S(*wpos), word, font=font(wsize), fill=(255, 255, 255), anchor="mm", stroke_width=2 * SS, stroke_fill=(255, 255, 255))
    tag, tsize, tpos = ("sample label", 34, (300, 268)) if real else ("sapmle lable", 33, (303, 270))
    d.text(S(*tpos), tag, font=font(tsize), fill=(60, 60, 60), anchor="mm")
    cx, cy = EMBLEM
    r_em = 92 if real else 96
    d.ellipse(S(cx - r_em, cy - r_em, cx + r_em, cy + r_em), fill=orange)
    pts = star(cx, cy, 62, 25, 5) if real else star(cx, cy, 64, 36, 6, rot=-60)
    d.polygon([tuple(S(x, y)) for x, y in pts], fill=(255, 255, 255))
    if real:
        cells = [(95 + i * 110, y0, 80, 55) for y0 in (545, 612) for i in range(4)]
    else:
        cells = [(103 + i * 134, y0, 104, 52) for y0 in (548, 614) for i in range(3)]
    for k, (x, y, w, h) in enumerate(cells):
        cols = len(cells) // 2
        d.rounded_rectangle(S(x, y, x + w, y + h), radius=8 * SS, fill=teal if (k + k // cols) % 2 == 0 else orange)
    net, nsize = ("NET 12 PCS", 30) if real else ("NET 21 PSC", 29)
    d.text(S(300, 725), net, font=font(nsize), fill=(255, 255, 255), anchor="mm")

    mask = Image.new("L", big, 0)
    ImageDraw.Draw(mask).rounded_rectangle(S(CARD[0], top, CARD[2], CARD[3]), radius=(RADIUS if real else 0) * SS, fill=255)
    out = Image.composite(content, Image.new("RGB", big, (242, 242, 242)), mask)
    return (np.asarray(out.reduce(SS), float) / 255.0, np.asarray(mask.reduce(SS), float) / 255.0)


# ---- the scene ----

def coverage(draw_fn):
    """Anti-aliased coverage of shapes drawn at SS x scene size."""
    m = Image.new("L", (SCENE_W * SS, SCENE_H * SS), 0)
    draw_fn(ImageDraw.Draw(m), lambda *v: [int(round(c * SS)) for c in v])
    return np.asarray(m.reduce(SS), float) / 255.0


def mug_outline(n=24):
    """Mug body silhouette (no handle): left side, bottom arc, right side, top arc; scene px."""
    x0, x1, top, bottom, ry = MUG["x0"], MUG["x1"], MUG["top"], MUG["bottom"], MUG["ry"]
    cx, rx = (x0 + x1) / 2, (x1 - x0) / 2
    t = np.linspace(np.pi, 0, n)
    bottom_arc = [(cx + rx * np.cos(a), bottom + ry * np.sin(a)) for a in t]
    top_arc = [(cx + rx * np.cos(a), top - ry * np.sin(a)) for a in np.linspace(0, np.pi, n)]
    return [(x0, top)] + bottom_arc + [(x1, top)] + top_arc[1:]


def build_scene(gen_print, gen_cover, H):
    Y, X = np.mgrid[0:SCENE_H, 0:SCENE_W].astype(float)
    side = 0.90 + 0.14 * X / SCENE_W                                   # key light from the right
    wall = rgb(205, 198, 188) + (rgb(188, 180, 168) - rgb(205, 198, 188)) * (Y / TABLE_Y)[..., None]
    table = rgb(140, 106, 80) + (rgb(112, 84, 62) - rgb(140, 106, 80)) * ((Y - TABLE_Y) / (SCENE_H - TABLE_Y))[..., None]
    k = np.clip((Y - TABLE_Y + 1.0) / 2.0, 0, 1)[..., None]            # 2 px edge between wall and table
    lin = srgb_to_lin(wall * (1 - k) + table * k) * side[..., None]

    bl, br = CORNERS_SCENE[3], CORNERS_SCENE[2]                       # contact shadow under the card
    contact = gaussian_filter(coverage(lambda d, S: d.polygon(
        [tuple(S(*bl)), tuple(S(*br)), tuple(S(br[0] + 4, br[1] + 9)), tuple(S(bl[0] - 4, bl[1] + 11))], fill=255)), 4)
    lin *= (1 - 0.35 * contact)[..., None]

    # the generated card, warped in with 3 x 3 samples per pixel
    x0, y0, x1, y1 = 340, 140, 640, 600
    n = 3
    off = (np.arange(n) + 0.5) / n - 0.5
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(float)
    Hi = np.linalg.inv(H)
    prem, cov = np.zeros(ys.shape + (3,)), np.zeros(ys.shape)
    plin = srgb_to_lin(gen_print)
    for dy in off:
        for dx in off:
            q = np.c_[(xs + dx).ravel(), (ys + dy).ravel(), np.ones(xs.size)] @ Hi.T
            u, v = (q[:, 0] / q[:, 2]).reshape(xs.shape), (q[:, 1] / q[:, 2]).reshape(xs.shape)
            a = map_coordinates(gen_cover, [v, u], order=1, mode="constant", cval=0.0)
            cov += a
            for c in range(3):
                prem[..., c] += a * map_coordinates(plin[..., c], [v, u], order=1, mode="nearest")
    prem, cov = prem / n ** 2, cov / n ** 2
    t = np.clip((xs - 370) / 242, 0, 1)
    light = (0.86 + 0.20 * t) * (1.0 - 0.07 * np.clip((ys - 170) / 408, 0, 1))
    box = lin[y0:y1, x0:x1]
    lin[y0:y1, x0:x1] = box * (1 - cov[..., None]) + prem * light[..., None]

    # the mug's shadow falls left, across the card's lower right corner and the wall behind it
    shadow = gaussian_filter(coverage(lambda d, S: d.ellipse(S(527, 420, 623, 600), fill=255)), 7)
    lin *= (1 - 0.42 * shadow)[..., None]

    # the mug: a red cylinder lit from the right, dark inside, handle on the right
    body = coverage(lambda d, S: d.polygon([tuple(S(x, y)) for x, y in mug_outline()], fill=255))
    handle = coverage(lambda d, S: (d.ellipse(S(690, 478, 748, 600), fill=255), d.ellipse(S(704, 494, 734, 584), fill=0)))
    handle *= (X > MUG["x1"] - 2)
    inside = coverage(lambda d, S: d.ellipse(S(MUG["x0"] + 6, MUG["top"] - MUG["ry"] + 3,
                                               MUG["x1"] - 6, MUG["top"] + MUG["ry"] - 3), fill=255))
    tm = np.clip((X - MUG["x0"]) / (MUG["x1"] - MUG["x0"]), 0, 1)
    shade = 0.30 + 0.85 * tm ** 1.6 + 0.9 * np.exp(-((tm - 0.80) / 0.045) ** 2)
    red = srgb_to_lin(rgb(150, 42, 48))
    mug = red * shade[..., None]
    mug = mug * (1 - inside[..., None]) + srgb_to_lin(rgb(58, 18, 20)) * inside[..., None]
    lin = lin * (1 - body[..., None]) + mug * body[..., None]
    lin = lin * (1 - handle[..., None]) + (red * 0.9) * handle[..., None]

    srgb = lin_to_srgb(lin)
    srgb = np.stack([gaussian_filter(srgb[..., c], 0.75) for c in range(3)], -1)   # the generator's softness
    srgb += np.random.default_rng(7).normal(0, 1.3 / 255, srgb.shape)               # and its grain
    return (np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)


# ---- the job ----

def build_job(H):
    def h(pts):
        return [r2(p) for p in apply_h(H, pts)]

    left = h([(40, y) for y in np.linspace(100, 700, 8)])
    right_ys = [y for y in np.linspace(100, 700, 25) if apply_h(H, [(560, y)])[0][1] < MUG["top"] - MUG["ry"] - 8]
    right = h([(560, y) for y in right_ys])
    points = [("header band, bottom left corner", (40, HEADER_Y1)), ("header band, bottom right corner", (560, HEADER_Y1)),
              ("bottom band, top left corner", (40, BOTTOM_Y0)), ("emblem centre", EMBLEM)]
    return {
        "about": "Demo (tools/make_demo.py): a made-up product card whose print came out garbled in a generated "
                 "still, with a mug in front of its lower right corner and the mug's shadow across it. The landmarks "
                 "and outlines come from the geometry that placed the card; on a real image they are measured.",
        "scene": "imagepost_demo_scene.png",
        "reference": "imagepost_demo_packshot.png",
        "output": "imagepost_demo_scene_refined.png",
        "scene_size": [SCENE_W, SCENE_H],
        "reference_size": [REF_W, REF_H],
        "roi": ROI,
        "reference_outline": {"type": "rounded_rect", "x0": CARD[0], "y0": CARD[1], "x1": CARD[2], "y1": CARD[3],
                              "r_top": RADIUS, "r_bottom": RADIUS},
        "align": {
            "model": "homography",
            "points": [{"label": label, "ref": list(p), "scene": r2(apply_h(H, [p])[0]), "w": 2} for label, p in points],
            "y_only": [],
            "lines": [
                {"label": "left card edge", "ref": [[40, 100], [40, 700]], "n": 12, "w": 2, "scene": left},
                {"label": "right card edge, above the mug", "ref": [[560, 100], [560, round(right_ys[-1], 1)]],
                 "n": 8, "w": 2, "scene": right},
            ],
        },
        "old_silhouette": h([(CARD[0], GEN_TOP), (CARD[2], GEN_TOP), (CARD[2], CARD[3]), (CARD[0], CARD[3])]),
        "occluders": [{"label": "mug in front of the lower right corner",
                       "polygon": [r2(p) for p in mug_outline()]}],
        "grade": {"gain": "white_level", "white_percentile": 92, "window": 41, "erode": 4, "deep_erode": 20,
                  "colour": "reference", "ref_white_box": [60, 290, 160, 330],
                  "relight": {"sigma": 14}},
        "finish": {"blur": "auto", "blur_probe": None, "grain": "auto", "edge_softness": 0.6, "rolloff": 0.85},
        "fill": {"dilate": 2},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(PACK / "examples" / "demo"))
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    H = homography([(CARD[0], CARD[1]), (CARD[2], CARD[1]), (CARD[2], CARD[3]), (CARD[0], CARD[3])], CORNERS_SCENE)
    real, _ = draw_print("real")
    gen, gen_cover = draw_print("generated")
    Image.fromarray((real * 255 + 0.5).astype(np.uint8)).save(out / "imagepost_demo_packshot.png", optimize=True)
    Image.fromarray(build_scene(gen, gen_cover, H)).save(out / "imagepost_demo_scene.png", optimize=True)
    (out / "job.json").write_text(json.dumps(build_job(H), indent=2) + "\n")
    for name in ("imagepost_demo_packshot.png", "imagepost_demo_scene.png", "job.json"):
        print(f"wrote {out / name}  {(out / name).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
