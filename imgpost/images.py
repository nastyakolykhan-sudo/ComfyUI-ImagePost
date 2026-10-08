"""Image I/O and colour helpers. Full frames stay uint8; only the work box (roi) becomes float."""
from pathlib import Path

import numpy as np
from PIL import Image

LUMA = np.array([0.2126, 0.7152, 0.0722])


def resolve(p, base=None):
    """Expand ~ and resolve relative paths against base (the job file's folder)."""
    p = Path(str(p)).expanduser()
    if not p.is_absolute() and base is not None:
        p = Path(base) / p
    return p


def load_u8(p):
    """-> (rgb uint8 HxWx3, alpha uint8 HxW or None, PIL info dict)"""
    im = Image.open(p)
    info = dict(im.info)
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        a = np.asarray(im.convert("RGBA"))
        return a[..., :3].copy(), a[..., 3].copy(), info
    return np.asarray(im.convert("RGB")).copy(), None, info


def save_u8(p, rgb, alpha=None, info=None):
    arr = rgb if alpha is None else np.dstack([rgb, alpha])
    kw = {}
    if info and info.get("icc_profile"):
        kw["icc_profile"] = info["icc_profile"]
    if Path(p).suffix.lower() in (".jpg", ".jpeg"):
        kw.update(quality=95, subsampling=0)
    Image.fromarray(arr).save(p, **kw)


def to_float(u8):
    return u8.astype(np.float64) / 255.0


def to_u8(f):
    return (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)


def srgb_to_lin(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(c, 0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def lum(rgb):
    return rgb @ LUMA
