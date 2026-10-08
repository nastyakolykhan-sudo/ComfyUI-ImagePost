"""Layered PSD writer for hand finishing in Photoshop: 8-bit RGB, PackBits channels, optional layer masks.

numpy + struct only (no psd-tools, no Photoshop), following Adobe's Photoshop File Formats Specification (PSD v1).
Layers go bottom to top, as dicts:
  name   layer name (kept in full in a 'luni' block; the Pascal copy is cut to 31 bytes)
  rgb    h x w x 3 uint8 placed with its top-left pixel at (left, top) in document px; 0 x 0 = empty layer
  alpha  h x w uint8 transparency, None = opaque
  mask   h x w uint8 layer mask over the same rect, hidden outside it; None = no mask
The flattened image goes in the image data section, so Quick Look, Pillow and other readers show the result.
write_refined() builds image-post's three layers from the arrays cmd_run already has.
"""
import struct
from pathlib import Path

import numpy as np

from .images import to_u8

BLOCK_ROWS = 512   # PackBits works in row blocks to bound memory on 4K frames
BLEND_RANGES = b"\x00\x00\xff\xff" * 10   # Blend If defaults: composite gray and 4 channels, source and destination


def packbits(ch):
    """PackBits per row of a uint8 h x w channel -> (row byte counts as big-endian uint16, data bytes).

    Runs of 3 or more equal bytes become repeat packets; the rest goes in literal packets of up to 128 bytes.
    """
    if ch.size == 0:
        return np.zeros(ch.shape[0], ">u2"), b""
    counts, parts = [], []
    for r in range(0, ch.shape[0], BLOCK_ROWS):
        c, d = _packbits_block(np.ascontiguousarray(ch[r:r + BLOCK_ROWS], np.uint8))
        counts.append(c)
        parts.append(d)
    return np.concatenate(counts).astype(">u2"), b"".join(parts)


def _packbits_block(ch):
    h, w = ch.shape
    a = ch.ravel()
    brk = np.ones(a.size, bool)
    brk[1:] = a[1:] != a[:-1]
    brk[::w] = True
    starts = np.flatnonzero(brk)
    lens = np.diff(np.append(starts, a.size))
    rep = lens >= 3
    # repeat runs, in chunks of up to 128: [257 - n, value] ([0, value], a 1-byte literal, for a 1-byte tail)
    rs, rl = starts[rep], lens[rep]
    nck = (rl + 127) // 128
    r_src = np.repeat(rs, nck) + 128 * (np.arange(nck.sum()) - np.repeat(np.cumsum(nck) - nck, nck))
    r_len = np.minimum(np.repeat(rs + rl, nck) - r_src, 128)
    # literal bytes: unbroken stretches within a row, in chunks of up to 128: [n - 1, bytes]
    lpos = np.flatnonzero(~np.repeat(rep, lens))
    seg = np.ones(lpos.size, bool)
    seg[1:] = (np.diff(lpos) != 1) | (lpos[1:] % w == 0)
    k = lpos - lpos[seg][np.cumsum(seg) - 1]
    head = k % 128 == 0
    ck = np.cumsum(head) - 1
    l_src = lpos[head]
    l_len = np.bincount(ck, minlength=l_src.size)
    # packets in stream order
    src = np.concatenate([r_src, l_src])
    size = np.concatenate([np.full(r_src.size, 2), l_len + 1])
    order = np.argsort(src, kind="stable")
    off = np.empty(src.size, np.int64)
    off[order] = np.cumsum(size[order]) - size[order]
    r_off, l_off = off[:r_src.size], off[r_src.size:]
    out = np.empty(int(size.sum()), np.uint8)
    out[r_off] = np.where(r_len > 1, 257 - r_len, 0)
    out[r_off + 1] = a[r_src]
    out[l_off] = l_len - 1
    out[l_off[ck] + 1 + lpos - l_src[ck]] = a[lpos]
    return np.bincount(src // w, weights=size, minlength=h).astype(np.int64), out.tobytes()


def _channel(arr):
    """Channel image data: compression flag, then PackBits row counts and rows (raw flag alone when empty)."""
    if arr.size == 0:
        return struct.pack(">H", 0)
    counts, data = packbits(arr)
    return struct.pack(">H", 1) + counts.tobytes() + data


def _pascal(name, pad):
    b = name.encode("mac_roman", "replace")[:31]
    s = bytes([len(b)]) + b
    return s + b"\0" * (-len(s) % pad)


def _tagged(key, data):
    data += b"\0" * (-len(data) % 4)
    return b"8BIM" + key + struct.pack(">I", len(data)) + data


def _resource(rid, data):
    return b"8BIM" + struct.pack(">H", rid) + _pascal("", 2) + struct.pack(">I", len(data)) + data + b"\0" * (len(data) % 2)


def write(path, layers, composite, icc=None, dpi=None):
    """Write a layered PSD. composite: H x W x 3 uint8 (x 4 when the image has transparency), the flattened image."""
    H, W, nc = composite.shape
    records, blobs = [], []
    for L in layers:
        rgb = L["rgb"]
        h, w = rgb.shape[:2]
        top, left = (int(L.get("top", 0)), int(L.get("left", 0))) if h and w else (0, 0)
        rect = (top, left, top + h, left + w)
        alpha = L.get("alpha")
        chans = [(-1, np.full((h, w), 255, np.uint8) if alpha is None else alpha)] + [(c, rgb[..., c]) for c in range(3)]
        if L.get("mask") is not None:
            chans.append((-2, L["mask"]))
        data = [_channel(np.asarray(c, np.uint8)) for _, c in chans]
        blobs += data
        mask = struct.pack(">4iBB2x", *rect, 0, 0) if L.get("mask") is not None else b""
        u = L["name"].encode("utf-16-be")
        extra = (struct.pack(">I", len(mask)) + mask + struct.pack(">I", len(BLEND_RANGES)) + BLEND_RANGES
                 + _pascal(L["name"], 4) + _tagged(b"luni", struct.pack(">I", len(u) // 2) + u))
        records.append(struct.pack(">4iH", *rect, len(chans))
                       + b"".join(struct.pack(">hI", cid, len(d)) for (cid, _), d in zip(chans, data))
                       + b"8BIMnorm" + struct.pack(">4B", 255, 0, 0x08, 0)   # opacity, clipping, flags (visible), filler
                       + struct.pack(">I", len(extra)) + extra)
    info = struct.pack(">h", -len(layers) if nc == 4 else len(layers)) + b"".join(records)
    size = len(info) + sum(len(b) for b in blobs)
    pad = b"\0" * (size % 2)
    res = b""
    if dpi:
        hx, hy = (int(round(float(d) * 65536)) for d in dpi)   # Fixed 16.16 pixels per inch
        res += _resource(1005, struct.pack(">IHHIHH", hx, 1, 1, hy, 1, 1))
    if icc:
        res += _resource(1039, icc)
    flat = [packbits(composite[..., c]) for c in range(nc)]
    with open(path, "wb") as f:
        f.write(b"8BPS" + struct.pack(">H6xHIIHH", 1, nc, H, W, 8, 3))   # version 1, 8 bits, RGB
        f.write(struct.pack(">I", 0))                                     # no colour mode data
        f.write(struct.pack(">I", len(res)) + res)
        f.write(struct.pack(">II", size + len(pad) + 8, size + len(pad)))  # layer and mask info, layer info
        f.write(info)
        for b in blobs:
            f.write(b)
        f.write(pad + struct.pack(">I", 0))                               # no global layer mask
        f.write(struct.pack(">H", 1))                                     # flattened image, PackBits
        for c, _ in flat:
            f.write(c.tobytes())
        for _, d in flat:
            f.write(d)


def write_refined(path, scene_u8, scene_a, info, res, base, F, card, alpha, roi):
    """image-post's layers for finishing by hand, bottom to top: the original scene, the background fill alone
    (opaque where the fill ran, transparent elsewhere), the real product with matte x occluder visibility as its mask.
    base, F, card, alpha cover the work box roi; res is the saved output. Returns the report.json entry."""
    x0, y0, x1, y1 = roi
    base_u8, card_u8, mask_u8 = to_u8(base), to_u8(card), to_u8(alpha)
    ys, xs = np.nonzero(F)
    fy0, fy1, fx0, fx1 = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1) if len(ys) else (0, 0, 0, 0)
    layers = [{"name": "original scene", "rgb": scene_u8, "alpha": scene_a},
              {"name": "background fill", "rgb": base_u8[fy0:fy1, fx0:fx1], "top": y0 + fy0, "left": x0 + fx0,
               "alpha": np.where(F[fy0:fy1, fx0:fx1], 255, 0).astype(np.uint8)},
              {"name": "product", "rgb": card_u8, "mask": mask_u8, "top": y0, "left": x0}]
    write(path, layers, res if scene_a is None else np.dstack([res, scene_a]), info.get("icc_profile"), info.get("dpi"))
    # the 8-bit layers blended in numpy against the PNG: only rounding at the matte's soft edge may differ
    m = mask_u8[..., None].astype(np.int32)
    d = np.abs((base_u8 * (255 - m) + card_u8 * m + 127) // 255 - res[y0:y1, x0:x1]).max(2)
    return {"path": str(path), "mb": round(Path(path).stat().st_size / 2 ** 20, 1),
            "layers": [{"name": "original scene", "box": [0, 0, scene_u8.shape[1], scene_u8.shape[0]]},
                       {"name": "background fill", "box": [x0 + fx0, y0 + fy0, x0 + fx1, y0 + fy1] if len(ys) else None},
                       {"name": "product", "box": [x0, y0, x1, y1], "mask": "matte x occluders"}],
            "layers_vs_png": {"max_levels": int(d.max()), "px_differ": int((d > 0).sum())}}
