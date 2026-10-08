"""Softness and grain matching so the real product sits at the generated frame's level of detail."""
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter


def edge_sigma(L, box):
    """Gaussian-equivalent edge blur (px) from the steepest horizontal edges in box (x0, y0, x1, y1), work-box coords."""
    x0, y0, x1, y1 = box
    sub = L[y0:y1, x0:x1]
    gx = np.abs(np.diff(sub, axis=1))
    contrast = np.percentile(sub, 95) - np.percentile(sub, 5)
    gmax = np.percentile(gx.max(1), 50)
    return float(contrast / (gmax * np.sqrt(2 * np.pi))) if gmax > 0 else float("nan")


def auto_probe(L, mask, size=(80, 50)):
    """Work-box window with the strongest horizontal edges inside mask (fallback when no blur_probe is given)."""
    gx = np.abs(np.diff(gaussian_filter(L, 0.7), axis=1, append=L[:, -1:]))
    score = uniform_filter(np.where(mask, gx, 0.0), size=(size[1], size[0]))
    yy, xx = np.unravel_index(np.argmax(score), score.shape)
    x0 = int(np.clip(xx - size[0] // 2, 0, L.shape[1] - size[0]))
    y0 = int(np.clip(yy - size[1] // 2, 0, L.shape[0] - size[1]))
    return x0, y0, x0 + size[0], y0 + size[1]


def blur_rgb(img, sigma):
    return np.stack([gaussian_filter(img[..., c], sigma) for c in range(img.shape[2])], -1) if sigma > 0 else img


def highpass(img, sigma=1.2):
    return img - blur_rgb(img, sigma)


def mad_std(x):
    """Robust noise std (MAD). Plain std over-reads when the sample crosses edges."""
    med = np.median(x, axis=0)
    return 1.4826 * np.median(np.abs(x - med), axis=0)


def flat_mask(L, mask, q=40):
    """Pixels in mask whose local gradient is in the lowest q percent (where grain is measurable)."""
    gy, gx = np.gradient(gaussian_filter(L, 1.0))
    gm = np.hypot(gx, gy)
    return mask & (gm <= np.percentile(gm[mask], q))


def make_grain(shape, std, mask, seed=7, size=0.6):
    """Gaussian grain, std per channel measured over mask."""
    rng = np.random.default_rng(seed)
    n = np.stack([gaussian_filter(rng.standard_normal(shape), size) for _ in range(3)], -1)
    return n * (np.asarray(std, float) / (n[mask].std(0) + 1e-9))
