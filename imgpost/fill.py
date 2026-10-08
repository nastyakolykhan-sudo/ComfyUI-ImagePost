"""Harmonic (membrane) fill of the background where the generated product showed but the real one doesn't."""
import numpy as np
from scipy.ndimage import find_objects, label
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import factorized

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class FillError(RuntimeError):
    pass


def harmonic_fill(img, F, D, offset=(0, 0)):
    """Solve Laplace on F with values from D (4-neighbours) and zero flux elsewhere (occluders, the product).

    Raises FillError when a region of F touches no D pixel: it has no background to borrow from,
    which means the real product's edge stops short of an occluder (extend the outline under it).
    """
    h, w = F.shape
    ys, xs = np.nonzero(F)
    n = len(ys)
    if n == 0:
        return img.copy(), 0
    idx = -np.ones(F.shape, np.int64)
    idx[ys, xs] = np.arange(n)
    diag, b = np.zeros(n), np.zeros((n, img.shape[2]))
    rows, cols, touch = [], [], np.zeros(n, bool)
    for dy, dx in DIRS:
        qy, qx = ys + dy, xs + dx
        ok = (qy >= 0) & (qy < h) & (qx >= 0) & (qx < w)
        k, qy, qx = np.nonzero(ok)[0], qy[ok], qx[ok]
        inF = F[qy, qx]
        inD = D[qy, qx] & ~inF
        diag[k[inF | inD]] += 1
        rows.append(k[inF])
        cols.append(idx[qy[inF], qx[inF]])
        np.add.at(b, k[inD], img[qy[inD], qx[inD]])
        touch[k[inD]] = True
    lab, nl = label(F)
    anchored = np.zeros(nl + 1, bool)
    anchored[lab[ys, xs][touch]] = True
    bad = [i for i in range(1, nl + 1) if not anchored[i]]
    if bad:
        sl = find_objects(lab)
        ox, oy = offset
        raise FillError("; ".join(
            f"{int((lab == i).sum())} px at x {sl[i - 1][1].start + ox}-{sl[i - 1][1].stop + ox}, "
            f"y {sl[i - 1][0].start + oy}-{sl[i - 1][0].stop + oy}" for i in bad))
    r, c = np.concatenate(rows), np.concatenate(cols)
    A = csr_matrix((-np.ones(len(r)), (r, c)), shape=(n, n)) + diags(diag)
    solve = factorized(A.tocsc())
    out = img.copy()
    out[ys, xs] = np.stack([solve(b[:, ch]) for ch in range(img.shape[2])], -1)
    return out, n
