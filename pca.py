import numpy as np
import scipy.sparse
from tqdm import tqdm


def indices_tocsc(indices, cols=None):
    col_ind, row_ind = [], []

    for n, idx in enumerate(indices):
        col_ind.extend(idx)
        row_ind.extend(np.full(len(idx), n))

    data = np.ones(len(row_ind), dtype=np.uint8)
    shape = max(row_ind) + 1, cols if cols else (max(col_ind) + 1)

    return scipy.sparse.csc_matrix((data, (row_ind, col_ind)), shape)


def idxarr_tocsc(arr, cols=None):
    if not cols:
        cols = max(arr) + 1

    return indices_tocsc(arr + 1, cols + 1)[:, 1:]


def bincoef(coef_fn, mat, cols):
    cols = cols if cols else (np.max() + 1)
    mat = idxarr_tocsc(mat, cols)

    n, m = mat.shape
    coef = np.empty((m, m), dtype=np.float32)

    with tqdm(total=m * (m + 1) // 2) as bar:
        for i, (s1, r1) in enumerate(zip(mat.indptr, mat.indptr[1:])):
            v1 = set(mat.indices[s1:r1])
            j = i

            for s2, r2 in zip(mat[:, i:].indptr, mat[:, i:].indptr[1:]):
                v2 = set(mat.indices[s2:r2])

                n11, n01, n10 = len(v1 & v2), len(v1), len(v2)

                coef[i, j] = coef[j, i] = coef_fn(n, n11, n01, n10)

                j += 1
                bar.update()

    return coef


def jaccard(mat, cols=None):
    def fn(n, n11, n01, n10):
        d = n01 + n10 - n11
        return n11 / d if d else 0

    return bincoef(fn, mat, cols)


def covariance(mat, cols=None):
    def fn(n, n11, n01, n10):
        return (n * n11 - n01 * n10) / (n * (n - 1))

    return bincoef(fn, mat, cols)


def eigfix(w, v):
    v = np.where(w < 0, -v, v)
    w = np.where(w < 0, -w, w)
    idx = np.argsort(w)
    return w[idx], v[idx]


def kvectors(eigvals, threshold):
    cumulative_var = np.cumsum(eigvals[::-1]) / np.sum(eigvals)
    return np.argmax(cumulative_var > threshold) + 1
