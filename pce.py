import numpy as np
import scipy.sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh
from tqdm import tqdm


def eigvec(a, k=1):
    return eigsh(a, k=k)[1]


def randbool_csc(rand_state, n, m, density):
    arr = scipy.sparse.rand(n, m, density, "csc", random_state=rand_state)
    arr.data[:] = 1
    return arr.astype(np.uint8)


def indices_tocsc(indices, cols=None):
    col_ind, row_ind = [], []

    for n, idx in enumerate(indices):
        col_ind.extend(idx)
        row_ind.extend(np.full(len(idx), n))

    data = np.ones(len(row_ind), dtype=np.uint8)
    shape = max(row_ind) + 1, cols if cols else (max(col_ind) + 1)

    return csc_matrix((data, (row_ind, col_ind)), shape)


def jaccard(mat, cols=None):
    m = cols if cols else (np.max() + 1)
    mat = indices_tocsc(mat + 1, m + 1)[:, 1:]
    coef = np.empty((m, m), dtype=np.float32)

    with tqdm(total=m * (m + 1) // 2) as bar:
        for i, (s1, r1) in enumerate(zip(mat.indptr, mat.indptr[1:])):
            v1 = set(mat.indices[s1:r1])
            j = i

            for s2, r2 in zip(mat[:, i:].indptr, mat[:, i:].indptr[1:]):
                v2 = set(mat.indices[s2:r2])

                n11, n01, n10 = len(v1 & v2), len(v1), len(v2)
                f = n01 + n10 - n11

                coef[i, j] = coef[j, i] = (n11 / f) if f else 0

                j += 1
                bar.update()

    return coef
