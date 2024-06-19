import numpy as np
import scipy.sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh
from tqdm import tqdm


def eigvec(a, k=1):
    return eigsh(a, k=k)[1]


def randbool_csc(density, rand_state, m=1000, n=1000):
    arr = scipy.sparse.rand(m, n, density, "csc", random_state=rand_state)
    arr.data[:] = 1
    return arr.astype(np.uint8)


def indices_tocsc(indices, shape=None):
    col_ind = [i for c in indices for i in c]
    row_ind = [j for j, c in enumerate(indices) for _ in c]
    data = np.ones(len(row_ind), dtype=np.uint8)
    return csc_matrix((data, (row_ind, col_ind)), shape=shape)


def phicoef(mat, status=False):
    mat = csc_matrix(mat).astype(np.uint8)
    n, m = mat.shape

    coef = np.empty((m, m), dtype=np.float32)

    if status:
        bar = tqdm(total=m * (m + 1) // 2)

    for i, (s1, r1) in enumerate(zip(mat.indptr, mat.indptr[1:])):
        v1 = set(mat.indices[s1:r1])
        j = i

        for s2, r2 in zip(mat[:, i:].indptr, mat[:, i:].indptr[1:]):
            v2 = set(mat.indices[s2:r2])

            # https://en.wikipedia.org/wiki/Phi_coefficient#Definition
            n11, n01, n10 = len(v1 & v2), len(v1), len(v2)

            d = n01 * n10 * (n - n01) * (n - n10)

            coef[i, j] = (n * n11 - n01 * n10) / np.sqrt(float(d)) if d else 0
            coef[j, i] = coef[i, j]
            j += 1

            if status:
                bar.update()

    if status:
        bar.close()

    return coef
