import os

import numpy as np
import scipy.sparse
from dotenv import load_dotenv
from scipy.sparse import csc_matrix
from tqdm import tqdm

load_dotenv()
USE_CUDA = int(os.environ.get("CHS_PCE_USE_CUDA", 0))


if USE_CUDA:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import eigsh
else:
    from scipy.sparse.linalg import eigsh


def eigvec(a, k=1):
    if USE_CUDA:
        a = cp.array(a)

    b = eigsh(a, k=k)[1]

    return b.get() if USE_CUDA else b


def randb_csc(density, rand_state, m=1000, n=1000):
    arr = scipy.sparse.rand(m, n, density, "csc", random_state=rand_state)
    arr.data[:] = 1
    return arr.astype(np.uint8)


def phicoef(arr, eps=1e-7, status=False):
    mat = csc_matrix(arr).astype(np.uint8)
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

            coef[i, j] = (n * n11 - n01 * n10) / (
                np.sqrt(n01 * n10 * (n - n01) * (n - n10)) + eps
            )

            coef[j, i] = coef[i, j]
            j += 1

            if status:
                bar.update(1)

    if status:
        bar.close()

    return coef
