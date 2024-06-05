import scipy.sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh
import numpy as np


def sparse_cov(a):
    a = csc_matrix(a).astype(np.float32)
    c = (a.T @ a).toarray()

    a = a.toarray()
    m = a.mean(axis=0)

    c -= 2 * np.einsum("ij,k->jk", a, m)

    c += a.shape[0] * np.outer(m, m)

    return c


def eigbase(a, k=1):
    return eigsh(a, k=k)[1]


def rand_bincsc(density, rand_state, m=1000, n=1000):
    arr = scipy.sparse.rand(m, n, density, 'csc', random_state=rand_state)
    arr.data[:] = 1
    return arr.astype(np.uint8)


def phicoef(arr, eps=1e-7):
    mat = csc_matrix(arr).astype(np.uint8)
    n, m = mat.shape

    coef = np.empty((m, m), dtype=np.float32)

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

    return coef
