#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 09:05:20 2024

@author: notuser
"""
import scipy.sparse
from scipy.sparse import csc_matrix, csr_matrix

try:
    import cupy as np
    from cupyx.scipy.sparse.linalg import eigsh

    USE_CUDA = True
except ImportError:
    import numpy as np
    from scipy.sparse.linalg import eigsh

    USE_CUDA = False


def sparsecov(a):
    a = csr_matrix(a).astype(np.float32)
    c = np.array((a.T @ a).toarray())

    a = np.array(a.toarray())
    m = a.mean(axis=0)

    if USE_CUDA:
        c -= 2 * np.einsum("ij,k->jk", a, m)
    else:
        c -= 2 * np.einsum("ij,k->jk", a, m)

    c += a.shape[0] * np.outer(m, m)

    return c


def eigbase(a, k=1):
    a = np.array(a)
    return eigsh(a, k=k)[1]


def rand_bincsc(density, n=1000, m=1000, seed=42):
    r = scipy.sparse.random(
        n, m, density, "csc", random_state=np.random.default_rng(seed)
    )

    arr = (2 * r).astype(np.uint8)
    arr.eliminate_zeros()
    return arr


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
