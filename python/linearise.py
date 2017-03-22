from __future__ import print_function, division

import scipy.sparse as sparse

def linearise_symm(M, C, K, **kwargs):
    csc = kwargs.get('csc', False)

    A = sparse.block_diag((-K, M))
    B = sparse.bmat([[C, M], [M, None]])

    if csc:
        return (A.tocsc(), B.tocsc())

    return (A, B)

def linearise(M, C, K, **kwargs):
    csc = kwargs.get('csc', False)

    n = M.shape[0]
    A = sparse.bmat([[-K, None],
        [None, sparse.identity(n)]])
    B = sparse.bmat([[C, M],
        [sparse.identity(n), None]])

    if csc:
        return (A.tocsc(), B.tocsc())

    return (A, B)
