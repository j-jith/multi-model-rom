from __future__ import print_function, division

class multimodel:

    def __init__(self, M, C, K):
        self.M = M
        self.C = C
        self.K = K

    def linearise_symm(self, M, C, K, **kwargs):
        csc = kwargs.get('csc', False)

        A = sparse.block_diag((-K, M))
        B = sparse.bmat([[C, M], [M, None]])

        if csc:
            return (A.tocsc(), B.tocsc())
        return (A, B)

    def linearise(self, M, C, K, **kwargs):
        csc = kwargs.get('csc', False)

        n = M.shape[0]
        A = sparse.bmat([[-K, None],
            [None, sparse.identity(n)]])
        B = sparse.bmat([[C, M],
            [sparse.identity(n), None]])

        if csc:
            return (A.tocsc(), B.tocsc())

        return (A, B)

    def eigen_solve(self, omega0, **kwargs):
        A, B = linearise_symm(self.M,
                self.damp_func(omega0)*self.C,
                self.K, csc=True)

        k = kwargs.get('k', 4)
        which = kwargs.get('which', 'SM')

        w, v = linalg.eigs(A, M=B, k=k, which=which)

        return (w, v)

    def reduce(self, omega, n_ip, n_eig
