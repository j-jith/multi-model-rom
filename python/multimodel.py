from __future__ import print_function, division

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

class MultiModelROM:

    def __init__(self, M, C, K, f, damp_func):
        self.M = M
        self.C = C
        self.K = K
        self.f = f
        self.damp_func = damp_func

        self.Q = None
        self.Mr = None
        self.Cr = None
        self.Kr = None
        self.fr = None

        self.is_reduced = False

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
        A, B = self.linearise_symm(self.M,
                self.damp_func(omega0)*self.C,
                self.K, csc=True)

        k = kwargs.get('k', 4)
        which = kwargs.get('which', 'SM')

        w, v = linalg.eigs(A, M=B, k=k, which=which, sigma=1j*omega0)
        return (w, v)

    def pod_orthogonalise(self, Q, **kwargs):
        tol = kwargs.get('tol', 0.)

        # Eigenproblem of covariance matrix
        Cqq = np.dot(Q.conj().T, Q)
        lmbda, psi = np.linalg.eig(Cqq)
        # Filter out eigenvalues by tolerance
        lmbda = np.real(lmbda)
        mask = lmbda/np.max(lmbda) > tol
        lmbda = lmbda[mask]
        psi = psi[:, mask]
        # Normalise eigenvectors
        psi = np.dot(psi, np.diag(1/np.sqrt(lmbda)))
        # Return orthogonal basis
        return np.dot(Q, psi)


    def reduce(self, omega, n_ip, n_eig, **kwargs):
        ind_ip_tmp = np.linspace(0, len(omega)-1, n_ip+1)
        ind_ip = np.int_(np.around((ind_ip_tmp[:-1] + ind_ip_tmp[1:])/2.))
        N = self.M.shape[0]
        # Solve eigenproblem at interpolation points
        print('Solving eigenproblem at interpolation point #1 ...')
        lmbda, Q = self.eigen_solve(omega[ind_ip[0]], k=n_eig)
        Q = Q[:N, :]
        if n_ip > 1:
            for i, w in enumerate(omega[ind_ip[1:]]):
                print('Solving eigenproblem at interpolation point #{} ...'.format(i+2))
                lmbda, q = self.eigen_solve(w, k=n_eig)
                Q = np.hstack((Q, q[:N, :]))

        # Orthogonalise union of eigenvectors
        print('Performing POD orthogonalisation ...')
        Q = self.pod_orthogonalise(Q)
        self.Q = Q

        # Reduce system matrices and load vector
        print('Reducing system matrices ...')
        self.Mr = np.dot(Q.conj().T, self.M.dot(Q))
        self.Cr = np.dot(Q.conj().T, self.C.dot(Q))
        self.Kr = np.dot(Q.conj().T, self.K.dot(Q))
        self.fr = np.dot(Q.conj().T, self.f)

        self.is_reduced=True

    def get_frf(self, omega, ndof, **kwargs):
        if self.is_reduced:
            print('Computing FRF ...')
            sol = np.empty((self.Mr.shape[0], len(omega)), dtype=complex)

            g = self.damp_func(omega)
            for i, w in enumerate(omega):
                A = -w**2*self.Mr + 1j*w*g[i]*self.Cr + self.Kr
                sol[:, i] = np.linalg.solve(A, self.fr);

            u = np.dot(self.Q, sol)
            return u[ndof, :]
        else:
            print('Call reduce() before this function!')

