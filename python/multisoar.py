from __future__ import print_function, division

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

class MultiSOARROM:

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

    def soar(self, omega0, n):
        '''
        Second-order Arnoldi Reduction(SOAR).
        Input: M, C, K - scipy csr_matrix
            b       - numpy vector
            n       - number of Arnoldi vectors
        Output: q      - Arnoldi basis
        '''

        C = 2*1j*omega0*self.M + self.damp_func(omega0)*self.C
        K = -omega0**2 * self.M + 1j*omega0*self.damp_func(omega0)*self.C + self.K

        solve = linalg.factorized(K)

        r_0 = solve(self.f)

        q = np.zeros((len(r_0), n), dtype=complex)
        p = np.zeros((len(r_0), n), dtype=complex)

        q[:, 0] = r_0/np.linalg.norm(r_0)

        for j in range(n-1):
            r = -solve(C.dot(q[:, j]) + self.M.dot(p[:, j]))
            s = q[:, j]

            for i in range(j+1):
                t_ij = np.dot(q[:, i].conj(), r)
                r -= q[:, i]*t_ij
                s -= p[:, i]*t_ij

            r_norm = np.linalg.norm(r)
            if r_norm == 0:
                return q

            q[:, j+1] = r/r_norm
            p[:, j+1] = s/r_norm

        return q


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


    def reduce(self, omega, n_ip, n_arn, **kwargs):
        ind_ip_tmp = np.linspace(0, len(omega)-1, n_ip+1)
        ind_ip = np.int_(np.around((ind_ip_tmp[:-1] + ind_ip_tmp[1:])/2.))
        N = self.M.shape[0]
        # Solve eigenproblem at interpolation points
        print('SOAR at interpolation point #1 ...')
        Q = self.soar(omega[ind_ip[0]], n_arn)
        if n_ip > 1:
            for i, w in enumerate(omega[ind_ip[1:]]):
                print('SOAR at interpolation point #{} ...'.format(i+2))
                q = self.soar(w, n_arn)
                Q = np.hstack((Q, q))

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


