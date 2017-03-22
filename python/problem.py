from __future__ import division, print_function

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

class problem:

    def __init__(self, m, c, k, N):
        self.N = N
        self.m = m
        self.c = c
        self.k = k

    def assemble_mass(self, **kwargs):
        if len(self.m) == 1:
            m = np.ones((self.N,))*mean
        else:
            m = self.m

        # (row, col) coordinates of mass matrix
        i_m = np.arange(self.N)
        j_m = np.arange(self.N)

        # return COO sparse mass matrix
        return sparse.coo_matrix((m, (i_m, j_m)), shape=(self.N, self.N)).tocsr()

    def assemble_stiffness(self, **kwargs):
        if len(self.k) == 1:
            k = np.ones((N,))*mean
        else:
            k = self.k

        # (row, col, value) coordinates of stiffness matrix
        i_k = np.empty((3*N-2,))
        j_k = np.empty((3*N-2,))
        v_k = np.empty((3*N-2,))

        # Assemble stiffness matrix
        i_k[0] = 0; j_k[0] = 0; v_k[0] = k[0]+k[1];
        i_k[1] = 0; j_k[1] = 1; v_k[1] = -k[1];

        idx = 2
        for row in np.arange(1, N-1):
            i_k[idx] = row; j_k[idx] = row-1; v_k[idx] = -k[row]
            idx = idx + 1

            i_k[idx] = row; j_k[idx] = row; v_k[idx] = k[row]+k[row+1]
            idx = idx + 1

            i_k[idx] = row; j_k[idx] = row+1; v_k[idx] = -k[row+1]
            idx = idx + 1

        i_k[3*N-4] = N-1; j_k[3*N-4] = N-2; v_k[3*N-4] = -k[N-1];
        i_k[3*N-3] = N-1; j_k[3*N-3] = N-1; v_k[3*N-3] = k[N-1];

        # return COO sparse stiffness matrix
        return sparse.coo_matrix((v_k, (i_k, j_k)), shape=(N, N)).tocsr()

    def assemble_damping(self, **kwargs):
        # Seed for random numbers
        seed = kwargs.get('seed', 3.)
        mean = kwargs.get('mean', 1.)
        return assemble_stiffness(N, seed=seed, mean=mean)

    def assemble_load(N, **kwargs):
        ndof = kwargs.get('ndof', N-1);
        mag = kwargs.get('mag', 1.)
        f = np.zeros((N,))
        # Unit load at f[ndof]
        f[ndof] = mag
        return f

    def damp_func(omega, **kwargs):
        return 1e4*(omega**3-omega)

    def get_frf(M, C, K, f, omega, ndof, **kwargs):
        u = np.empty((len(omega),), dtype=complex)

        nonvisc = kwargs.get('nonvisc', False)

        if nonvisc:
            g = damp_func(omega)
        else:
            g = np.ones(omega.shape)

        for i, w in enumerate(omega):
            A = -w**2*M + 1j*w*g[i]*C + K
            sol = linalg.spsolve(A, f);
            u[i] = sol[ndof]

        return u

    def eigen_solve(M, C, K, omega0, **kwargs):
        A, B = linearise_symm(M, damp_func(omega0)*C, K, csc=True)

        k = kwargs.get('k', 4)
        which = kwargs.get('which', 'SM')

        w, v = linalg.eigs(A, M=B, k=k, which=which)

        return (w, v)
