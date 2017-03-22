from __future__ import division, print_function

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt

from linearise import linearise_symm

def assemble_mass(N, **kwargs):
    # Seed and mean for random numbers
    seed = kwargs.get('seed', 1.)
    mean = kwargs.get('mean', 1.)

    # Generate repeatable random mass
    #np.random.seed(seed)
    #m = np.random.rand(N) + mean;
    m = np.ones((N,))*mean

    # (row, col) coordinates of mass matrix
    i_m = np.arange(N)
    j_m = np.arange(N)

    # return COO sparse mass matrix
    return sparse.coo_matrix((m, (i_m, j_m)), shape=(N, N)).tocsr()

def assemble_stiffness(N, **kwargs):
    # Seed and mean for random numbers
    seed = kwargs.get('seed', 2.)
    mean = kwargs.get('mean', 100.)

    # Generate repeatable random stiffness
    #np.random.seed(2)
    #k = np.random.rand(N) + mean;
    k = np.ones((N,))*mean

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

def assemble_damping(N, **kwargs):
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

if __name__ == '__main__':

    # Degrees of freedom
    N = 500;
    omega = np.linspace(0, 0.5, 500)

    # Assemble mass, stiffness, damping and load
    M = assemble_mass(N, mean=1.)
    K = assemble_stiffness(N, mean=100.)
    C = assemble_damping(N, mean=1e-2)
    f = assemble_load(N, mag=1e-3)

    # Frequency response
    #u = get_frf(M, C, K, f, omega, N-1)
    #u1 = get_frf(M, C, K, f, omega, N-1, nonvisc=True)
    #plt.semilogy(omega, np.abs(u))
    #plt.semilogy(omega, np.abs(u1), 'r');
    #plt.show()

    # Eigenvalue problem
    l, v = eigen_solve(M, C, K, omega[len(omega)//2], k=10, which='SM')

    print(l)
