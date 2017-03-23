from __future__ import division, print_function

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

class Load(object):
    def __init__(self, N, vec):
        self.N = N
        self.vec = vec

class PointLoad(Load):
    def __init__(self, N, mag, ndof):
        self.N = N
        self.vec = np.zeros((N, ))
        self.vec[ndof] = mag

class SerialSMD(object):
    '''
    Serially connected spring-mass-damper (SMD) system with non-viscous damping
    '''
    def __init__(self, **kwargs):
        '''
        N: Total degree of freedom
        m: Vector of masses (pass a list of one item if all masses are equal)
        c: Vector of damping constants (pass list of one item if all equal)
        k: Vector of spring stiffnesses (pass list of one item if all equal)
        damp_func: Non-viscous damping function (should be a function)
        '''
        self.N = kwargs.get('N', 500)
        self.m = kwargs.get('m', [1.])
        self.c = kwargs.get('c', [1e-2])
        self.k = kwargs.get('k', [100.])
        self.damp_func = kwargs.get('damp_func', None)

        self.M = self.C = self. K = None
        self.is_assembled = False

    def assemble_mass(self, **kwargs):
        '''
        Assemble mass matrix
        Output: mass matrix
        '''
        if len(self.m) == 1:
            m = np.ones((self.N,))*self.m
        else:
            m = self.m

        # (row, col) coordinates of mass matrix
        i_m = np.arange(self.N)
        j_m = np.arange(self.N)

        # return CSR sparse mass matrix
        return sparse.coo_matrix((m, (i_m, j_m)), shape=(self.N, self.N)).tocsr()

    def assemble_aux(self, N, k):
        '''
        Auxiliary function for assembling stiffness and damping matrices
        '''
        # (row, col, value) coordinates of stiffness/damping matrix
        i_k = np.empty((3*N-2,))
        j_k = np.empty((3*N-2,))
        v_k = np.empty((3*N-2,))

        # Assemble stiffness/damping matrix
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

        # return CSR sparse stiffness/damping matrix
        return sparse.coo_matrix((v_k, (i_k, j_k)), shape=(N, N)).tocsr()

    def assemble_stiffness(self, **kwargs):
        '''
        Assemble stiffness matrix
        Output: stiffness matrix
        '''
        if len(self.k) == 1:
            k = np.ones((self.N,))*self.k
        else:
            k = self.k
        return self.assemble_aux(self.N, k)


    def assemble_damping(self, **kwargs):
        '''
        Assemble damping matrix
        Output: damping matrix
        '''
        if len(self.k) == 1:
            c = np.ones((self.N,))*self.c
        else:
            c = self.c
        return self.assemble_aux(self.N, c)

    def assemble(self, **kwargs):
        '''
        Assemble all matrices and store them internally
        '''
        self.M = self.assemble_mass()
        self.K = self.assemble_stiffness()
        self.C = self.assemble_damping()

        self.is_assembled = True

    def get_frf(self, omega, load, ndof, **kwargs):
        '''
        Get the frequency response function (FRF) of the system
        Input:
        - omega: frequency range [rad/s] in a numpy array
        - load: load vector defined as an instance of the class Load
        - ndof: degree of freedom at which response is to be calculated
        Optional input:
        - nonvisc: boolean. If true, include non-viscous damping (default: True)
        Output:
        - u: frequency response
        '''

        if self.is_assembled:
            u = np.empty((len(omega),), dtype=complex)
            f = load.vec
            nonvisc = kwargs.get('nonvisc', True)

            if nonvisc and self.damp_func:
                g = self.damp_func(omega)
            else:
                g = np.ones(omega.shape)

            for i, w in enumerate(omega):
                A = -w**2*self.M + 1j*w*g[i]*self.C + self.K
                sol = linalg.spsolve(A, f);
                u[i] = sol[ndof]

            return u
        else:
            print('Call assemble() function first')

    def linearise_symm(self, M, C, K, **kwargs):
        '''
        Transforms a quadratic eigenproblem to a generalised one.
        Outputs a symmetric matrix pencil.
        '''
        csc = kwargs.get('csc', False)

        A = sparse.block_diag((-K, M))
        B = sparse.bmat([[C, M], [M, None]])

        if csc:
            return (A.tocsc(), B.tocsc())

        return (A, B)

    def linearise(self, M, C, K, **kwargs):
        '''
        Transforms a quadratic eigenproblem to a generalised one.
        Outputs an unsymmetric matrix pencil.
        '''
        csc = kwargs.get('csc', False)

        n = M.shape[0]
        A = sparse.bmat([[-K, None],
            [None, sparse.identity(n)]])
        B = sparse.bmat([[C, M],
            [sparse.identity(n), None]])

        if csc:
            return (A.tocsc(), B.tocsc())

        return (A, B)

    def eigen_solve(self, **kwargs):
        '''
        Solves the eigenvalue problem of the system
        Input (optional):
        - nonvisc: boolean. If true, include non-viscous damping (default: True)
        - shift: frequency [rad/s] at which damping matrix is to be computed (default: 0)
        Output:
        - w: eigenvvalues
        - v: eigenvectors
        '''
        if self.is_assembled:
            nonvisc = kwargs.get('nonvisc', True)
            shift = kwargs.get('shift', 0)

            if nonvisc and self.damp_func:
                C = self.damp_func(shift)*self.C
            else:
                C = self.C

            A, B = self.linearise_symm(self.M, C, self.K, csc=True)

            k = kwargs.get('k', 4)
            which = kwargs.get('which', 'SM')

            w, v = linalg.eigs(A, M=B, k=k, which=which)

            return (w, v)
        else:
            print('Call assemble() function first')

