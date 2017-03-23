from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from problem import SerialSMD, PointLoad
from multimodel import MultiModelROM

if __name__ == '__main__':

    # Degrees of freedom
    N = 500;
    # Mass
    m = [1.]
    # Damping
    c = [1e-2]
    # Stiffness
    k = [100]
    # Frequency domain
    omega = np.linspace(0, 0.5, 500)
    # Damping function
    def damp_func(omega, **kwargs):
        return 1e4*(omega**3-omega)
    # Load
    f = PointLoad(N, 1e-3, N-1)

    prob = SerialSMD(N=N, m=m, c=c, k=k, damp_func=damp_func)
    prob.assemble()

    # # Eigenproblem
    # w, v = prob.eigen_solve(shift=omega[10])

    # # Frequency response
    # u = prob.get_frf(omega, f, N-1)
    # u0 = prob.get_frf(omega, f, N-1, nonvisc=False)
    # plt.semilogy(omega, np.abs(u0))
    # plt.semilogy(omega, np.abs(u), 'r');
    # plt.show()

    # Compare with ROM
    rom = MultiModelROM(prob.M, prob.C, prob.K, f.vec, damp_func)
    rom.reduce(omega, 5, 5)
    ur = rom.get_frf(omega, N-1)
    u = prob.get_frf(omega, f, N-1)
    plt.semilogy(omega, np.abs(u))
    plt.semilogy(omega, np.abs(ur), 'r');
    plt.show()
