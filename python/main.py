from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from problem import SerialSMD, PointLoad
from multimodel import MultiModelROM
from multisoar import MultiSOARROM
from piecewise_soar import PiecewiseSOARROM
from fit_complex import SuperModel

def rel_err(u0, u1):
    return np.abs(1-u1/u0)

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
    omega = np.linspace(1e-6, 0.5, 500)
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

    # Full system
    u = prob.get_frf(omega, f, N-1)

    rom_m = 3
    rom_n = 6

    # Multi-model ROM
    mmr = MultiModelROM(prob.M, prob.C, prob.K, f.vec, damp_func)
    mmr.reduce(omega, rom_m, rom_n)
    ur1 = mmr.get_frf(omega, N-1)

    #soar = MultiSOARROM(prob.M, prob.C, prob.K, f.vec, damp_func)
    #soar.reduce(omega, 2, 5)
    #ur2 = soar.get_frf(omega, N-1)

    s = 1j*omega
    ydata = damp_func(omega)
    def fit(x, cc):
        return cc[0]/x + cc[1] + cc[2]*x
    models = SuperModel(fit, 3, s, ydata)
    models.greedy_fit()
    models.compute_weights()

    psoar = PiecewiseSOARROM(prob.M, prob.C, prob.K, f.vec, damp_func, models)
    psoar.reduce(omega, rom_m, rom_n)
    ur3 = psoar.get_frf(omega, N-1)

    #e2 = rel_err(u, ur2)
    #e3 = rel_err(u, ur3)

    fig, ax = plt.subplots()
    ax.semilogy(omega, np.abs(u), 'k')
    ax.semilogy(omega, np.abs(ur1), 'k:')
    #ax.semilogy(omega, np.abs(ur2), 'g')
    ax.semilogy(omega, np.abs(ur3), 'k--')
    ax.set_xlabel(r'Frequency, $\omega$ [rad/s]')
    ax.set_ylabel('Displacment [m]')
    ax.legend(['Full system', 'Multi-model', 'Proposed ROM'])
    fig.tight_layout()
    plt.show()
