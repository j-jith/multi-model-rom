from __future__ import division, print_function

import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

MAX_ITER = 10       # maximum number of iterations in piecewise fitting
MIN_SPLIT_LEN = 2  # minimum size of the pieces in piecewise fitting

class Model(object):
    def __init__(self, func, xdata, ydata, **kwargs):
        self.tol = kwargs.get('tol', 1e-5)
        self.eps = kwargs.get('eps', 1e-12)

        self.func = func
        self.xdata = xdata
        self.ydata = ydata

        self.params = None
        self.solinfo = None

    def residual(self, params):
        cc = params.view(np.complex128)
        return (self.func(self.xdata, cc) - self.ydata).view(np.double)

    def fit(self, init, **kwargs):
        full_output = kwargs.get('full_output', False)

        if not full_output:
            params, ier = leastsq(self.residual, init)
            self.params = params.view(np.complex128)
        else:
            params, cov, info, mesg, ier = leastsq(self.residual, init, full_output=1)
            self.params = params.view(np.complex128)
            self.solinfo = {'params': self.params,
                            'cov': cov,
                            'info': info,
                            'mesg': mesg,
                            'ier': ier,
                           }

    def eval(self):
        return self.func(self.xdata, self.params)

    def get_err_max(self):
        err = 1 - self.eval()/self.ydata
        err_r = np.max(np.abs(err.real))
        err_i = np.max(np.abs(err.imag))
        return max(err_r, err_i)

    def is_okay(self):
        if self.get_err_max() < self.tol:
            return True
        else:
            return False



def greedy_fit(func, s, ydata):
    split_index = [0, len(s)-1]
    fit_list = []

    init = np.array([0.+0.j, 0.+0.j, 0.+0.j]).view(np.double)
    n_iter = 0
    ii = 1
    fit = None
    while ii < len(split_index):
        s_i = s[split_index[ii-1]:split_index[ii]]
        y_i = ydata[split_index[ii-1]:split_index[ii]]

        if len(s_i) <= MIN_SPLIT_LEN:
            print('Split too small (length < {}). Moving on'.format(MIN_SPLIT_LEN))
            if fit:
                if fit not in fit_list:
                    fit_list.append(fit)
                split_index.pop(ii)
                ii += 1
                n_iter = 0
            else:
                break
        elif n_iter > MAX_ITER:
            print('Exceeded MAX_ITER in split. Moving on. Max. error = {}'.format(fit.get_err_max()))
            if fit not in fit_list:
                fit_list.append(fit)
            split_index.pop(ii)
            ii += 1
            n_iter = 0
        else:
            fit = Model(func, s_i, y_i)
            fit.fit(init)
            n_iter += 1

            if fit.is_okay():
                print('Max. error below TOLERANCE. Moving on')
                fit_list.append(fit)
                ii += 1
                n_iter = 0
            else:
                split_index.insert(ii, split_index[ii-1] + len(s_i)//2)
                #split_index.append(split_index[ii-1] + len(w)//2)
                #split_index.sort()

    return fit_list

if __name__ == '__main__':

    # Frequency domain
    omega = np.linspace(0, 0.5, 500)
    s = 1j*omega
    # Damping function
    def damp_func(s, **kwargs):
        return 1e4*(s.imag**3-s.imag)

    y = damp_func(s)

    def fit(x, cc):
        return cc[0] + cc[1]*x + cc[2]*x**2

    # model = Model(fit, s, y)
    # init = np.array([0.+0.j, 0.+0.j, 0.+0.j]).view(np.double)
    # model.fit(init)

    # cc = model.params
    # y1 = fit(s, cc)

    fit_list = greedy_fit(fit, s, y)

    # Real part
    fig, ax = plt.subplots()
    ax.plot(s.imag, y.real)
    for f in fit_list:
        ax.plot(f.xdata.imag, f.eval().real)
    # Imaginary part
    fig1, ax1 = plt.subplots()
    ax1.plot(s.imag, y.imag)
    for f in fit_list:
        ax1.plot(f.xdata.imag, f.eval().imag)
    # Show plot
    plt.show()
