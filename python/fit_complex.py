from __future__ import division, print_function

import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

class Model(object):
    def __init__(self, func, xdata, ydata, **kwargs):
        self.tol = kwargs.get('tol', 1e-5)

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

class SuperModel(object):
    def __init__(self, func, n_params, xdata, ydata, **kwargs):
        self.tol = kwargs.get('tol', 1e-5) # Relative error at which curve fit is accepted
        self.maxit = kwargs.get('maxit', 10) # Max. iteration in curve fitting
        self.eps = kwargs.get('eps', 1e-12) # Any value less than this is equated to zero (for weights)
        self.min_split_len = kwargs.get('min_split_len', 2) # Min. length of x-segments (won't go below this)
        self.weight_scale = kwargs.get('weight_scale', 25) # Scale factor in weighting function

        self.func = func # Function for fitting
        self.n_params = n_params # No. of parameters in fitting function
        self.xdata = xdata
        self.ydata = ydata

        self.models = None
        self.weights = None

    def greedy_fit(self):
        split_index = [0, len(self.xdata)-1]
        self.models = []

        init = np.zeros(2*self.n_params)
        n_iter = 0
        ii = 1
        fit = None
        while ii < len(split_index):
            s_i = self.xdata[split_index[ii-1]:split_index[ii]]
            y_i = self.ydata[split_index[ii-1]:split_index[ii]]

            if len(s_i) <= self.min_split_len:
                print('Split too small (length < {}). Moving on'.format(self.min_split_len))
                if fit:
                    if fit not in self.models:
                        self.models.append(fit)
                    split_index.pop(ii)
                    ii += 1
                    n_iter = 0
                else:
                    break
            elif n_iter > self.maxit:
                print('Exceeded max. iterations in split. Moving on. Max. error = {}'.format(fit.get_err_max()))
                if fit not in self.models:
                    self.models.append(fit)
                split_index.pop(ii)
                ii += 1
                n_iter = 0
            else:
                fit = Model(self.func, s_i, y_i, tol=self.tol)
                fit.fit(init)
                n_iter += 1

                if fit.is_okay():
                    print('Max. error below tolerance. Moving on')
                    self.models.append(fit)
                    ii += 1
                    n_iter = 0
                else:
                    split_index.insert(ii, split_index[ii-1] + len(s_i)//2)
                    #split_index.append(split_index[ii-1] + len(w)//2)
                    #split_index.sort()

    def compute_weights(self):
        centres = np.array([(f.xdata[0] + f.xdata[-1])/2. for f in self.models])
        beta = self.weight_scale
        #sigma = 0.1
        weights = np.zeros((len(self.xdata), len(centres)))
        for i, s_i in enumerate(self.xdata):
            d_i = np.abs(centres-s_i)
            m = np.min(d_i)
            if m == 0:
                j = np.argmin(d_i)
                weights[i,:] = np.zeros(d_i.shape)
                weights[i,j] = 1
            else:
                weights[i,:] = np.exp(-beta*d_i/m)
                #weights[i,:] = np.exp(-d_i**2/sigma**2)
                weights[i,:] = weights[i,:]/np.sum(weights[i,:])

        weights[weights<self.eps] = 0
        self.weights = weights.T


if __name__ == '__main__':

    # Frequency domain
    omega = np.linspace(1e-5, 0.5, 500)
    s = 1j*omega
    # Damping function
    def damp_func(s, **kwargs):
        return 1e4*(s.imag**3-s.imag)

    y = damp_func(s)

    def fit(x, cc):
        return cc[0]/x + cc[1] + cc[2]*x

    models = SuperModel(fit, 3, s, y)
    models.greedy_fit()
    models.compute_weights()

    # Real part
    fig, ax = plt.subplots()
    ax.plot(s.imag, y.real)
    for f in models.models:
        ax.plot(f.xdata.imag, f.eval().real)
    # Imaginary part
    fig1, ax1 = plt.subplots()
    ax1.plot(s.imag, y.imag)
    for f in models.models:
        ax1.plot(f.xdata.imag, f.eval().imag)
    # Weights
    fig2, ax2 = plt.subplots()
    for w in models.weights:
        ax2.plot(s.imag, w)
    # Show plot
    plt.show()
