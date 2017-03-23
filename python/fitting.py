import numpy as np
from math import pi, sqrt
from piecewise_fit_greedy import greedy_fit, get_weights
#from myparams import *

omega_0 = 0
omega_1 = 0.5
n_omega = 500


def func1(x):
    return 1e4*(x**3-x)

def func2(x):
    return 1e4*(x**3-x)

def func_fit(x, e0, e1, e2):
    return e0/x + e1 + e2*x
    #return e0 + e1*x + e2*x**2

omega = np.linspace(omega_0, omega_1, n_omega)

fit_list = greedy_fit(omega, func1, func2, func_fit)

split_means = np.array([(xx.w[0] + xx.w[-1])/2 for xx in fit_list])

weights = get_weights(omega, split_means)


#  # Plot weighting functions
#  import matplotlib.pyplot as plt
#  fig, ax = plt.subplots()
#  ii = [np.argmin(np.abs(omega-500*2*pi)), np.argmin(np.abs(omega-625*2*pi))]
#  ax.plot(omega[ii[0]:ii[1]], weights[ii[0]:ii[1], 9], 'k-')
#  ax.plot(omega[ii[0]:ii[1]], weights[ii[0]:ii[1], 10], 'k--')
#  #ylims = ax.get_ylim()
#  #ylims = [ylims[0]-0.05, ylims[1]+0.05]
#  #ax.plot(split_means[9]*np.ones((2,)), ylims, 'k:')
#  #ax.plot(split_means[10]*np.ones((2,)), ylims, 'k:')
#  #ax.set_ylim(ylims)
#  ax.set_xlabel(r'$\omega$ [rad/s]')
#  ax.set_ylabel(r'$w_i(\omega)$')
#  fig.savefig('weights.pdf')
#  plt.show()
#  print('[{}, {})'.format(fit_list[9].w[0], fit_list[9].w[-1]))
#  print('[{}, {})'.format(fit_list[10].w[0], fit_list[10].w[-1]))

# Plot approximate functions
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

lf1, = ax1.plot(omega/2/pi, func1(omega), 'k-')

ax2 = ax1.twinx()
lf2, = ax2.plot(omega/2/pi, func2(omega), 'k--')

for fit in fit_list:
    f1 = func_fit(fit.w, fit.c1[0], fit.c1[1], fit.c1[2])
    f2 = func_fit(fit.w, fit.c2[0], fit.c2[1], fit.c2[2])

    #lg1, = ax1.plot(fit.w[0:2:]/2/pi, f1[0:2:], 'ko', markersize=4.)
    #lg2, = ax2.plot(fit.w[0:2:]/2/pi, f2[0:2:], 'ks', markersize=4.)
    lg1, = ax1.plot(fit.w[0:1:]/2/pi, f1[0:1:], 'ko', markersize=4.)
    lg2, = ax2.plot(fit.w[0:1:]/2/pi, f2[0:1:], 'ks', markersize=4.)

ylims1 = ax1.get_ylim()
ylims2 = ax2.get_ylim()

for fit in fit_list:
    ax1.plot([fit.w[0]/2/pi, fit.w[0]/2/pi], [ylims1[0], ylims1[1]], 'k:', linewidth=1.)

ax1.plot([fit_list[-1].w[-1]/2/pi, fit_list[-1].w[-1]/2/pi], [ylims1[0], ylims1[1]], 'k:', linewidth=1.)

ax1.set_xlabel(r'Frequency, $\omega/2\pi$ [Hz]')
ax1.set_ylabel(r'$g^R(\omega)$')
ax2.set_ylabel(r'$g^I(\omega)$')

ax1.legend([lf1, lg1, lf2, lg2], [r'$g^R(\omega)$', r'$\hat g^R(\omega)$',
    r'$g^I(\omega)$', r'$\hat g^I(\omega)$'])
#ax1.legend([lf1, lg1, lf2, lg2], [r'$g^R(\omega)$', r'$\sum_{i=-1}^1 c_i^R \omega^i$',
#    r'$g^I(\omega)$', r'$\sum_{i=-1}^1 c_i^I \omega^i$'])

fig.tight_layout()
fig.savefig('fitting_{}_segments.pdf'.format(len(fit_list)))

plt.show()

np.savetxt('fit_list.txt', np.vstack([np.hstack((ff.c1, ff.c2)) for ff in fit_list]))
np.savetxt('weights.txt', weights)
