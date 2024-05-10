# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 2024

@author: marie
"""

import matplotlib.pylab as plt
import numpy as np
from scipy.optimize import curve_fit

def func_power_law(x, a, b, c):
    return c +  a * (x**b)

def func_parabolic(x, E, c0):
    v = 0.5
    R_c = 10e-9
    return c0 + (4/3)*np.sqrt(R_c)*(E/(1-v**2))*x**(3/2)

def func_conical(x, E, c0):
    v = 0.5
    alpha = np.pi/4
    return c0 + (E/(1-v**2))*((2*np.tan(alpha))/np.pi)*x**2

def func_pyramidal(x, E, c0):
    v = 0.5
    theta = np.pi/4
    return c0 + ((np.tan(theta))/np.sqrt(2))*(E/(1-v**2))*x**2

def fitYoungsModulus(F, delta, contact_point_list, substrate_contact_list, indenter='pyramidal'):
    # R_c: radius of tip curvature
    # F: force
    # delta: indentation
    # v: Poisson's ratio
    # E: Young's modulus
    # R_c = 10e-9
    # v = 0.5
    popt_list = []
    if indenter == 'parabolic':
        func = func_parabolic
    if indenter == 'conical':
        func = func_conical
    if indenter == 'pyramidal':
        func = func_pyramidal
    for k in range(len(F)):
        #perc_top = 95
        slice_bottom = contact_point_list[k]
        slice_top = substrate_contact_list[k]
        delt = delta[k][0]*10**(-6)
        f = F[k][0]*10**(-9)
        popt, pcov = curve_fit(func, delt[slice_bottom:slice_top], f[slice_bottom:slice_top],  maxfev = 100000)
        popt_list.append(popt)
        print(k, popt[0])
        
        fig, ax = plt.subplots()
        ax.plot(delt, f, 'r') # F_bS[k][0]
        ax.plot(delt[slice_bottom:slice_top], func(delt[slice_bottom:slice_top], *popt), 'orange')
        ax.plot(delt[slice_bottom], f[slice_bottom], 'ro', label='cell contact point estimation')
        ax.plot(delt[slice_top], f[slice_top], 'go', label='hard substrate contact point estimation')
        ax.set(xlabel='indentation (m)', ylabel='force (N)', title='Force-delta curve for pyramidal indenter %i' % k)
        fig.savefig('Results\Fdelta_E_pyramidal' + str(k) + '.png')

    # F = (4/3)*np.sqrt(R_c)*(E/(1-v**2))*delta**(3/2)
    # E = (parabolic_fit_param*3*(1-v**2))/(4*np.sqrt(R_c))
    return popt_list, fig

