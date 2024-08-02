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
    v = 0.5 # v: Poisson's ratio
    R_c = 100e-9 # R_c: radius of tip curvature
    return c0 + (4/3)*np.sqrt(R_c)*(E/(1-v**2))*x**(3/2)

def func_conical(x, E, c0):
    v = 0.5 # v: Poisson's ratio
    alpha = np.pi/4 # alpha: cone half angle
    return c0 + (E/(1-v**2))*((2*np.tan(alpha))/np.pi)*x**2

def func_pyramidal(x, E, c0):
    v = 0.5 # v: Poisson's ratio
    theta = np.pi/4 # theta: pyramid half angle
    return c0 + ((np.tan(theta))/np.sqrt(2))*(E/(1-v**2))*x**2

def fitYoungsModulus(F, delta, contact_point_list, substrate_contact_list, first_peak_list, indenter='pyramidal', plot=False, save=False):
    # R_c: radius of tip curvature
    # F: force
    # delta: indentation
    # v: Poisson's ratio
    # E: Young's modulus
    # R_c = 100e-9 m / 100 nm
    # v = 0.5
    
    if indenter == 'parabolic':
        func = func_parabolic
    if indenter == 'conical':
        func = func_conical
    if indenter == 'pyramidal':
        func = func_pyramidal
    
    E_list = []
    for k in range(len(F)):
        # Defining range to fit the Young's modulus
        slice_bottom = contact_point_list[k]
        if first_peak_list[k] is None:
            slice_top = substrate_contact_list[k] # fit between contact point and hard substrate contact point (no penetration point)
        else:
            slice_top = first_peak_list[k] # fit between contact point and first penetration point
        
        # Conversion to m and N for Young's modulus calculation
        delt = delta[k][0]*10**(-6)
        f = F[k][0]*10**(-9)
        
        if slice_top > (slice_bottom+50):
            # fitting the curve with the function depending on the geometry of the indenter
            popt, pcov = curve_fit(func, delt[slice_bottom:slice_top], f[slice_bottom:slice_top],  maxfev = 100000)
            E_modulus = popt[0]*10**(-3)# kPa
        else:
            E_modulus = 10 # kPa 
        
        E_list.append(E_modulus) # the first fitting parameter (popt[0]) is the Young's modulus
        print(k, E_modulus, ' kPa')
        
        if plot:
            fig, ax = plt.subplots()
            ax.plot(delt, f, 'r') 
            if slice_top > (slice_bottom+50):
                ax.plot(delt[slice_bottom:slice_top], func(delt[slice_bottom:slice_top], *popt), 'orange')
            ax.plot(delt[slice_bottom], f[slice_bottom], 'ro', label='cell contact point estimation')
            if first_peak_list[k] is None:
                ax.plot(delt[slice_top], f[slice_top], 'go', label='hard substrate contact point estimation')
            else:
                ax.plot(delt[slice_top], f[slice_top], 'bo', label='first penetration point estimation')
            ax.set(xlabel='indentation (m)', ylabel='force (N)', title='Force-delta curve for pyramidal indenter %i' % k)
            ax.text(2*10**(-6), 0.4*10**(-8), 'E = %.2f kPa' % E_modulus, fontsize=12)
            plt.legend(loc="upper right")
            if save:
                fig.savefig('Results\Fdelta_E_pyramidal' + str(k) + '.png')
            plt.close()
    return E_list, fig

