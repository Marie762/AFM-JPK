# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 2024

@author: marie
"""

from tkinter import X
import matplotlib.pylab as plt
import numpy as np
from scipy.optimize import curve_fit
from contactPoint import contactPoint1
from plot import Fdsubplot

def func_power_law(x, a, b, c):
    return c +  a * (x**(b)) #3/2

def func_E(x, E, c1):
    v = 0.5
    R_c = 50e-9
    return c1 + (4/3)*np.sqrt(R_c)*(E/(1-v**2))*x**(3/2)

def parabolicIndenter(F, delta, argmin_list):
    # R_c: radius of tip curvature
    # F: force
    # delta: indentation
    # v: Poisson's ratio
    # E: Young's modulus
    # R_c = 10e-9
    # v = 0.5
    popt_list = []
    for k in range(len(F)):
        #perc_top = 95
        slice_bottom = argmin_list[k]
        #slice_top = round((perc_top/100)*len(F[k][0])) 
        slice_top = slice_bottom + 500
        delt = delta[k][0]*10**(-6)
        f = F[k][0]*10**(-9)
        popt, pcov = curve_fit(func_E, delt[slice_bottom:slice_top], f[slice_bottom:slice_top],  maxfev = 100000)
        popt_list.append(popt)
        print(k, popt[0])
        
        fig, ax = plt.subplots()
        ax.plot(delt, f, 'r') # F_bS[k][0]
        ax.plot(delt[slice_bottom:slice_top], func_E(delt[slice_bottom:slice_top], *popt), 'orange')
        ax.set(xlabel='tip-sample distance (m)', ylabel='force (N)', title='Force-delta curve %i' % k)
    
    # F = (4/3)*np.sqrt(R_c)*(E/(1-v**2))*delta**(3/2)
    # E = (parabolic_fit_param*3*(1-v**2))/(4*np.sqrt(R_c))
    return popt_list, fig

def PolyFit(F, d, plot='False', saveplot='False'):
    # two empty lists to store the gradients 'm' and constants 'b' of the linear fit function for each curve
    M = []
    B = []
    C = []
    argmin_list = contactPoint1(F, d)
    perc_top = 90
    for i in range(len(F)):
        slice_bottom = argmin_list[i]
        slice_top = round((perc_top/100)*len(F[i][0])) #compute first ..% of data set and round to nearest integer
        m,b,c = np.polyfit(d[i][0][slice_bottom:slice_top], F[i][0][slice_bottom:slice_top], 2) # linear fit of first ..% of dataset
        M.append(m) # store in lists
        B.append(b)
        C.append(c)
        
        if plot == 'True':
            x = d[i][0]
            lin_fit = m*x**2 + b*x + c
            fig, ax = plt.subplots()
            ax.plot(d[i][0], F[i][0], 'deepskyblue', label='force-distance curve')
            ax.plot(d[i][0][slice_bottom:slice_top], F[i][0][slice_bottom:slice_top], 'red', label='part of curve used in the poly fit')
            ax.plot(x[slice_bottom:slice_top], lin_fit[slice_bottom:slice_top], 'orange', label='Quadratic fit line')
            ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
            plt.legend(loc="upper right")
            if saveplot == 'True':
                fig.savefig('Results\Fd_baseline_linearfit_' + str(i) + '.png')
    
    return M, B, C