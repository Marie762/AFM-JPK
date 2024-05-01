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
    return c +  a * (x**b)

def func_parabolic(x, E, c0):
    v = 0.5
    R_c = 10e-9
    return c0 + (4/3)*np.sqrt(R_c)*(E/(1-v**2))*x**(3/2)

def func_conical(x, E, c0):
    v = 0.5
    alpha = np.pi/4
    return c0 + (E/(1-v**2))*((2*np.tan(alpha))/np.pi)*x**2

def func_pyramid(x, E, c0):
    v = 0.5
    theta = np.pi/4
    return c0 + ((np.tan(theta))/np.sqrt(2))*(E/(1-v**2))*x**2

def fitYoungsModulus(F, delta, argmin_list):
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
        popt, pcov = curve_fit(func_pyramid, delt[slice_bottom:slice_top], f[slice_bottom:slice_top],  maxfev = 100000)
        popt_list.append(popt)
        print(k, popt)
        
        fig, ax = plt.subplots()
        ax.plot(delt, f, 'r') # F_bS[k][0]
        ax.plot(delt[slice_bottom:slice_top], func_pyramid(delt[slice_bottom:slice_top], *popt), 'orange')
        ax.set(xlabel='tip-sample distance (m)', ylabel='force (N)', title='Force-delta curve %i' % k)
    
    # F = (4/3)*np.sqrt(R_c)*(E/(1-v**2))*delta**(3/2)
    # E = (parabolic_fit_param*3*(1-v**2))/(4*np.sqrt(R_c))
    return popt_list, fig

def variationYoungsModulus(F, delta, argmin_list, indenter='parabolic'):
    E = []
    v = 0.5
    R_c = 5e-6
    alpha = np.pi/4
    theta = np.pi/4
    for k in range(len(F)):
        E_local = []
        slice = argmin_list[k]
        delt = delta[k][0][slice:]*10**(-6)
        f = F[k][0][slice:]*10**(-9)
        for i in range(len(f)):
            if indenter == 'parabolic':
                e = ((3*(1-v**2)*f[i])/(4*np.sqrt(R_c)*(delt[i]**(3/2))))
            if indenter == 'conical':
                e = ((np.pi*(1-v**2))/(2*np.tan(alpha)))*f[i]*delt[i]**(-2)
            if indenter == 'pyramidal':
                e = ((np.sqrt(2)*(1-v**2))/(np.tan(theta)))*f[i]*delt[i]**(-2)
            E_local.append(e)
        E.append(np.array(E_local))
        
        fig, ax = plt.subplots()
        ax.plot(delt, E_local, 'r')
        ax.set(xlabel='tip-sample distance (m)', ylabel='local Youngs Modulus (Pa)', title='E-delta curve %i' % k)
    return E, fig