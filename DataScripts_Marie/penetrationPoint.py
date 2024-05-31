# -*- coding: utf-8 -*-
"""
Created on Friday May 10 2024

@author: marie
"""

import matplotlib.pylab as plt
import numpy as np
from contactPoint import contactPoint1, contactPoint2, contactPoint3


def substrateLinearFit(F, d, perc_bottom=98, plot='False', saveplot='False'):
    # two empty lists to store the gradients 'm' and constants 'b' of the linear fit function for each curve
    M = []
    B = []
    for i in range(len(F)):
        slice_bottom = round((perc_bottom/100)*len(F[i][0]))
        m,b = np.polyfit(d[i][0][slice_bottom:], F[i][0][slice_bottom:], 1) # linear fit of first ..% of dataset
        M.append(m) # store in lists
        B.append(b)
        
        if plot == 'True':
            x = d[i][0]
            lin_fit = m*x + b
            fig, ax = plt.subplots()
            ax.plot(d[i][0], F[i][0], 'deepskyblue', label='force-distance curve')
            ax.plot(x[slice_bottom:], lin_fit[slice_bottom:], 'orange', label='linear fit line')
            ax.plot(d[i][0][slice_bottom:], F[i][0][slice_bottom:], 'red', label='part of curve used in the linear fit')
            ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
            plt.legend(loc="upper left")
            if saveplot == 'True':
                fig.savefig('Results\Fd_substrate_linearfit_' + str(i) + '.png')
    
    return M, B

def substrateContact(F, d, perc_bottom=98, plot='False', saveplot='False'):
    substrate_contact_list = []
    M, B = substrateLinearFit(F,d, perc_bottom=perc_bottom)
    plot_bottom = 96
    for i in range(len(F)):
        difference_list = []
        for j in range(len(F[i][0])):
            f = M[i]*(d[i][0][j]) + B[i] # linear fit line
            difference_squared = (F[i][0][j] - f)**2 # the difference-swuared between the force value and the value of the linear fit line at each point
            difference_list.append(difference_squared)
        
        argmin_val = [i for i,el in enumerate(difference_list) if abs(el) < 500][0]
    
        substrate_contact_list.append(argmin_val)
    
        if plot == 'True':
            x = d[i][0]
            lin_fit = M[i]*x + B[i]
            slice_bottom = round((plot_bottom/100)*len(F[i][0]))
            fig, ax = plt.subplots()
            ax.plot(d[i][0], F[i][0], 'deepskyblue', label='force-distance curve')
            ax.plot(x[slice_bottom:], lin_fit[slice_bottom:], 'orange', label='linear fit line')
            ax.plot(d[i][0][argmin_val], F[i][0][argmin_val], 'ro', label='hard substrate contact point estimation')
            ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
            plt.legend(loc="upper left")
            if saveplot == 'True':
                fig.savefig('Results\Fd_substrate_contact_' + str(i) + '.png')
    
    return substrate_contact_list

def penetrationPoint(F, d, plot='False', saveplot='False'):
    substrate_contact_list = substrateContact(F, d)
    contact_point_list = contactPoint3(F,d)
    P0,P1,P2,P3,P4,P5 = [],[],[],[],[],[]
    dP_real_roots, dP2_real_roots = [],[]
    for k in range(len(F)):
        slice_top = substrate_contact_list[k]
        slice_bottom = contact_point_list[k]
        p0,p1,p2,p3,p4,p5 = np.polyfit(d[k][0][slice_bottom:slice_top], F[k][0][slice_bottom:slice_top], 5) #,p4,p5
        P0.append(p0) # store in lists
        P1.append(p1)
        P2.append(p2)
        P3.append(p3)
        P4.append(p4)
        P5.append(p5)
        
        dP = np.polyder(np.poly1d([p0,p1,p2,p3,p4,p5]), 1)
        dP_roots_local = np.roots(dP)
        print(k, dP_roots_local)
        for i in range(len(dP_roots_local)):
            dP_real_imag = [el for i,el in enumerate(dP_roots_local) if abs(np.imag(el)) < 1e-5]
            dP_real = np.real(dP_real_imag)
        dP_real_roots.append(dP_real)
        
        dP2 = np.polyder(np.poly1d([p0,p1,p2,p3,p4,p5]), 2)
        dP2_roots_local = np.roots(dP2)
        print(k,dP2_roots_local)
        for i in range(len(dP2_roots_local)):
            dP2_real_imag = [el for i,el in enumerate(dP2_roots_local) if abs(np.imag(el)) < 1e-5]
            dP2_real = np.real(dP2_real_imag)
        dP2_real_roots.append(dP2_real)

        if plot == 'True':
            x = d[k][0]
            poly_fit = p0*x**5 + p1*x**4 + p2*x**3 + p3*x**2 + p4*x + p5
            fig, ax = plt.subplots()
            ax.plot(d[k][0], F[k][0], 'deepskyblue', label='force-distance curve')
            ax.plot(d[k][0][slice_bottom:slice_top], F[k][0][slice_bottom:slice_top], 'red', label='part of curve used in the poly-fit')
            ax.plot(x[slice_bottom:slice_top], poly_fit[slice_bottom:slice_top], 'orange', label='5th order polynomial fit')
            ax.plot(x[slice_bottom:slice_top], dP(x[slice_bottom:slice_top]), 'green', label='1st derivative of polynomial fit')
            ax.plot(x[slice_bottom:slice_top], dP2(x[slice_bottom:slice_top]), 'purple', label='2nd derivative of polynomial fit')
            ax.grid()
            ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % k)
            plt.legend(loc="upper left")
            if saveplot == 'True':
                fig.savefig('Results\Fd_3rd_order_polyfit_' + str(k) + '.png')
    
    return dP_real_roots, dP2_real_roots