# -*- coding: utf-8 -*-
"""
Created on Friday May 10 2024

@author: marie
"""

import matplotlib.pylab as plt
import numpy as np
from scipy.signal import find_peaks
from contactPoint import contactPoint1, contactPoint2, contactPoint3


def substrateLinearFit(F, d, perc_bottom=98, plot=False, save=False):
    # two empty lists to store the gradients 'm' and constants 'b' of the linear fit function for each curve
    M = []
    B = []
    for i in range(len(F)):
        slice_bottom = round((perc_bottom/100)*len(F[i][0]))
        m,b = np.polyfit(d[i][0][slice_bottom:], F[i][0][slice_bottom:], 1) # linear fit of first ..% of dataset
        M.append(m) # store in lists
        B.append(b)
        
        if plot:
            x = d[i][0]
            lin_fit = m*x + b
            fig, ax = plt.subplots()
            ax.plot(d[i][0], F[i][0], 'deepskyblue', label='force-distance curve')
            ax.plot(x[slice_bottom:], lin_fit[slice_bottom:], 'orange', label='linear fit line')
            ax.plot(d[i][0][slice_bottom:], F[i][0][slice_bottom:], 'red', label='part of curve used in the linear fit')
            ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
            plt.legend(loc="upper right")
            if save:
                fig.savefig('Results\Fd_substrate_linearfit_' + str(i) + '.png')
            plt.close()
    return M, B

def substrateContact(F, d, contact_point_list, perc_bottom=98, plot=False, save=False):
    substrate_contact_list = []
    M, B = substrateLinearFit(F,d, perc_bottom=perc_bottom)
    plot_bottom = 95
    for i in range(len(F)):
        if M[i] < -5:
            difference_list = []
            for j in range(len(F[i][0])):
                f = M[i]*(d[i][0][j]) + B[i] # linear fit line
                difference_squared = (F[i][0][j] - f)**2 # the difference-swuared between the force value and the value of the linear fit line at each point
                difference_list.append(difference_squared)
            
            argmin_val = [i for i,el in enumerate(difference_list) if abs(el) < 0.01][0]
        
            substrate_contact_list.append(argmin_val)
            if plot:
                x = d[i][0]
                lin_fit = M[i]*x + B[i]
                slice_bottom = round((plot_bottom/100)*len(F[i][0]))
                fig, ax = plt.subplots()
                ax.plot(d[i][0], F[i][0], 'deepskyblue', label='force-distance curve')
                ax.plot(x[slice_bottom:], lin_fit[slice_bottom:], 'orange', label='linear fit line')
                ax.plot(d[i][0][argmin_val], F[i][0][argmin_val], 'go', label='hard substrate contact point estimation')
                ax.plot(d[i][0][contact_point_list[i]], F[i][0][contact_point_list[i]], 'ro', label='=contact point estimation')
                ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
                plt.legend(loc="upper right")
                if save:
                    fig.savefig('Results\Fd_substrate_contact_' + str(i) + '.png')
                plt.close()
                
        else:
            last_index = len(F[i][0]) - 1
            substrate_contact_list.append(last_index)
    
            if plot:
                x = d[i][0]
                lin_fit = M[i]*x + B[i]
                slice_bottom = round((plot_bottom/100)*len(F[i][0]))
                fig, ax = plt.subplots()
                ax.plot(d[i][0], F[i][0], 'deepskyblue', label='force-distance curve')
                ax.plot(x[slice_bottom:], lin_fit[slice_bottom:], 'orange', label='linear fit line')
                ax.plot(d[i][0][last_index], F[i][0][last_index], 'go', label='hard substrate contact point estimation')
                ax.plot(d[i][0][contact_point_list[i]], F[i][0][contact_point_list[i]], 'ro', label='=contact point estimation')
                ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
                plt.legend(loc="upper left")
                if save:
                    fig.savefig('Results\Fd_substrate_contact_' + str(i) + '.png')
                plt.close()
    return substrate_contact_list

def findPeaks(F, d, contact_point_list, plot=False, saveplot=False):
    first_peak_list = []
    number_of_peaks_list = []
    all_peaks_list = []
    for k in range(len(F)):
        peaks, properties = find_peaks(F[k][0][contact_point_list[k]:], prominence=0.1)
        peaks = peaks + contact_point_list[k]
        if len(peaks) != 0:
            first_peak_list.append(peaks[0])
            number_of_peaks_list.append(len(peaks))
            all_peaks_list.append(peaks)
            if plot:
                fig, ax = plt.subplots()
                ax.plot(d[k][0], F[k][0], 'deepskyblue', label='force-distance curve')
                ax.plot(d[k][0][peaks], F[k][0][peaks], 'yo', label='peaks identified')
                ax.plot(d[k][0][peaks[0]], F[k][0][peaks[0]], 'bo', label='first peak identified')
                ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % k)
                ax.text(2.5, 3, '# of peaks = %i' % len(peaks), fontsize=12)
                plt.legend(loc="upper right")
                if saveplot:
                    fig.savefig('Results\Fd_find_peaks_' + str(k) + '.png')
                plt.close()
        else:
            first_peak_list.append(None)
            number_of_peaks_list.append(0)
            all_peaks_list.append(None)
            if plot:
                fig, ax = plt.subplots()
                ax.plot(d[k][0], F[k][0], 'deepskyblue', label='force-distance curve')
                ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % k)
                ax.text(2.5, 3, '# of peaks = 0', fontsize=12)
                plt.legend(loc="upper right")
                if saveplot:
                    fig.savefig('Results\Fd_find_peaks_' + str(k) + '.png')
                plt.close()
    return first_peak_list, number_of_peaks_list, all_peaks_list

def indentationDepth(F, d, contact_point_list, first_peak_list):
    indentation_depth_arr = np.zeros(len(d)) 
    for k in range(len(d)): 
        if first_peak_list[k]:   
            indentation_depth = d[k][0][contact_point_list[k]] - d[k][0][first_peak_list[k]]
            indentation_depth_arr[k] = indentation_depth
    return indentation_depth_arr

## don't know if I will still use this:
# def penetrationPoint(F, d, plot='False', saveplot='False'):
#     substrate_contact_list = substrateContact(F, d)
#     contact_point_list = contactPoint3(F,d)
#     P0,P1,P2,P3,P4,P5 = [],[],[],[],[],[]
#     dP_real_roots, dP2_real_roots = [],[]
#     for k in range(len(F)):
#         slice_top = substrate_contact_list[k]
#         slice_bottom = contact_point_list[k]
#         p0,p1,p2,p3,p4,p5 = np.polyfit(d[k][0][slice_bottom:slice_top], F[k][0][slice_bottom:slice_top], 5) #,p4,p5
#         P0.append(p0) # store in lists
#         P1.append(p1)
#         P2.append(p2)
#         P3.append(p3)
#         P4.append(p4)
#         P5.append(p5)
        
#         dP = np.polyder(np.poly1d([p0,p1,p2,p3,p4,p5]), 1)
#         dP_roots_local = np.roots(dP)
#         print(k, dP_roots_local)
#         for i in range(len(dP_roots_local)):
#             dP_real_imag = [el for i,el in enumerate(dP_roots_local) if abs(np.imag(el)) < 1e-5]
#             dP_real = np.real(dP_real_imag)
#         dP_real_roots.append(dP_real)
        
#         dP2 = np.polyder(np.poly1d([p0,p1,p2,p3,p4,p5]), 2)
#         dP2_roots_local = np.roots(dP2)
#         print(k,dP2_roots_local)
#         for i in range(len(dP2_roots_local)):
#             dP2_real_imag = [el for i,el in enumerate(dP2_roots_local) if abs(np.imag(el)) < 1e-5]
#             dP2_real = np.real(dP2_real_imag)
#         dP2_real_roots.append(dP2_real)

#         if plot == 'True':
#             x = d[k][0]
#             poly_fit = p0*x**5 + p1*x**4 + p2*x**3 + p3*x**2 + p4*x + p5
#             fig, ax = plt.subplots()
#             ax.plot(d[k][0], F[k][0], 'deepskyblue', label='force-distance curve')
#             ax.plot(d[k][0][slice_bottom:slice_top], F[k][0][slice_bottom:slice_top], 'red', label='part of curve used in the poly-fit')
#             ax.plot(x[slice_bottom:slice_top], poly_fit[slice_bottom:slice_top], 'orange', label='5th order polynomial fit')
#             ax.plot(x[slice_bottom:slice_top], dP(x[slice_bottom:slice_top]), 'green', label='1st derivative of polynomial fit')
#             ax.plot(x[slice_bottom:slice_top], dP2(x[slice_bottom:slice_top]), 'purple', label='2nd derivative of polynomial fit')
#             ax.grid()
#             ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % k)
#             plt.legend(loc="upper left")
#             if saveplot == 'True':
#                 fig.savefig('Results\Fd_3rd_order_polyfit_' + str(k) + '.png')
    
#     return dP_real_roots, dP2_real_roots