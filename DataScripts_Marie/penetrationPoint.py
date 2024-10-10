# -*- coding: utf-8 -*-
"""
Created on Friday May 10 2024

@author: marie
"""

from calendar import isleap
import matplotlib.pylab as plt
import numpy as np
from scipy.signal import find_peaks
from contactPoint import contactPoint1, contactPoint2, contactPoint3


def substrateLinearFit(F, d, perc_bottom=98, plot=False, save=False):
    # two empty lists to store the gradients 'm' and constants 'b' of the linear fit function for each curve
    M = []
    B = []
    for i in range(len(F)):
        if len(F[i][0]) > 300:
            slice_bottom = round((perc_bottom/100)*len(F[i][0]))
            m,b = np.polyfit(d[i][0][slice_bottom:], F[i][0][slice_bottom:], 1) # linear fit of first ..% of dataset
            # print(i, m)
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
        else:
            M.append(None) 
            B.append(None)
            if plot:
                fig, ax = plt.subplots()
                ax.plot(0, 0, 'deepskyblue')
                ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i has no extend curve' % i)
                plt.xlim(0,6)
                plt.ylim(0,6)
                if save:
                    fig.savefig('Results\Fd_substrate_linearfit_' + str(i) + '.png')
                plt.close()
    return M, B

def substrateContact(F, d, contact_point_list, perc_bottom=98, plot=False, save=False):
    substrate_contact_list = []
    M, B = substrateLinearFit(F,d, perc_bottom=perc_bottom)
    plot_bottom = 95
    for i in range(len(F)):
        if contact_point_list[i]:
            #if M[i] < -1:
            difference_list = []
            for j in range(len(F[i][0])):
                f = M[i]*(d[i][0][j]) + B[i] # linear fit line
                difference_squared = (F[i][0][j] - f)**2 # the difference-swuared between the force value and the value of the linear fit line at each point
                difference_list.append(difference_squared)
            
            argmin_val = [i for i,el in enumerate(difference_list) if abs(el) < 0.01]
            if argmin_val is None or len(argmin_val) == 0:
                argmin_val = len(F[i][0]) - 1
            else:
                argmin_val = argmin_val[0]
        
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
                
            # else:
            #     last_index = len(F[i][0]) - 1
            #     substrate_contact_list.append(last_index)
        
            #     if plot:
            #         x = d[i][0]
            #         lin_fit = M[i]*x + B[i]
            #         slice_bottom = round((plot_bottom/100)*len(F[i][0]))
            #         fig, ax = plt.subplots()
            #         ax.plot(d[i][0], F[i][0], 'deepskyblue', label='force-distance curve')
            #         ax.plot(x[slice_bottom:], lin_fit[slice_bottom:], 'orange', label='linear fit line')
            #         ax.plot(d[i][0][last_index], F[i][0][last_index], 'go', label='hard substrate contact point estimation')
            #         ax.plot(d[i][0][contact_point_list[i]], F[i][0][contact_point_list[i]], 'ro', label='=contact point estimation')
            #         ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
            #         plt.legend(loc="upper right")
            #         if save:
            #             fig.savefig('Results\Fd_substrate_contact_' + str(i) + '.png')
            #         plt.close()
        else:
            substrate_contact_list.append(0)
            if plot:
                fig, ax = plt.subplots()
                ax.plot(0, 0, 'deepskyblue')
                ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i has no extend curve' % i)
                plt.xlim(0,6)
                plt.ylim(0,6)
                if save:
                    fig.savefig('Results\Fd_substrate_contact_' + str(i) + '.png')
                plt.close()
    return substrate_contact_list

def substrateContact2(F, d, contact_point_list, plot=False, save=False):
    substrate_contact_list = []
    for i in range(len(F)):
        last_index = len(F[i][0]) - 1
        substrate_contact_list.append(last_index)
        if plot:
            fig, ax = plt.subplots()
            ax.plot(d[i][0], F[i][0], 'deepskyblue', label='force-distance curve')
            ax.plot(d[i][0][last_index], F[i][0][last_index], 'go', label='hard substrate contact point estimation')
            ax.plot(d[i][0][contact_point_list[i]], F[i][0][contact_point_list[i]], 'ro', label='=contact point estimation')
            ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
            plt.legend(loc="upper right")
            if save:
                fig.savefig('Results\Fd_substrate_contact_' + str(i) + '.png')
            plt.close()
    return substrate_contact_list

def findPeaks(F, d, contact_point_list, prominence=0.2, plot=False, save=False):
    first_peak_list = []
    number_of_peaks_list = []
    all_peaks_list = []
    peak_heights_list, right_bases_list = [], []
    # properties_list = []
    for k in range(len(F)):
        if contact_point_list[k]:
            peaks, properties = find_peaks(F[k][0][contact_point_list[k]+100:], width=(None, None), distance=100, prominence=prominence, height=(None, None)) #distance=100
            if len(peaks) != 0:
                peaks = peaks + contact_point_list[k] +100
                width = properties["widths"]
                peak_heights = properties["peak_heights"]
                right_bases = properties["right_bases"] + contact_point_list[k] +100
                # for m in range(len(peaks)):
                #     right_bases[m] = right_bases[m] + contact_point_list[k]
                
                peaks_filtered = []
                peak_heights_filtered = []
                right_bases_filtered = []
                for p in range(len(peaks)):
                    if width[p]>50:
                        peaks_filtered.append(peaks[p])
                        peak_heights_filtered.append(peak_heights[p])
                        right_bases_filtered.append(right_bases[p])
                            
                if peaks_filtered:
                    first_peak_list.append(peaks_filtered[0])
                    number_of_peaks_list.append(len(peaks_filtered))
                    all_peaks_list.append(peaks_filtered)
                    peak_heights_list.append(peak_heights_filtered)
                    right_bases_list.append(right_bases_filtered)

                    if plot:
                        fig, ax = plt.subplots()
                        ax.plot(d[k][0], F[k][0], 'deepskyblue', label='force-distance curve', linewidth=4)
                        ax.plot(d[k][0][peaks_filtered], F[k][0][peaks_filtered], 'yo', markersize=8, label='peaks identified')
                        ax.plot(d[k][0][peaks_filtered[0]], F[k][0][peaks_filtered[0]], 'bo', markersize=8, label='first peak identified')
                        ax.plot(d[k][0][contact_point_list[k]], F[k][0][contact_point_list[k]], 'ro', markersize=8, label='contact point estimation')
                        ax.set_xlabel(u'Distance (\u03bcm)', fontsize=15)
                        ax.set_ylabel('Force (nN)', fontsize=15)
                        plt.tick_params(axis='both', which='major', labelsize=15)
                        ax.text(2.5, 3, '# of peaks = %i' % len(peaks_filtered), fontsize=15)
                        plt.legend(loc="upper right", prop={'size': 15})
                        if save:
                            fig.savefig('Results\Fd_find_peaks_' + str(k) + '.png')
                            fig.savefig('Results\Fd_find_peaks_' + str(k) + '.pdf', format='pdf')
                        plt.close()
                else:
                    first_peak_list.append(None)
                    number_of_peaks_list.append(0)
                    all_peaks_list.append(None)
                    peak_heights_list.append([])
                    right_bases_list.append([])
                    if plot:
                        fig, ax = plt.subplots()
                        ax.plot(d[k][0], F[k][0], 'deepskyblue', label='force-distance curve', linewidth=4)
                        ax.plot(d[k][0][contact_point_list[k]], F[k][0][contact_point_list[k]], 'ro', markersize=8, label='contact point estimation')
                        ax.set_xlabel(u'Distance (\u03bcm)', fontsize=15)
                        ax.set_ylabel('Force (nN)', fontsize=15)
                        plt.tick_params(axis='both', which='major', labelsize=15)
                        ax.text(2.5, 3, '# of peaks = 0', fontsize=15)
                        plt.legend(loc="upper right", prop={'size': 15})
                        if save:
                            fig.savefig('Results\Fd_find_peaks_' + str(k) + '.png')
                            fig.savefig('Results\Fd_find_peaks_' + str(k) + '.pdf', format='pdf')
                        plt.close()
            else:
                first_peak_list.append(None)
                number_of_peaks_list.append(0)
                all_peaks_list.append(None)
                peak_heights_list.append([])
                right_bases_list.append([])
                if plot:
                    fig, ax = plt.subplots()
                    ax.plot(d[k][0], F[k][0], 'deepskyblue', label='force-distance curve', linewidth=4)
                    ax.plot(d[k][0][contact_point_list[k]], F[k][0][contact_point_list[k]], 'ro', markersize=8, label='contact point estimation')
                    ax.set_xlabel(u'Distance (\u03bcm)', fontsize=15)
                    ax.set_ylabel('Force (nN)', fontsize=15)
                    ax.text(2.5, 3, '# of peaks = 0', fontsize=15)
                    plt.tick_params(axis='both', which='major', labelsize=15)
                    plt.legend(loc="upper right", prop={'size': 15})
                    if save:
                        fig.savefig('Results\Fd_find_peaks_' + str(k) + '.png')
                        fig.savefig('Results\Fd_find_peaks_' + str(k) + '.pdf', format='pdf')
                    plt.close()
        else:
            first_peak_list.append(None)
            number_of_peaks_list.append(0)
            all_peaks_list.append(None)
            peak_heights_list.append([])
            right_bases_list.append([])
            if plot:
                fig, ax = plt.subplots()
                ax.plot(0, 0, 'deepskyblue')
                ax.set(xlabel=u'Distance (\u03bcm)', ylabel='Force (nN)', title='Force-distance curve %i has no extend curve' % k)
                plt.xlim(0,6)
                plt.ylim(0,6)
                plt.tick_params(axis='both', which='major', labelsize=15)
                if save:
                    fig.savefig('Results\Fd_find_peaks_' + str(k) + '.png')
                    fig.savefig('Results\Fd_find_peaks_' + str(k) + '.pdf', format='pdf')
                plt.close()
                
    return first_peak_list, number_of_peaks_list, all_peaks_list, peak_heights_list, right_bases_list

def forceDrop(F, d, first_peak_list,peak_heights_list, right_bases_list, plot=True):
    penetration_force_list = []
    right_bases_list2 = []
    force_drop_list = []
    first_penetration_force_list = []
    first_force_drop_list = []
    
    for k in range(len(F)):
        peak_height = peak_heights_list[k] # penetration force
        right_bases = right_bases_list[k] # index of base of peak to the right (for this case it would seem to the left)
        val_list = []
        indx_list = []
        force_drop_local = []
        for m in range(len(right_bases)):
            indx = right_bases[m]
            indx_list.append(indx)
            val = F[k][0][indx] # value of base of peak to the right
            val_list.append(val)
            force_drop = peak_height[m] - val # the force drop is the difference between the penetration force and the base of the peak to the right 
            force_drop_local.append(force_drop)
   
        penetration_force_list.append(peak_height)
        right_bases_list2.append(val_list)
        force_drop_list.append(force_drop_local)
        
        if len(force_drop_local) >= 1:
            first_penetration_force = peak_height[0]
            first_force_drop = force_drop_local[0]
        else:
            first_penetration_force = 0
            first_force_drop = 0
        
        first_penetration_force_list.append(first_penetration_force)
        first_force_drop_list.append(first_force_drop)
        
        if first_peak_list[k]:
            limit = 500
            
            lower_x_lim = indx_list[0] - limit
            lower_y_lim = indx_list[0] - limit
            if lower_x_lim < 0:
                lower_x_lim = 0
                lower_y_lim = lower_x_lim
            
            upper_x_lim = indx_list[0] + limit
            upper_y_lim = indx_list[0] + limit
            if upper_x_lim >= len(d[k][0]):
                upper_x_lim = len(d[k][0]) - 1
                upper_y_lim = upper_x_lim
            
            if plot and k == 6:
                fig, ax = plt.subplots()
                ax.plot(d[k][0], F[k][0], 'deepskyblue', linewidth=4, label='force-distance curve')
                ax.plot(d[k][0][first_peak_list[k]], F[k][0][first_peak_list[k]], 'bo', markersize=8, label='first peak')
                ax.plot(d[k][0][indx_list[0]], F[k][0][indx_list[0]], 'yo', markersize=8, label='base of peak')
                ax.set_xlabel(u'Distance (\u03bcm)', fontsize=15)
                ax.set_ylabel('Force (nN)', fontsize=15)
                ax.text(d[k][0][lower_x_lim], F[k][0][lower_y_lim], 'Force drop = %.3f' % first_force_drop, fontsize=15)
                plt.xlim(d[k][0][lower_x_lim], d[k][0][upper_x_lim])
                plt.ylim(F[k][0][lower_y_lim], F[k][0][upper_y_lim])
                # plt.legend(loc="upper left", prop={'size': 12})
                plt.tick_params(axis='both', which='major', labelsize=15)
                fig.savefig('Results\Fd_first_force_drop_' + str(k) + '.png')
                fig.savefig('Results\Fd_first_force_drop_' + str(k) + '.pdf', format='pdf')
                plt.close()  
    return penetration_force_list, first_penetration_force_list, right_bases_list2, force_drop_list, first_force_drop_list

def indentationDepth(F, d, contact_point_list, first_peak_list):
    indentation_depth_arr = np.zeros(len(d)) 
    for k in range(len(d)): 
        if contact_point_list[k]:
            if first_peak_list[k]:   
                indentation_depth = d[k][0][contact_point_list[k]] - d[k][0][first_peak_list[k]]
                indentation_depth_arr[k] = indentation_depth
        else:
            indentation_depth_arr[k] = 0
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