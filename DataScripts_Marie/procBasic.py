# -*- coding: utf-8 -*-
"""
Created on Tue Apr 3 2024

@author: marie
"""
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pylab as plt
from metadata import Sensitivity, SpringConstant


def max(F):
    max_value = []
    max_element = []
    for k in range(len(F)):
        max_value.append(np.ndarray.max(F[k][0]))
        max_element.append(np.argmax(F[k][0]))
    return max_value, max_element

def baselineSubtraction(F):
    F_bS = []
    for k in range(len(F)):
        F_bS_local = []
        min_value = np.ndarray.min(F[k][0])
        F_bS_local.append(F[k][0] - min_value)
        F_bS_local.append(F[k][1] - min_value)
        if len(F[k]) > 2:
            F_bS_local.append(F[k][2] - min_value)
        F_bS.append(F_bS_local)
    return F_bS

def baselineSubtraction2(F):
    F_bS = []
    for k in range(len(F)):
        F_bS_local = []
        slice_top = round((50/100)*len(F[k][0]))
        min_value = np.ndarray.min(F[k][0][:slice_top])
        F_bS_local.append(F[k][0] - min_value)
        F_bS_local.append(F[k][1] - min_value)
        if len(F[k]) > 2:
            F_bS_local.append(F[k][2] - min_value)
        F_bS.append(F_bS_local)
    return F_bS

def heightCorrection(d):
    d_hC = []
    for k in range(len(d)):
        d_hC_local = []
        value_first_element = d[k][0][0]
        d_hC_local.append(-(d[k][0] - value_first_element))
        d_hC_local.append(-(d[k][1] - value_first_element))
        if len(d[k]) > 2:
            d_hC_local.append(-(d[k][2] - value_first_element))
        d_hC.append(d_hC_local)
    return d_hC

def heightCorrection2(d):
    d_hC = []
    for k in range(len(d)):
        d_hC_local = []
        value_last_element = d[k][0][-1]
        d_hC_local.append((d[k][0] - value_last_element))
        d_hC_local.append((d[k][1] - value_last_element))
        if len(d[k]) > 2:
            d_hC_local.append((d[k][2] - value_last_element))
        d_hC.append(d_hC_local)
    return d_hC

def heightZeroAtContactPoint(d, contact_point_list):
    d_hZ = []
    for k in range(len(d)):
        d_hZ_local = []
        if len(d[k][0]) > 300:
            contact_point_arg = contact_point_list[k]
            contact_point_value = d[k][0][contact_point_arg]
            d_hZ_local.append(d[k][0] - contact_point_value)
            d_hZ_local.append(d[k][1] - contact_point_value)
            if len(d[k]) > 2:
                d_hZ_local.append(d[k][2] - contact_point_value)
            d_hZ.append(d_hZ_local)
        else:
            d_hZ.append(None)
    return d_hZ

def sensitivityCorrection(F, new_sensitivity):
    F_corr = []
    sensitivity_list = Sensitivity()
    sensitivity_correction_factor = new_sensitivity/sensitivity_list[0]
    print('Sensitivity from file: ', sensitivity_list[0])
    print('Actual sensitivity: ', new_sensitivity)
    print('Correction factor: ', sensitivity_correction_factor)
    for k in range(len(F)):
        F_corr_local = []
        F_corr_local.append(F[k][0]*sensitivity_correction_factor*10)
        F_corr_local.append(F[k][1]*sensitivity_correction_factor*10)
        if len(F[k]) > 2:
            F_corr_local.append(F[k][2]*sensitivity_correction_factor*10)
        
        F_corr.append(F_corr_local)  
    return F_corr

def tipDisplacement(F,d, plot=False, save=False):
    delta = []
    spring_constant_list = SpringConstant()
    
    for k in range(len(F)):
        delta_local = []
        stiffness = spring_constant_list[k]*10
        
        f0 = F[k][0]*10**(-9)
        f1 = F[k][1]*10**(-9)
        z0 = d[k][0]*10**(-6)
        z1 = d[k][1]*10**(-6)
        
        deflection0 = f0/stiffness
        deflection1 = f1/stiffness
        delta0 = z0 + deflection0
        delta1 = z1 + deflection1
        
        delta_local.append(delta0*10**(6))
        delta_local.append(delta1*10**(6))
        
        if len(d[k]) > 2:
            f2 = F[k][2]*10**(-9)
            z2 = d[k][2]*10**(-6)
            
            deflection2 = f2/stiffness
            delta2 = z2 + deflection2
            delta_local.append(delta2*10**(6))
        
        delta.append(delta_local)
        
        if plot:
            fig, ax = plt.subplots()
            ax.plot(d[k][0], F[k][0], 'deepskyblue')
            ax.plot(d[k][1], F[k][1], 'deepskyblue')
            ax.plot(delta[k][0], F[k][0], 'r')
            ax.plot(delta[k][1], F[k][1], 'r')
            ax.set(xlabel='height measured (blue) and indentation (red) (um)', ylabel='force (nN)', title='Force-delta curve %i' % k)
            if save:
                fig.savefig('Results\Fdelta_' + str(k) + '.png')
            plt.close()
    return delta


def smoothingSG(F, window_size, poly_order):
    F_smoothSG = []
    for k in range(len(F)):
        F_smoothSG_local = []
        if len(F[k][0]) > window_size:
            smoothed_data1 = savgol_filter(F[k][0], window_size, poly_order) # smoothing Savitzky-Golay filter
        else:
            window_size1 = len(F[k][0])
            if window_size1 > poly_order:
                smoothed_data1 = savgol_filter(F[k][0], window_size1, poly_order)
            else:
                poly_order1 = window_size1 - 1
                smoothed_data1 = savgol_filter(F[k][0], window_size1, poly_order1)
        F_smoothSG_local.append(smoothed_data1)

        if len(F[k][1]) > window_size:
            smoothed_data2 = savgol_filter(F[k][1], window_size, poly_order)
        else:
            window_size2 = len(F[k][1])
            if window_size2 > poly_order:
                smoothed_data2 = savgol_filter(F[k][1], window_size2, poly_order)
            else:
                poly_order2 = window_size2 - 1
                smoothed_data2 = savgol_filter(F[k][1], window_size2, poly_order2)
        F_smoothSG_local.append(smoothed_data2)

        if len(F[k]) > 2:
            if len(F[k][2]) > window_size:
                smoothed_data3 = savgol_filter(F[k][2], window_size, poly_order)
            else:
                window_size3 = len(F[k][2])
                if window_size3 > poly_order:
                    smoothed_data3 = savgol_filter(F[k][2], window_size3, poly_order)
                else:
                    poly_order3 = window_size3 - 1
                    smoothed_data3 = savgol_filter(F[k][2], window_size3, poly_order3)
            F_smoothSG_local.append(smoothed_data3)

        F_smoothSG.append(F_smoothSG_local)
        
    return F_smoothSG

