# -*- coding: utf-8 -*-
"""
Created on Tue Apr 3 2024

@author: marie
"""
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pylab as plt


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

def baselineLinearFit(F, d, percentage=50, plot='False', saveplot='False'):
    # two empty lists to store the gradients 'm' and constants 'b' of the linear fit function for each curve
    M = []
    B = []
    for i in range(len(F)):
        percentage_slice = round((percentage/100)*len(F[i][0])) #compute first ..% of data set and round to nearest integer
        m,b = np.polyfit(d[i][0][:percentage_slice], F[i][0][:percentage_slice], 1) # linear fit of first ..% of dataset
        M.append(m) # store in lists
        B.append(b)
        
        if plot == 'True':
            x = d[i][0]
            lin_fit = m*x + b
            fig, ax = plt.subplots()
            ax.plot(d[i][0], F[i][0], 'deepskyblue', label='force-distance curve')
            ax.plot(x, lin_fit, 'orange', label='linear fit line')
            ax.plot(d[i][0][:percentage_slice], F[i][0][:percentage_slice], 'red', label='part of curve used in the linear fit')
            ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
            plt.legend(loc="upper right")
            if saveplot == 'True':
                fig.savefig('Results\Fd_baseline_linearfit_' + str(i) + '.png')
    
    return M, B

def contactPoint(F, d, plot='False', saveplot='False'):
    F_bS = baselineSubtraction(F)
    M, B = baselineLinearFit(F_bS, d)
    # empty list to store the index of the last intersection point of the F-d curve with the linear fit line 
    argmin_list = []
    for i in range(len(F_bS)):
        difference_list = []
        for j in range(len(F_bS[i][0])):
            f = M[i]*(d[i][0][j]) + B[i] # linear fit line
            difference_squared = (F_bS[i][0][j] - f)**2 # the difference-swuared between the force value and the value of the linear fit line at each point
            difference_list.append(difference_squared)

        argmin_val = [i for i,el in enumerate(difference_list) if abs(el) < 0.001][-1] # index of the last intersection point of the F-d curve with the linear fit line 
        argmin_list.append(argmin_val)

        if plot == 'True':
            fig, ax = plt.subplots()
            ax.plot(d[i][0], F_bS[i][0], 'deepskyblue', label='force-distance curve')
            ax.plot(d[i][0], M[i]*(d[i][0]) + B[i], 'orange', label='linear fit line')
            ax.plot(d[i][0][argmin_val], F_bS[i][0][argmin_val], 'ro', label='contact point estimation')
            ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
            plt.legend(loc="upper right")
            if saveplot == 'True':
                fig.savefig('Results\Fd_contact_point_' + str(i) + '.png')
    
    return argmin_list