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

