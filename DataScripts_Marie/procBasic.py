# -*- coding: utf-8 -*-
"""
Created on Tue Apr 3 2024

@author: marie
"""
from scipy.signal import savgol_filter
import numpy as np


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
        F_bS_local.append(F[k][2] - min_value)
        F_bS.append(F_bS_local)
    return F_bS

def smoothingSG(F_approach, F_inter, F_retract, window_size, poly_order):
    F_smoothSG1 = []
    F_smoothSG2 = []
    F_smoothSG3 = []
    for k in range(len(F_approach)):
        if len(F_approach[k])>window_size:
            smoothed_data1 = savgol_filter(F_approach[k], window_size, poly_order) # smoothing Savitzky-Golay filter
        else:
            window_size = len(F_approach[k])
            if window_size > poly_order:
                smoothed_data1 = savgol_filter(F_approach[k], window_size, poly_order)
            else:
                poly_order = window_size - 1
                smoothed_data1 = savgol_filter(F_approach[k], window_size, poly_order)
        
        if len(F_inter[k])>window_size:
            smoothed_data2 = savgol_filter(F_inter[k], window_size, poly_order)
        else:
            window_size = len(F_inter[k])
            if window_size > poly_order:
                smoothed_data2 = savgol_filter(F_inter[k], window_size, poly_order)
            else:
                poly_order = window_size - 1
                smoothed_data2 = savgol_filter(F_inter[k], window_size, poly_order)
        
        if len(F_retract[k])>window_size:
            smoothed_data3 = savgol_filter(F_retract[k], window_size, poly_order)
        else:
            window_size = len(F_retract[k])
            if window_size > poly_order:
                smoothed_data3 = savgol_filter(F_retract[k], window_size, poly_order)
            else:
                poly_order = window_size - 1
                smoothed_data3 = savgol_filter(F_retract[k], window_size, poly_order)
        
        F_smoothSG1.append(smoothed_data1)
        F_smoothSG2.append(smoothed_data2)
        F_smoothSG3.append(smoothed_data3)
    return F_smoothSG1, F_smoothSG2, F_smoothSG3

