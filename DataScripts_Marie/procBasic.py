# -*- coding: utf-8 -*-
"""
Created on Tue Apr 3 2024

@author: marie
"""
from scipy.signal import savgol_filter
import numpy as np


def max(F_approach):
    max_value = []
    max_element = []
    for k in range(len(F_approach)):
        max_value.append(np.ndarray.max(F_approach[k]))
        max_element.append(np.argmax(F_approach[k]))
    return max_value, max_element

def baselineSubtraction(F_approach, F_inter, F_retract):
    F_bS1 = []
    F_bS2 = []
    F_bS3 = []
    for k in range(len(F_approach)):
        min_value = np.ndarray.min(F_approach[k])
        F_bS1.append(F_approach[k] - min_value)
        F_bS2.append(F_inter[k]- min_value)
        F_bS3.append(F_retract[k]- min_value)
    return F_bS1, F_bS2, F_bS3

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

