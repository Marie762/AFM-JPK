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
        max_value.append(np.ndarray.max(F[k]))
        max_element.append(np.argmax(F[k]))
    return max_value, max_element

def baselineSubtraction(F):
    F_bS = []
    for k in range(len(F)):
        min_value = np.ndarray.min(F[k])
        F_bS.append(F[k] - min_value)
    return F_bS

def smoothingSG(F, window_size, poly_order):
    F_smoothSG = []
    for k in range(len(F)):
        smoothed_data = savgol_filter(F[k], window_size, poly_order) # smoothing Savitzky-Golay filter
        F_smoothSG.append(smoothed_data)
    return F_smoothSG

