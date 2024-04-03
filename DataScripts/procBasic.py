# -*- coding: utf-8 -*-
"""
Created on Tue Apr 3 2024

@author: marie
"""
from scipy.signal import savgol_filter
import numpy as np


def max(F):
    max_value = max(F)
    max_element = np.argmax(F)
    return max_value, max_element

def baselineSubtraction(F):
    for i in range(len(F)):
        
    o = min(F)
    F_bS = F - o
    return F_bS

def smoothingSG(F, window_size, poly_order):
    F_smoothSG = savgol_filter(F, window_size, poly_order) # smoothing Savitzky-Golay filter
    return F_smoothSG

