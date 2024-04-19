# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 2024

@author: marie
"""

import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import pandas as pd
import procBasic
import contactPoint

def PolyFit(F, d, plot='False', saveplot='False'):
    # two empty lists to store the gradients 'm' and constants 'b' of the linear fit function for each curve
    M = []
    B = []
    C = []
    argmin_list = contactPoint.contactPoint1(F, d)
    perc_top = 90
    for i in range(len(F)):
        slice_bottom = argmin_list[i]
        slice_top = round((perc_top/100)*len(F[i][0])) #compute first ..% of data set and round to nearest integer
        m,b,c = np.polyfit(d[i][0][slice_bottom:slice_top], F[i][0][slice_bottom:slice_top], 2) # linear fit of first ..% of dataset
        M.append(m) # store in lists
        B.append(b)
        C.append(c)
        
        if plot == 'True':
            x = d[i][0]
            lin_fit = m*x**2 + b*x + c
            fig, ax = plt.subplots()
            ax.plot(d[i][0], F[i][0], 'deepskyblue', label='force-distance curve')
            ax.plot(d[i][0][slice_bottom:slice_top], F[i][0][slice_bottom:slice_top], 'red', label='part of curve used in the poly fit')
            ax.plot(x[slice_bottom:slice_top], lin_fit[slice_bottom:slice_top], 'orange', label='Quadratic fit line')
            ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
            plt.legend(loc="upper right")
            if saveplot == 'True':
                fig.savefig('Results\Fd_baseline_linearfit_' + str(i) + '.png')
    
    return M, B, C