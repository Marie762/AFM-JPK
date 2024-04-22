# -*- coding: utf-8 -*-
"""
Created on Tue Apr 3 2024

@author: marie
"""

import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import pandas as pd
import procBasic

def baselineLinearFit(F, d, perc_bottom=0, perc_top=50, plot='False', saveplot='False'):
    # two empty lists to store the gradients 'm' and constants 'b' of the linear fit function for each curve
    M = []
    B = []
    for i in range(len(F)):
        slice_bottom = round((perc_bottom/100)*len(F[i][0]))
        slice_top = round((perc_top/100)*len(F[i][0])) #compute first ..% of data set and round to nearest integer
        m,b = np.polyfit(d[i][0][slice_bottom:slice_top], F[i][0][slice_bottom:slice_top], 1) # linear fit of first ..% of dataset
        M.append(m) # store in lists
        B.append(b)
        
        if plot == 'True':
            x = d[i][0]
            lin_fit = m*x + b
            fig, ax = plt.subplots()
            ax.plot(d[i][0], F[i][0], 'deepskyblue', label='force-distance curve')
            ax.plot(x, lin_fit, 'orange', label='linear fit line')
            ax.plot(d[i][0][slice_bottom:slice_top], F[i][0][slice_bottom:slice_top], 'red', label='part of curve used in the linear fit')
            ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
            plt.legend(loc="upper right")
            if saveplot == 'True':
                fig.savefig('Results\Fd_baseline_linearfit_' + str(i) + '.png')
    
    return M, B

def contactPoint1(F, d, plot='False', saveplot='False', perc_bottom=0, perc_top=50):
    F_bS = procBasic.baselineSubtraction(F)
    M, B = baselineLinearFit(F_bS, d, perc_bottom=perc_bottom, perc_top=perc_top)
    # empty list to store the index of the last intersection point of the F-d curve with the linear fit line 
    argmin_list = []
    for i in range(len(F_bS)):
        difference_list = []
        for j in range(len(F_bS[i][0])):
            f = M[i]*(d[i][0][j]) + B[i] # linear fit line
            difference_squared = (F_bS[i][0][j] - f)**2 # the difference-swuared between the force value and the value of the linear fit line at each point
            difference_list.append(difference_squared)

        # argmin_val = [i for i,el in enumerate(difference_list) if abs(el) < 0.0001]
        # if len(argmin_val) != 0:
        #     argmin_val = argmin_val[-1]
        # else:
        argmin_val = [i for i,el in enumerate(difference_list) if abs(el) < 0.001]
        if len(argmin_val) != 0:
            argmin_val = argmin_val[-1]
        else:
            argmin_val = [i for i,el in enumerate(difference_list) if abs(el) < 0.01]
            if len(argmin_val) != 0:
                argmin_val = argmin_val[-1]
            else:
                argmin_val = [i for i,el in enumerate(difference_list) if abs(el) < 0.1][-1]
        
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

def contactPoint2(F, d, plot='False', saveplot='False'):
    F_bS = procBasic.baselineSubtraction(F)
    argmin_list = []
    perc_bottom = 80
    for i in range(len(F_bS)):
        slice_bottom = round((perc_bottom/100)*len(F_bS[i][0]))
        argmin_val = np.argmin(F_bS[i][0][slice_bottom:])
        argmin_val = argmin_val + slice_bottom
        argmin_list.append(argmin_val)

        if plot == 'True':
            fig, ax = plt.subplots()
            ax.plot(d[i][0], F_bS[i][0], 'deepskyblue', label='force-distance curve')
            ax.plot(d[i][0][argmin_val], F_bS[i][0][argmin_val], 'ro', label='contact point estimation')
            ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
            plt.legend(loc="upper right")
            if saveplot == 'True':
                fig.savefig('Results\Fd_contact_point_' + str(i) + '.png')
    
    return argmin_list

def QIcontactPoint1(F, d, perc_bottom=0, perc_top=50):
    contact_point_height = []
    for m in range(len(F)):
        contact_point_height_cols = []
        argmin_list = contactPoint1(F[m],d[m],perc_bottom=perc_bottom, perc_top=perc_top)
        
        for n in range(len(F[m])):
            contact_point_height_cols.append(d[m][n][0][argmin_list[n]])
        
        contact_point_height.append(contact_point_height_cols)
    
    return contact_point_height

def QIcontactPoint2(F,d):
    contact_point_height = []
    for m in range(len(F)):
        contact_point_height_cols = []
        argmin_list = contactPoint2(F[m],d[m])
        
        for n in range(len(F[m])):
            contact_point_height_cols.append(d[m][n][0][argmin_list[n]])
        
        contact_point_height.append(contact_point_height_cols)
    
    return contact_point_height