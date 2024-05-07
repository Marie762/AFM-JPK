# -*- coding: utf-8 -*-
"""
Created on Tue Apr 3 2024

@author: marie
"""

import matplotlib.pylab as plt
import numpy as np
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
    contact_point_list = []
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
        
        contact_point_list.append(argmin_val)

        if plot == 'True':
            fig, ax = plt.subplots()
            ax.plot(d[i][0], F_bS[i][0], 'deepskyblue', label='force-distance curve')
            ax.plot(d[i][0], M[i]*(d[i][0]) + B[i], 'orange', label='linear fit line')
            ax.plot(d[i][0][argmin_val], F_bS[i][0][argmin_val], 'ro', label='contact point estimation')
            ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
            plt.legend(loc="upper right")
            if saveplot == 'True':
                fig.savefig('Results\Fd_contact_point_' + str(i) + '.png')
    
    return contact_point_list

def contactPoint2(F, d, plot='False', saveplot='False'):
    F_bS = procBasic.baselineSubtraction(F)
    contact_point_list = []
    perc_bottom = 80
    for i in range(len(F_bS)):
        slice_bottom = round((perc_bottom/100)*len(F_bS[i][0]))
        argmin_val = np.argmin(F_bS[i][0][slice_bottom:])
        argmin_val = argmin_val + slice_bottom
        contact_point_list.append(argmin_val)

        if plot == 'True':
            fig, ax = plt.subplots()
            ax.plot(d[i][0], F_bS[i][0], 'deepskyblue', label='force-distance curve')
            ax.plot(d[i][0][argmin_val], F_bS[i][0][argmin_val], 'ro', label='contact point estimation')
            ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
            plt.legend(loc="upper right")
            if saveplot == 'True':
                fig.savefig('Results\Fd_contact_point_' + str(i) + '.png')
    
    return contact_point_list

def contactPoint3(F, d, plot='False', saveplot='False', perc_bottom=0, perc_top=50):
    M, B = baselineLinearFit(F, d, perc_bottom=perc_bottom, perc_top=perc_top)
    # empty list to store the index of the last intersection point of the F-d curve with the linear fit line 
    contact_point_list = []
    standard_deviation_list = []
    for i in range(len(F)):
        deviation_list = []
        slice_bottom = round((perc_bottom/100)*len(F[i][0]))
        slice_top = round((perc_top/100)*len(F[i][0]))
        # calculate standard deviation
        for j in range(len(F[i][0])):
            f = M[i]*(d[i][0][j]) + B[i] # linear fit line
            deviation_squared = (F[i][0][j] - f)**2 # the difference-squared between the force value and the value of the linear fit line at each point
            deviation_list.append(deviation_squared)
        standard_deviation = np.sqrt(np.sum(deviation_list[slice_bottom:slice_top])/len(F[i][0][slice_bottom:slice_top]))
        standard_deviation_list.append(standard_deviation)
            
        argmax_val = [i for i,el in enumerate(deviation_list) if abs(np.sqrt(el)) > 4*standard_deviation][0]
        argmin_val = [i for i,el in enumerate(deviation_list[:argmax_val]) if abs(el) < 0.01][-1]
        contact_point_list.append(argmin_val)

        if plot == 'True':
            fig, ax = plt.subplots()
            ax.plot(d[i][0], F[i][0], 'deepskyblue', label='force-distance curve')
            ax.plot(d[i][0], M[i]*(d[i][0]) + B[i], 'orange', label='linear fit line')
            ax.plot(d[i][0][argmin_val], F[i][0][argmin_val], 'ro', label='contact point estimation')
            ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
            plt.legend(loc="upper right")
            if saveplot == 'True':
                fig.savefig('Results\Fd_contact_point_' + str(i) + '.png')
    
    return standard_deviation_list, contact_point_list

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

def substrateLinearFit(F, d, perc_bottom=98, plot='False', saveplot='False'):
    # two empty lists to store the gradients 'm' and constants 'b' of the linear fit function for each curve
    M = []
    B = []
    for i in range(len(F)):
        slice_bottom = round((perc_bottom/100)*len(F[i][0]))
        m,b = np.polyfit(d[i][0][slice_bottom:], F[i][0][slice_bottom:], 1) # linear fit of first ..% of dataset
        M.append(m) # store in lists
        B.append(b)
        
        if plot == 'True':
            x = d[i][0]
            lin_fit = m*x + b
            fig, ax = plt.subplots()
            ax.plot(d[i][0], F[i][0], 'deepskyblue', label='force-distance curve')
            ax.plot(x[slice_bottom:], lin_fit[slice_bottom:], 'orange', label='linear fit line')
            ax.plot(d[i][0][slice_bottom:], F[i][0][slice_bottom:], 'red', label='part of curve used in the linear fit')
            ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
            plt.legend(loc="upper left")
            if saveplot == 'True':
                fig.savefig('Results\Fd_substrate_linearfit_' + str(i) + '.png')
    
    return M, B

def substrateContact(F, d, perc_bottom=98, plot='False', saveplot='False'):
    substrate_contact_list = []
    M, B = substrateLinearFit(F,d, perc_bottom=perc_bottom)
    plot_bottom = 96
    for i in range(len(F)):
        difference_list = []
        for j in range(len(F[i][0])):
            f = M[i]*(d[i][0][j]) + B[i] # linear fit line
            difference_squared = (F[i][0][j] - f)**2 # the difference-swuared between the force value and the value of the linear fit line at each point
            difference_list.append(difference_squared)
        
        argmin_val = [i for i,el in enumerate(difference_list) if abs(el) < 500][0]
    
        substrate_contact_list.append(argmin_val)
    
        if plot == 'True':
            x = d[i][0]
            lin_fit = M[i]*x + B[i]
            slice_bottom = round((plot_bottom/100)*len(F[i][0]))
            fig, ax = plt.subplots()
            ax.plot(d[i][0], F[i][0], 'deepskyblue', label='force-distance curve')
            ax.plot(x[slice_bottom:], lin_fit[slice_bottom:], 'orange', label='linear fit line')
            ax.plot(d[i][0][argmin_val], F[i][0][argmin_val], 'ro', label='hard substrate contact point estimation')
            ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
            plt.legend(loc="upper left")
            if saveplot == 'True':
                fig.savefig('Results\Fd_substrate_contact_' + str(i) + '.png')
    
    return substrate_contact_list

def penetrationPoint(F, d, plot='False', saveplot='False'):
    substrate_contact_list = substrateContact(F, d)
    contact_point_list = contactPoint1(F,d)
    P0,P1,P2,P3,P4,P5 = [],[],[],[],[],[]
    for k in range(len(F)):
        slice_top = substrate_contact_list[k]
        slice_bottom = contact_point_list[k]
        p0,p1,p2,p3,p4,p5 = np.polyfit(d[k][0][slice_bottom:slice_top], F[k][0][slice_bottom:slice_top], 5)
        P0.append(p0) # store in lists
        P1.append(p1)
        P2.append(p2)
        P3.append(p3)
        P4.append(p4)
        P5.append(p5)
        
        if plot == 'True':
            x = d[k][0]
            poly_fit = p0*x**5 + p1*x**4 + p2*x**3 + p3*x**2 + p4*x + p5
            fig, ax = plt.subplots()
            ax.plot(d[k][0], F[k][0], 'deepskyblue', label='force-distance curve')
            ax.plot(x[slice_bottom:slice_top], poly_fit[slice_bottom:slice_top], 'orange', label='3rd order polynomial fit')
            ax.plot(d[k][0][slice_bottom:slice_top], F[k][0][slice_bottom:slice_top], 'red', label='part of curve used in the poly-fit')
            ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % k)
            plt.legend(loc="upper left")
            if saveplot == 'True':
                fig.savefig('Results\Fd_3rd_order_polyfit_' + str(k) + '.png')
    
    return