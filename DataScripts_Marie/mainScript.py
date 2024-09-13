# -*- coding: utf-8 -*-
"""
Created on Tue Apr 2 15:29:30 2024

@author: marie
"""
import os
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import pandas as pd
import pickle
from extractJPK import QI, force
from procBasic import baselineSubtraction, heightCorrection, heightCorrection2, heightZeroAtContactPoint, sensitivityCorrection, tipDisplacement, smoothingSG
from plot import Fd, FdGrid_Emodulus, FdGrid_ForceDrop, FdGrid_Height, FdGrid_Indentation, FdGrid_Peaks, FdGrid_PenetrationForce, Fdsubplot, QIMap
from contactPoint import QIcontactPoint3, contactPoint1, contactPoint2, QIcontactPoint1, QIcontactPoint2, contactPoint_RoV, contactPoint_ruptures, contactPoint3, contactPoint_evaluation, contactPoint_derivative
from metadata import Sensitivity, SpringConstant, Position, Speed, Setpoint
from createGrid import grid10x10, grid10x10_specialcase, grid15x15, grid15x15_specialcase, grid20x20, grid25x25
from youngsModulus import fitYoungsModulus
from penetrationPoint import forceDrop, indentationDepth, substrateContact, findPeaks, substrateContact2


###### Fd ###############################################################################
k = 1
date = '2024.07.18_'
# extract the force spectroscopy data from all the jpk-force files in the directory 'Data'
d, F, t = force(load_from_pickle = True, grid=k, date=date)
new_sensitivity = 80*10**(-9)
F_corr = sensitivityCorrection(F, new_sensitivity)

F_bS = baselineSubtraction(F_corr)
d_hC = heightCorrection2(d)
delta = tipDisplacement(F_bS, d_hC, plot=True, save=True)
delta_hC = heightCorrection2(delta)

# RoV_list = contactPoint_RoV(F_bS, delta_hC, plot=True)
# derivative_list = contactPoint_derivative(F_bS, delta_hC, plot=True)

data_path = r'StoredValues/' 
load_from_pickle=True
# load_from_pickle=True
if not load_from_pickle: 
    contact_point_fit = contactPoint_ruptures(F_bS, d_hC)
    with open(data_path + '/contactPoint_'+ date + 'grid_' + str(k) + '.pkl', "wb") as output_file:
        pickle.dump(contact_point_fit, output_file)
else:
    with open(data_path + '/contactPoint_'+ date + 'grid_' + str(k) + '.pkl', "rb") as output_file:
        contact_point_fit = pickle.load(output_file)

# # # contact_point_list = contactPoint3(F_bS, d_hC, perc_top=50,multiple=10, multiple1=3, multiple2=2)


# find index in array with closest value to the change point in the fit
contact_point_list = []
for n in range(len(d_hC)):
    array = np.asarray(d_hC[n][0])
    value = contact_point_fit[n]
    diff_abs = np.abs(array - value)
    idx = (diff_abs).argmin()
    contact_point_list.append(idx) 

number_of_points_correct, percentage_of_points_correct, total_error = contactPoint_evaluation(F_bS, d_hC, contact_point_list)


# fig = Fd(F_bS, d_hC, real_contact_point_list, real_contact_point_fit, save=True)

# # find height data for height grid plot
# contact_point_height = []
# for n in range(len(d_hC)):
#     contact_point_height.append(d_hC[n][0][contact_point_list[n]])


# substrate_contact_list = substrateContact(F_bS, delta, contact_point_list)

# find penetration points
# first_peak_list, number_of_peaks_list, all_peaks_list, properties_list = findPeaks(F_bS, d_hC, contact_point_list) # , plot=True, save=True

# # # # find penetration force & force drop
# penetration_force_list, first_penetration_force_list, right_bases_list, force_drop_list, first_force_drop_list = forceDrop(F_bS, d_hC, contact_point_list, first_peak_list, properties_list, plot=True)

# # # # find indentation depth
# indentation_depth_arr = indentationDepth(F_bS, d_hC, contact_point_list, first_peak_list)

# # find apparant Youngs modulus
# delta_hZ = heightZeroAtContactPoint(delta, contact_point_list)
# E_list, fig = fitYoungsModulus(F_bS, delta_hZ, contact_point_list, substrate_contact_list, first_peak_list, plot=True, save=True) # indenter='parabolic', 'conical', or 'pyramidal'


# # # # create grid plot
# grid_data, x_and_y_data = grid15x15(contact_point_height) # x_and_y_data        x_data, y_data
# fig = FdGrid_Height(grid_data, x_and_y_data, x_and_y_data, k, save='True', name='Height (um) ') # x_and_y_data, x_and_y_data

# # grid_data, x_and_y_data = grid15x15(number_of_peaks_list) 
# fig = FdGrid_Peaks(grid_data, x_and_y_data, x_and_y_data, k, save='True', name='Number of peaks ')

# grid_data, x_and_y_data = grid15x15(first_penetration_force_list) 
# fig = FdGrid_PenetrationForce(grid_data, x_and_y_data, x_and_y_data, k, save='True', name='Penetration force of 1st peak ')

# grid_data, x_and_y_data = grid15x15(first_force_drop_list) 
# fig = FdGrid_ForceDrop(grid_data, x_and_y_data, x_and_y_data, k, save='True', name='Force drop of 1st peak ') # set NaN values to zero

# grid_data, x_and_y_data = grid15x15(indentation_depth_arr) 
# fig = FdGrid_Indentation(grid_data, x_and_y_data, x_and_y_data, k, save='True', name='Indentation depth (um) ') 

# grid_data, x_and_y_data = grid15x15(E_list)
# fig = FdGrid_Emodulus(grid_data, x_and_y_data, x_and_y_data, k, save='True', name='Youngs modulus (kPa) ')

# plt.show() 

# # # convert metadata to csv file:
# # x_position_list, y_position_list = Position()
# prominences_list = []
# widths_list = []
# for n in range(len(F_bS)):
#     prominences_list.append(properties_list[n]["prominences"])
#     widths_list.append(properties_list[n]["widths"])
# index = range(len(F_bS))
# # data = {'X Position (um/s)': x_position_list, 'Y Position (um/s)': y_position_list}
# data = {'Index': index, 'Peak height': peak_height_list, 'Peaks': all_peaks_list, 'Right bases': right_bases_list, 'Force drop' : force_drop_list} # 'Widths': widths_list, 'Prominences': prominences_list,
# # Create a DataFrame 
# data_frame = pd.DataFrame(data) 

# # Save DataFrame to text file 
# data_frame.to_csv('Results_metadata\metadata_output.csv', index=False, encoding='utf-8')





# # dP_real_roots, dP2_real_roots = penetrationPoint(F_bS, d_hC, plot='True') # plot='True'

# # dP_n_roots, dP2_n_roots = [],[]
# # for k in range(len(dP_real_roots)):
# #     dP_number_of_roots = len(dP_real_roots[k])
# #     dP2_number_of_roots = len(dP2_real_roots[k])
    
# #     dP_n_roots.append(dP_number_of_roots)
# #     dP2_n_roots.append(dP2_number_of_roots)
    
# #     print(k, ':', 'dP=', dP_n_roots[k], 'dP2=', dP2_n_roots[k])



# ################################################################################

# ############### QI ###########################################################
 
# # # extract the QI data from all the jpk-qi-data files in the directory 'Data_QI'
# # qmap, Q, XY = QI(load_from_pickle=True)

# # for k in range(len(qmap)):
# #     d = Q[k][0]
# #     F = Q[k][1]
# #     x_data = XY[k][0]
# #     y_data = XY[k][1]
# #     if k == 0:
# #         contact_point_height = QIcontactPoint3(F,d, perc_top=80)
# #         fig = QIMap(contact_point_height, y_data, x_data, k, save='True')
        
# #     if k == 1:
# #         contact_point_height = QIcontactPoint3(F,d, perc_top=80)
# #         fig = QIMap(contact_point_height, y_data, x_data, k, save='True')
        
# #     if k == 2:
# #         contact_point_height = QIcontactPoint3(F,d)
# #         fig = QIMap(contact_point_height, y_data, x_data, k, save='True')

# # plt.show()


# #####################################################################

# # k = 2
# # d = Q[k][0]
# # F = Q[k][1]
# # x_data = XY[k][0]
# # y_data = XY[k][1]
# # # for m in range(len(F)):
# # #     F_bS = baselineSubtraction(F[m][:10])
# # #     d_hC = heightCorrection(d[m][:10])
# # #     argmin_list3 = contactPoint3(F_bS, d_hC, plot='True', perc_bottom=0, perc_top=60, multiple=10, multiple1=4) # , multiple=25, multiple1=20, multiple2=10
# # #     argmin_list2 = contactPoint2(F_bS, d_hC, plot='True')
# # #     plt.show()


# # contact_point_height2 = QIcontactPoint2(F,d)
# # fig = QIMap(contact_point_height2, y_data, x_data, k, name = 'contactPoint2')
# # contact_point_height3 = QIcontactPoint3(F,d, perc_bottom=0, perc_top=50)
# # fig = QIMap(contact_point_height3, y_data, x_data, k, name = 'contactPoint3')
# # plt.show()
    
# ##########################################################################

# # Q data storage structure:
# # Q: [ [d1, F1, t1], [d2, F2, t2], ...]

# # F1: [F approach, (F intermediate,) F retract]

# # F[0][0][0]: approach data for y = 0, x = 0
# # F[0][1][0]: approach data for y = 0, x = 1
# # ...

# # F[0][0][1]: retract data for y = 0, x = 0
# # F[1][0][1]: retract data for y = 1, x = 0
# # ...
