# -*- coding: utf-8 -*-
"""
Created on Tue Apr 2 15:29:30 2024

@author: marie
"""
import os
import matplotlib.pylab as plt
import time
import numpy as np
import seaborn as sns
import pandas as pd
import pickle
import sys
from matplotlib import cm
from extractJPK import QI, force
from procBasic import baselineSubtraction, baselineSubtraction2, heightCorrection, heightCorrection2, heightZeroAtContactPoint, sensitivityCorrection, tipDisplacement, smoothingSG
from plot import Fd, Fd1, FdGrid_Emodulus, FdGrid_ForceDrop, FdGrid_Height, FdGrid_Indentation, FdGrid_Peaks, FdGrid_PenetrationForce, Fdsubplot, QIMap
from contactPoint import QIcontactPoint3, contactPoint1, contactPoint2, QIcontactPoint1, QIcontactPoint2, contactPoint_RoV, contactPoint_piecewise_regression, contactPoint_ruptures, contactPoint3, contactPoint_evaluation, contactPoint_derivative
from metadata import Sensitivity, SpringConstant, Position, Speed, Setpoint
from createGrid import grid10x10, grid10x10_specialcase, grid15x15, grid15x15_specialcase, grid20x20, grid25x25,grid10x10_specialcase2
from youngsModulus import fitYoungsModulus
from penetrationPoint import forceDrop, indentationDepth, substrateContact, findPeaks, substrateContact2

# Start the timer
start_time = time.time()

###### Fd ###############################################################################
k = 1 # grid number
date = '2024.07.18_' # experiment date

# Extract the force spectroscopy data from all the jpk-force files in the directory 'Data'
print('Loading force spectroscopy data')
d, F, t = force(load_from_pickle = True, grid=k, date=date)
print(f'Done {time.time() - start_time:.2f} seconds')


spring_constant_list = SpringConstant()
spring_constant = spring_constant_list[0]*10
print('spring constant', spring_constant)
tip_diameter = 2
velocity_list = Speed()
insertion_velocity = velocity_list[0][0]
print('insertion velocity', insertion_velocity)

# # # # Make a sensitivity correction if necessary
# print('Sensitivity correction')
# new_sensitivity = 100*10**(-9)
# F = sensitivityCorrection(F, new_sensitivity)
# print(f'Done {time.time() - start_time:.2f} seconds')

# # # # Height and baseline corrections
print('Height and baseline corrections')
F_bS = baselineSubtraction2(F)
d_hC = heightCorrection2(d)
print(f'Done {time.time() - start_time:.2f} seconds')

# # # # Deflection of the cantilever correction + another height correctiom
print('Finding delta')
delta = tipDisplacement(F_bS, d_hC)
delta_hC = heightCorrection2(delta)
print(f'Done {time.time() - start_time:.2f} seconds')

# # # # Check sensitivity by plotting normal F-Delta curves:
# print('Checking if sensitivity is correct by plotting F-Delta curves')
# fig = Fd(F_bS, delta_hC, save=True) 
# print(f'Done {time.time() - start_time:.2f} seconds')

# # # # Find and store contact point list
print('Finding contact points')
data_path = r'StoredValues/' 
load_from_pickle=True
if not load_from_pickle:  
    contact_point_list = contactPoint_derivative(F_bS, delta_hC, N=600, threshold1=2, threshold2=0.05, plot=True)
    with open(data_path + '/contactPoint_derivative_'+ date + 'grid_' + str(k) + '.pkl', "wb") as output_file:
        pickle.dump(contact_point_list, output_file)
else:
    with open(data_path + '/contactPoint_derivative_'+ date + 'grid_' + str(k) + '.pkl', "rb") as output_file:
        contact_point_list = pickle.load(output_file)
print(f'Done {time.time() - start_time:.2f} seconds')

# # # Check contact points
# print('Checking contact points')
# Fd1(F_bS, delta_hC, contact_point_list, save=True)
# print(f'Done {time.time() - start_time:.2f} seconds')

# # # # find height data for height grid plot
print('Finding the height data of the contact point')
contact_point_height = []
for n in range(len(d_hC)):
    if contact_point_list[n]:
        contact_point_height.append(d_hC[n][0][contact_point_list[n]])
    else:
        contact_point_height.append(0)
print(f'Done {time.time() - start_time:.2f} seconds')

# # # # # find insertion points
# print('Finding the insertion points')
# first_peak_list, number_of_peaks_list, all_peaks_list, peak_heights_list, right_bases_list = findPeaks(F_bS, delta_hC, contact_point_list, prominence=0.000000001, plot=True, save=True) # , plot=True, save=True
# print(f'Done {time.time() - start_time:.2f} seconds')

# # # # # find insertion force and force drop
# print('Finding the force drop and insertion force')
# penetration_force_list, first_penetration_force_list, right_bases_list, force_drop_list, first_force_drop_list = forceDrop(F_bS, delta_hC, first_peak_list, peak_heights_list, right_bases_list, plot=True)
# print(f'Done {time.time() - start_time:.2f} seconds')

# # # # # find indentation depth
# print('Finding the indentation depth')
# indentation_depth_arr = indentationDepth(F_bS, delta_hC, contact_point_list, first_peak_list)
# print(f'Done {time.time() - start_time:.2f} seconds')

# # # # # find contact point with hard substrate
# print('Finding the contact point with the hard substrate')
# substrate_contact_list = substrateContact(F_bS, delta_hC, contact_point_list)
# print(f'Done {time.time() - start_time:.2f} seconds')

# # # # # find elastic modulus
# print('Finding the elastic modulus')
# delta_hZ = heightZeroAtContactPoint(delta_hC, contact_point_list)
# E_list, fig = fitYoungsModulus(F_bS, delta_hZ, contact_point_list, substrate_contact_list, first_peak_list, plot=True, save=True) # indenter='parabolic', 'conical', or 'pyramidal'
# print(f'Done {time.time() - start_time:.2f} seconds')


# # # # # storing complete data in dataframe and pickle
# print('Storing complete data in dataframe')
# index_list = range(0, len(F))
# data = {'index': index_list, 
#         'contact point': contact_point_list, 
#         'substrate contact': substrate_contact_list,
#         'height': contact_point_height, 
#         'first peak': first_peak_list, 
#         'number of peaks': number_of_peaks_list, 
#         'insertion force': first_penetration_force_list, 
#         'force drop': first_force_drop_list, 
#         'indentation depth': indentation_depth_arr,
#         'E modulus': E_list,
#         'spring constant': spring_constant,
#         'tip diameter': tip_diameter,
#         'insertion velocity': insertion_velocity} 
# data_frame = pd.DataFrame(data) 

# # Save DataFrame to csv file 
# data_frame.to_csv('Results\Complete_data_' + date + 'grid_' + str(k) + '.csv', index=False, encoding='utf-8')

# with open(data_path + '/Complete_data_'+ date + 'grid_' + str(k) + '.pkl', "wb") as output_file:
#     pickle.dump(data_frame, output_file)
# print(f'Done {time.time() - start_time:.2f} seconds')


# # # # create grid plots
# print('Creating grid plot of height data')
# grid_data, x_and_y_data = grid15x15(contact_point_height) # x_and_y_data        x_data, y_data
# fig = FdGrid_Height(grid_data,x_and_y_data, x_and_y_data, k, save='True', name='Height (um) ') # x_and_y_data, x_and_y_data
# print(f'Done {time.time() - start_time:.2f} seconds')


# print('Creating grid plot of number of insertion points data')
# grid_data, x_and_y_data = grid15x15(number_of_peaks_list) 
# fig = FdGrid_Peaks(grid_data,x_and_y_data, x_and_y_data, k, save='True', name='Number of peaks ')
# print(f'Done {time.time() - start_time:.2f} seconds')

# print('Creating grid plot of penetration force data')
# grid_data, x_and_y_data = grid15x15(first_penetration_force_list) 
# fig = FdGrid_PenetrationForce(grid_data, x_and_y_data, x_and_y_data, k, save='True', name='Penetration force of 1st peak ')
# print(f'Done {time.time() - start_time:.2f} seconds')

# print('Creating grid plot of force drop data')
# grid_data,x_and_y_data = grid15x15(first_force_drop_list) 
# fig = FdGrid_ForceDrop(grid_data, x_and_y_data, x_and_y_data, k, save='True', name='Force drop of 1st peak ') # set NaN values to zero
# print(f'Done {time.time() - start_time:.2f} seconds')

# print('Creating grid plot of indentation depth data')
# grid_data, x_and_y_data = grid15x15(indentation_depth_arr) 
# fig = FdGrid_Indentation(grid_data, x_and_y_data, x_and_y_data, k, save='True', name='Indentation depth (um) ') 
# print(f'Done {time.time() - start_time:.2f} seconds')

# print('Creating grid plot of E modulus data')
# grid_data, x_and_y_data = grid15x15(E_list)
# fig = FdGrid_Emodulus(grid_data, x_and_y_data, x_and_y_data, k, save='True', name='Youngs modulus (kPa) ')
# print(f'Done {time.time() - start_time:.2f} seconds')

x_position_list, y_position_list = Position()

height_data, _ = grid15x15(contact_point_height)
x_data, _ = grid15x15(x_position_list)
y_data, _ = grid15x15(y_position_list)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surface = ax.plot_surface(x_data, y_data, np.array(height_data), rcount=100, ccount=100, linewidth=0, antialiased=False, cmap=cm.coolwarm, vmin=0, vmax=5)
ax.set_zlim3d(bottom=0, top=5)
plt.locator_params(axis='z', nbins=1) 
ax.set_aspect('equal', adjustable='box') # adjustable='box'
fig.colorbar(surface, shrink=0.7, aspect=15, location='right', label='Cell height (um)')
ax.set_xlabel(u'x (\u03bcm)', fontsize=15)
ax.set_ylabel(u'y (\u03bcm)', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
# ax.set_zlabel('Cell height (um)', fontsize=15)
fig.savefig('Results\AHeight3D_' + str(k) + '.png', dpi=500)
fig.savefig('Results\AHeight3D_' + str(k) + '.pdf', format='pdf')


# ################################################################################

# ############### Convert to CSV file ############################################

# # # convert metadata to csv file:
# # # x_position_list, y_position_list = Position()
# prominences_list = []
# widths_list = []
# for n in range(len(F_bS)):
#     prominences_list.append(properties_list[n]["prominences"])
#     widths_list.append(properties_list[n]["widths"])
# index = range(len(F_bS))
# # data = {'X Position (um/s)': x_position_list, 'Y Position (um/s)': y_position_list}
# data = {'Index': index, 'Peaks': all_peaks_list, 'Prominences': prominences_list,'Peak widths': widths_list} # 'Widths': widths_list, 'Prominences': prominences_list,
# # Create a DataFrame 
# data_frame = pd.DataFrame(data) 

# # Save DataFrame to text file 
# data_frame.to_csv('Results_metadata\metadata_output.csv', index=False, encoding='utf-8')


# ################################################################################

# ############### Piecewise Regression notes #####################################

# load_from_pickle=False
# if not load_from_pickle: 
#     contact_point_fit = contactPoint_piecewise_regression(F_bS, d_hC)
#     with open(data_path + '/contactPoint_'+ date + 'grid_' + str(k) + '.pkl', "wb") as output_file:
#         pickle.dump(contact_point_fit, output_file)
# else:
#     with open(data_path + '/contactPoint_'+ date + 'grid_' + str(k) + '.pkl', "rb") as output_file:
#         contact_point_fit = pickle.load(output_file)

# # find index in array with closest value to the change point in the fit
# contact_point_list = []
# for n in range(len(d_hC)):
#     array = np.asarray(d_hC[n][0])
#     value = contact_point_fit[n]
#     diff_abs = np.abs(array - value)
#     idx = (diff_abs).argmin()
#     contact_point_list.append(idx) 

# contact_point_list = contactPoint3(F_bS, d_hC, perc_top=50,multiple=10, multiple1=3, multiple2=2)



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
