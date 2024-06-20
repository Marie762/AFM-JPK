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
from extractJPK import QI, force
from procBasic import baselineSubtraction, heightCorrection, heightCorrection2, heightZeroAtContactPoint, tipDisplacement, smoothingSG
from plot import Fd, FdGrid, Fdsubplot, QIMap
from contactPoint import QIcontactPoint3, contactPoint1, contactPoint2, QIcontactPoint1, QIcontactPoint2, contactPoint3
from metadata import Sensitivity, SpringConstant, Position, Speed, Setpoint
from youngsModulus import fitYoungsModulus
from penetrationPoint import substrateContact, penetrationPoint

###### Fd ###############################################################################

# extract the force spectroscopy data from all the jpk-force files in the directory 'Data'
d, F, t = force()

F_bS = baselineSubtraction(F)
d_hC = heightCorrection2(d)
contact_point_list = contactPoint3(F_bS, d_hC,multiple=30, multiple1=20, plot='True', saveplot='True') #, plot='True', saveplot='True'

contact_point_height =[]
for n in range(len(d_hC)):
    contact_point_height.append(d_hC[n][0][contact_point_list[n]])

        
# create data array from contact point list
# 10x10
c0 = contact_point_height[:10]
c1 = contact_point_height[10:20]
r1 = c1[::-1]
c2 = contact_point_height[20:30]
c3 = contact_point_height[30:40]
r3 = c3[::-1]
c4 = contact_point_height[40:50]
c5 = contact_point_height[50:60]
r5 = c5[::-1]
c6 = contact_point_height[60:70]
c7 = contact_point_height[70:80]
r7 = c7[::-1]
c8 = contact_point_height[80:90]
c9 = contact_point_height[90:]
r9 = c9[::-1]
data = [c0,r1,c2,r3,c4,r5,c6,r7,c8,r9]

# # 25x25
# c0 = contact_point_height[:25]
# c1 = contact_point_height[25:50]
# r1 = c1[::-1]
# c2 = contact_point_height[50:75]
# c3 = contact_point_height[75:100]
# r3 = c3[::-1]
# c4 = contact_point_height[100:125]
# c5 = contact_point_height[125:150]
# r5 = c5[::-1]
# c6 = contact_point_height[150:175]
# c7 = contact_point_height[175:200]
# r7 = c7[::-1]
# c8 = contact_point_height[200:225]
# c9 = contact_point_height[225:250]
# r9 = c9[::-1]
# c10 = contact_point_height[250:275]
# c11 = contact_point_height[275:300]
# r11 = c11[::-1]
# c12 = contact_point_height[300:325]
# c13 = contact_point_height[325:350]
# r13 = c13[::-1]
# c14 = contact_point_height[350:375]
# c15 = contact_point_height[375:400]
# r15 = c15[::-1]
# c16 = contact_point_height[400:425]
# c17 = contact_point_height[425:450]
# r17 = c17[::-1]
# c18 = contact_point_height[450:475]
# c19 = contact_point_height[475:500]
# r19 = c19[::-1]
# c20 = contact_point_height[500:525]
# c21 = contact_point_height[525:550]
# r21 = c21[::-1]
# c22 = contact_point_height[550:575]
# c23 = contact_point_height[575:600]
# r23 = c23[::-1]
# c24 = contact_point_height[600:]

# data = [c0,r1,c2,r3,c4,r5,c6,r7,c8,r9,c10,r11,c12,r13,c14,r15,c16,r17,c18,r19,c20,r21,c22,r23,c24]

# metadata
x_position_list, y_position_list = Position()
x_and_y_data = x_position_list[0:10] # 10x10
# x_and_y_data = x_position_list[0:25] # 25x25

# spring_constant_list = SpringConstant()

# create grid plot
k = 5 
fig = FdGrid(data, x_and_y_data, x_and_y_data, k, save='True')
plt.show() 

# substrate_contact_list = substrateContact(F_bS, d_hC)  
# d_hZ = heightZeroAtContactPoint(d_hC, contact_point_list)





# # convert metadata to csv file:
# x_position_list, y_position_list = Position()

# data = {'X Position (um/s)': x_position_list, 'Y Position (um/s)': y_position_list}
# # Create a DataFrame 
# data_frame = pd.DataFrame(data) 

# # Save DataFrame to CSV file 
# data_frame.to_csv('Results_metadata\metadata_output.csv', index=False, encoding='utf-8')





# dP_real_roots, dP2_real_roots = penetrationPoint(F_bS, d_hC, plot='True') # plot='True'

# dP_n_roots, dP2_n_roots = [],[]
# for k in range(len(dP_real_roots)):
#     dP_number_of_roots = len(dP_real_roots[k])
#     dP2_number_of_roots = len(dP2_real_roots[k])
    
#     dP_n_roots.append(dP_number_of_roots)
#     dP2_n_roots.append(dP2_number_of_roots)
    
#     print(k, ':', 'dP=', dP_n_roots[k], 'dP2=', dP2_n_roots[k])


# delta = tipDisplacement(F_bS, d_hC)
# #delta_hC = heightCorrection(delta)
# delta_hZ = heightZeroAtContactPoint(delta, contact_point_list)

# # find apparant Youngs modulus
# popt_list, fig = fitYoungsModulus(F_bS, delta_hZ, contact_point_list, substrate_contact_list) # indenter='parabolic', 'conical', or 'pyramidal'



################################################################################

############### QI ###########################################################
 
# # extract the QI data from all the jpk-qi-data files in the directory 'Data_QI'
# qmap, Q, XY = QI(load_from_pickle=True)

# for k in range(len(qmap)):
#     d = Q[k][0]
#     F = Q[k][1]
#     x_data = XY[k][0]
#     y_data = XY[k][1]
#     if k == 0:
#         contact_point_height = QIcontactPoint3(F,d, perc_top=80)
#         fig = QIMap(contact_point_height, y_data, x_data, k, save='True')
        
#     if k == 1:
#         contact_point_height = QIcontactPoint3(F,d, perc_top=80)
#         fig = QIMap(contact_point_height, y_data, x_data, k, save='True')
        
#     if k == 2:
#         contact_point_height = QIcontactPoint3(F,d)
#         fig = QIMap(contact_point_height, y_data, x_data, k, save='True')

# plt.show()


#####################################################################

# k = 2
# d = Q[k][0]
# F = Q[k][1]
# x_data = XY[k][0]
# y_data = XY[k][1]
# # for m in range(len(F)):
# #     F_bS = baselineSubtraction(F[m][:10])
# #     d_hC = heightCorrection(d[m][:10])
# #     argmin_list3 = contactPoint3(F_bS, d_hC, plot='True', perc_bottom=0, perc_top=60, multiple=10, multiple1=4) # , multiple=25, multiple1=20, multiple2=10
# #     argmin_list2 = contactPoint2(F_bS, d_hC, plot='True')
# #     plt.show()


# contact_point_height2 = QIcontactPoint2(F,d)
# fig = QIMap(contact_point_height2, y_data, x_data, k, name = 'contactPoint2')
# contact_point_height3 = QIcontactPoint3(F,d, perc_bottom=0, perc_top=50)
# fig = QIMap(contact_point_height3, y_data, x_data, k, name = 'contactPoint3')
# plt.show()
    
##########################################################################

# Q data storage structure:
# Q: [ [d1, F1, t1], [d2, F2, t2], ...]

# F1: [F approach, (F intermediate,) F retract]

# F[0][0][0]: approach data for y = 0, x = 0
# F[0][1][0]: approach data for y = 0, x = 1
# ...

# F[0][0][1]: retract data for y = 0, x = 0
# F[1][0][1]: retract data for y = 1, x = 0
# ...
