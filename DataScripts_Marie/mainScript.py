# -*- coding: utf-8 -*-
"""
Created on Tue Apr 2 15:29:30 2024

@author: marie
"""
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import pandas as pd
from extractJPK import QI, force
from procBasic import baselineSubtraction, heightCorrection, heightZeroAtContactPoint, tipDisplacement, smoothingSG
from plot import Fd, FdGrid, Fdsubplot, QIMap
from contactPoint import QIcontactPoint3, contactPoint1, contactPoint2, QIcontactPoint1, QIcontactPoint2, contactPoint3
from metadata import Sensitivity, SpringConstant, Position, Speed, Setpoint
from youngsModulus import fitYoungsModulus
from penetrationPoint import substrateContact, penetrationPoint

###### Fd ###############################################################################

# extract the force spectroscopy data from all the jpk-force files in the directory 'Data'
d, F, t = force()

G0 = [d[0:99],F[0:99]]
G1 = [d[100:199],F[100:199]]
G2 = [d[200:299],F[200:299]]
G3 = [d[300:399],F[300:399]]
G4 = [d[400:499],F[400:499]]
G5 = [d[500:599],F[500:599]]
G6 = [d[600:699],F[600:699]]
G7 = [d[700:719],F[700:719]]
G8 = [d[720:1119],F[720:1119]]

F_bS0 = baselineSubtraction(G0[1])
d_hC0 = heightCorrection(G0[0])
contact_point_list = contactPoint1(F_bS0, d_hC0, perc_top=70)

x_position_list, y_position_list = Position()

contact_point_height = [] 
for n in range(0,10):
    lower = n*10
    higher = lower + 10
    for m in range(len(d_hC0)):
        contact_point_height_cols = []
        if lower <= m < higher:
            contact_point_height_cols.append(d_hC0[m][contact_point_list[m]])

    contact_point_height.append(contact_point_height_cols)

x_position = x_position_list[0:99]
y_position = y_position_list[0:99]

k = 0

fig = FdGrid(contact_point_height, x_position, y_position, k)

# F_bS = baselineSubtraction(F)
# d_hC = heightCorrection(d)
# contact_point_list = contactPoint1(F_bS, d_hC, perc_top=70, plot='True', saveplot='True')
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
