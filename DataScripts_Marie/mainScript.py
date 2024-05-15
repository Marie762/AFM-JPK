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
from plot import Fd, Fdsubplot, QIMap
from contactPoint import QIcontactPoint3, contactPoint1, contactPoint2, QIcontactPoint1, QIcontactPoint2, contactPoint3
from metadata import Sensitivity, SpringConstant, Speed
from youngsModulus import fitYoungsModulus
from penetrationPoint import substrateContact, penetrationPoint

###### Fd ###############################################################################

# extract the force spectroscopy data from all the jpk-force files in the directory 'Data'
d, F, t = force()
F_bS = baselineSubtraction(F)
d_hC = heightCorrection(d)
contact_point_list = contactPoint3(F_bS, d_hC)
substrate_contact_list = substrateContact(F_bS, d_hC)
d_hZ = heightZeroAtContactPoint(d_hC, contact_point_list)



penetrationPoint(F_bS, d_hC, plot='True')

delta = tipDisplacement(F_bS, d_hC)
#delta_hC = heightCorrection(delta)
delta_hZ = heightZeroAtContactPoint(delta, contact_point_list)



# find apparant Youngs modulus
popt_list, fig = fitYoungsModulus(F_bS, delta_hZ, contact_point_list, substrate_contact_list) # indenter='parabolic', 'conical', or 'pyramidal'

plt.show()

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
