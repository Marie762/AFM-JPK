# -*- coding: utf-8 -*-
"""
Created on Tue Apr 2 15:29:30 2024

@author: marie
"""
import matplotlib.pylab as plt
import numpy as np
from extractJPK import QI,force
import plot
import procBasic
import contactPoint
import metadata
import youngsModulus
import seaborn as sns
import pandas as pd

# extract the force spectroscopy data from all the jpk-force files in the directory 'Data'
d, F, t = extractJPK.force()
F_bS = procBasic.baselineSubtraction(F)
spring_constant_list = metadata.SpringConstant()

k = 3


# find contact point
# contactPoint.contactPoint1(F, d, plot='True')

argmin_list = contactPoint.contactPoint1(F_bS, d)

d_hC = procBasic.heightCorrection(d, argmin_list)

perc_top = 95
slice_bottom = argmin_list[k]
slice_top = round((perc_top/100)*len(F_bS[k][0])) 

f = F_bS[k][0]
z = d_hC[k][0]
stiffness = spring_constant_list[k]
deflection = f/stiffness
delta = z - deflection



# find apparant Youngs modulus
popt, pcov = youngsModulus.parabolicIndenter(F_bS[k][0], delta) # [slice_bottom:slice_top]

#y = popt[0]*(x**popt[1])

fig, ax = plt.subplots()
ax.plot(delta, F_bS[k][0], 'deepskyblue')
ax.plot(d_hC[k][0], F_bS[k][0], 'orange')
ax.set(xlabel='tip-sample separation (um)', ylabel='force (nN)', title='Force-distance curve %i' % k)

plt.show()



# extract the QI data from all the jpk-qi-data files in the directory 'Data_QI'
# qmap, Q, XY = QI()

# for k in range(len(qmap)):
#     d = Q[k][0]
#     F = Q[k][1]
#     x_data = XY[k][0]
#     y_data = XY[k][1]
#     # if k == 0:
#     #     contact_point_height = contactPoint.QIcontactPoint1(F,d)
#     #     fig = plot.QIMap(contact_point_height, y_data, x_data, k, save='True')
        
#     if k == 1:
#         contact_point_height = contactPoint.QIcontactPoint2(F,d)
#         fig = plot.QIMap(contact_point_height, y_data, x_data, k, save='True')

# plt.show()

# k = 1
# d = Q[k][0]
# F = Q[k][1]
# for m in range(len(F)):
#     argmin_list = contactPoint.contactPoint2(F[m],d[m],plot='True')
#     plt.show()

# Q: [ [d1, F1, t1], [d2, F2, t2], ...]

# F[0][0][0]: approach data for y = 0, x = 0
# F[0][1][0]: approach data for y = 0, x = 1
# ...

# F[0][0][1]: retract data for y = 0, x = 0
# F[1][0][1]: retract data for y = 1, x = 0
# m = 0
# F = F[m]   
# d = d[m]

# F_bS = procBasic.baselineSubtraction(F)
# argmin_list = []
# perc_bottom = 80
# for i in range(len(F_bS)):
#     slice_bottom = round((perc_bottom/100)*len(F_bS[i][0]))
#     argmin_val = np.argmin(F_bS[i][0][slice_bottom:])
#     argmin_val = argmin_val + slice_bottom
#     argmin_list.append(argmin_val)

    # fig, ax = plt.subplots()
    # ax.plot(d[i][0], F_bS[i][0], 'deepskyblue', label='force-distance curve')
    # ax.plot(d[i][0][slice_bottom:], F_bS[i][0][slice_bottom:], 'red', label='part of curve used in finding minimum')
    # ax.plot(d[i][0][argmin_val], F_bS[i][0][argmin_val], 'ro', label='contact point estimation')
    # ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
    # plt.legend(loc="upper right")





