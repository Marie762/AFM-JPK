# -*- coding: utf-8 -*-
"""
Created on Tue Apr 2 15:29:30 2024

@author: marie
"""
import matplotlib.pylab as plt
import numpy as np
import extractJPK
import plot
import procBasic
import metadata
from scipy.signal import argrelmin

# extract the QI data from all the jpk-qi-data files in the directory 'Data_QI'
#qmap = extractJPK.QI()

# extract the force spectroscopy data from all the jpk-force files in the directory 'Data'
d, F, t = extractJPK.force()
#spring_constant_list = metadata.SpringConstant()
F_bS = procBasic.baselineSubtraction(F)

M, B = procBasic.baselineLinearFit(F_bS, d)
argmin_list = procBasic.contactPoint(F,d)
print(argmin_list)


# fig, ax = plt.subplots()
# ax.plot(z, f, 'deepskyblue')
# ax.plot(delta, f, 'orange')
# ax.set(xlabel='z (blue) and delta (orange)', ylabel='force', title='Force-distance curve %i' % k)

# for i in range(len(F_bS)):
#     f = F_bS[i][0]
#     z = d[i][0]
#     k = spring_constant_list[i]
#     deflection = f/k
#     delta = z - deflection
        
    # fig, ax = plt.subplots()
    # ax.plot(t[i][0], f, 'deepskyblue')
    # ax.plot(-d[i][0], f, 'orange')
    # ax.set(xlabel='z (blue) and delta (orange)', ylabel='force', title='Force-distance curve %i' % k)
    
    # fig, ax = plt.subplots()
    # ax.plot(t[i][0], z, 'deepskyblue')
    # ax.plot(t[i][0], deflection, 'red')
    # ax.plot(t[i][0], delta, 'orange')
    # ax.set(xlabel='time', ylabel='z (blue) and delta (orange)', title='time-distance curve %i' % i)
    #fig.savefig('Results\dtt_' + str(i) + '.png')

#plt.show()

