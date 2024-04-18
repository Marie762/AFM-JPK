# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:57:50 2024

@author: marie
"""

import matplotlib.pylab as plt
from matplotlib.ticker import ScalarFormatter
from scipy.signal import savgol_filter
import numpy as np
import afmformats

# deriving and plotting tip-sample separation vs. force:

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




# plotting derivative 

#dFdd = []

# for i in range(len(F[k])-1):
#  	#dF = (F[k][i+1] - F[k][i]) / (d[k][i+1] - d[k][i])
#  	dF = F[k][i+1] - F[k][i]
#  	dd = d[k][i+1] - d[k][i]
#  	dFdd.append(dF/dd)
 	
# dFdd.append(dFdd[len(F[k])-2])

# dFdt = []

# for i in range(len(F[k])-1):
# 	dF = smoothed_data[i+1] - smoothed_data[i]
# 	dt = t[k][i+1] - t[k][i]
# 	dFdt.append(dF/dt)

# dFdt.append(dFdt[len(F[k])-2])


# find derivative  for moving avg smoothed function

# k = 49

# dFdd = []
# # for i in range(len(F[k])-1):
# # 	#dF = (F[k][i+1] - F[k][i]) / (d[k][i+1] - d[k][i])
# # 	dF = F[k][i+1] - F[k][i]
# # 	dd = d[k][i+1] - d[k][i]
# # 	dFdd.append(dF/dd)
# # 	
# # dFdd.append(dFdd[len(F[k])-2])
# naam_van_mijn_vrouw = "Marie"
# print(f"Beeldschone code hoor {naam_van_mijn_vrouw} \n -Gert")

# # fig, ax = plt.subplots()
# # ax.plot(d[k], F_corr, 'r')
# # ax.plot(d[k], dFdd, 'g')

# # ax.set(xlabel='height measured (um)', ylabel='force (nN)', title='Force-distance curve %i' % k)


# for i in range(len(F[k])-1):
#  	#dF = (F[k][i+1] - F[k][i]) / (d[k][i+1] - d[k][i])
#  	dF = moving_avg.loc[i+1].iat[0] - moving_avg.loc[i].iat[0] # .loc[i].iat[0]
#  	dd = d[k][i+1] - d[k][i]
#  	dFdd.append(dF/dd)
 	
# dFdd.append(dFdd[len(moving_avg)-2])
