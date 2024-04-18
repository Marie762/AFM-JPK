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
import seaborn as sns
import pandas as pd

# extract the force spectroscopy data from all the jpk-force files in the directory 'Data'
#d, F, t = extractJPK.force()


# extract the QI data from all the jpk-qi-data files in the directory 'Data_QI'
qmap = extractJPK.QI()

k = 1
get_qmap = qmap[k].get_qmap("data: height base point")

x_data = np.around(get_qmap[0], decimals=3)
y_data = np.around(get_qmap[1], decimals=3)

# scale conversion constants
ysc = 1e9 # nN
dsc = 1e6 # microns

# create three empty lists to store the height (d), force (F), and time (t) values   
d = []
F = []
t = []

d_cols = []
F_cols = []
t_cols = []

for i in range(len(qmap[k].group)):
    d_local = []
    F_local = []
    t_local = []
    
    # qmap[1].group[i]: force-distance data i 
    
    d_local.append(qmap[k].group[i].appr["height (measured)"]*dsc)
    F_local.append(qmap[k].group[i].appr["force"]*ysc)
    t_local.append(qmap[k].group[i].appr["time"])
    
    d_local.append(qmap[k].group[i].retr["height (measured)"]*dsc)
    F_local.append(qmap[k].group[i].retr["force"]*ysc)
    t_local.append(qmap[k].group[i].retr["time"])
    
    d_cols.append(d_local)
    F_cols.append(F_local)
    t_cols.append(t_local)

    if len(d_cols) == len(x_data):
        d.append(d_cols)
        F.append(F_cols)
        t.append(t_cols)
        
        d_cols = []
        F_cols = []
        t_cols = []

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


contact_point_height = []

for m in range(len(F)):
    contact_point_height_cols = []
    argmin_list = procBasic.contactPoint2(F[m],d[m])
    for n in range(len(F[m])):
        contact_point_height_cols.append(d[m][n][0][argmin_list[n]])
    contact_point_height.append(contact_point_height_cols)
        


dataframe_qmap = pd.DataFrame(data=contact_point_height, index=y_data, columns=x_data)



ax = sns.heatmap(dataframe_qmap)
ax.set(xlabel='x (um)', ylabel='y (um)', title='QI map')
plt.show()



