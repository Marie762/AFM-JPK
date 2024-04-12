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


# extract the force spectroscopy data from all the jpk-force files in the directory 'Data'
d, F, t = extractJPK.force()

# extract the QI data from all the jpk-qi-data files in the directory 'Data_QI'
qmap = extractJPK.QI()

# test plotting
fig = plot.Fd(F, d)
fig = plot.Ft(F, t)
plt.show()

# test basic processing
max_value, max_element = procBasic.max(F)

F_bS = procBasic.baselineSubtraction(F)
#fig = plot.Fdsubplot(F, F_approach, d_approach, F_inter, d_inter, F_retract, d_retract, F_bS1, F_bS2, F_bS3, subplot_name='baseline subtraction')
#fig = plot.Ftsubplot(F, F_approach, t_approach, F_inter, t_inter, F_retract, t_retract, F_bS1, F_bS2, F_bS3, subplot_name='baseline subtraction')



window_size = 10
poly_order = 2
#F_smoothSG1, F_smoothSG2, F_smoothSG3 = procBasic.smoothingSG(F_bS1, F_bS2, F_bS3, window_size, poly_order)

#fig = plot.Fdsubplot(F_bS1, d_approach, F_bS2, d_inter, F_bS3, d_retract, F_smoothSG1, F_smoothSG2, F_smoothSG3, subplot_name='S-G smoothing')
#plt.show()