# -*- coding: utf-8 -*-
"""
Created on Tue Apr 2 15:29:30 2024

@author: marie
"""

import matplotlib.pylab as plt
from sklearn import feature_extraction

import extractJPK
import plot
import procBasic


# extract the data from all the jpk-force files in the directory 'Data'
d, F, t, segment, d_approach, F_approach, t_approach, d_inter, F_inter, t_inter, d_retract, F_retract, t_retract = extractJPK.force()

# test plotting
#fig = plot.Fd(F, F_approach, d_approach, F_inter, d_inter, F_retract, d_retract)
#fig = plot.Ft(F, F_approach, t_approach, F_inter, t_inter, F_retract, t_retract)
#plt.show()

#fig = plot.Fdsubplot(F, F_approach, d_approach, F_inter, d_inter, F_retract, d_retract, F_sub1, F_sub2, F_sub3)

# test basic processing
max_value, max_element = procBasic.max(F_approach)

F_bS1, F_bS2, F_bS3 = procBasic.baselineSubtraction(F_approach, F_inter, F_retract)
#fig = plot.Fdsubplot(F, F_approach, d_approach, F_inter, d_inter, F_retract, d_retract, F_bS1, F_bS2, F_bS3, subplot_name='baseline subtraction')
#fig = plot.Ftsubplot(F, F_approach, t_approach, F_inter, t_inter, F_retract, t_retract, F_bS1, F_bS2, F_bS3, subplot_name='baseline subtraction')


window_size = 10
poly_order = 2
F_smoothSG1, F_smoothSG2, F_smoothSG3 = procBasic.smoothingSG(F_bS1, F_bS2, F_bS3, window_size, poly_order)

fig = plot.Fdsubplot(F_bS1, d_approach, F_bS2, d_inter, F_bS3, d_retract, F_smoothSG1, F_smoothSG2, F_smoothSG3, subplot_name='S-G smoothing')
plt.show()