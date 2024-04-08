# -*- coding: utf-8 -*-
"""
Created on Tue Apr 2 15:29:30 2024

@author: marie
"""

import matplotlib.pylab as plt

import extractJPK
import plot
import procBasic


# extract the data from all the jpk-force files in the directory 'Data'
d, F, t, segment, d_approach, F_approach, t_approach, d_inter, F_inter, t_inter, d_retract, F_retract, t_retract = extractJPK.force()

# test plotting
fig = plot.Fd(F, F_approach, d_approach, F_inter, d_inter, F_retract, d_retract)
fig = plot.Ft(F, F_approach, t_approach, F_inter, t_inter, F_retract, t_retract)
plt.show()

# test basic processing
#max_value, max_element = procBasic.max(F_approach)

#F_bS = procBasic.baselineSubtraction(F)
#fig = plot.Fdsubplot(F, d, F_bS, subplot_name='baseline subtraction')

#window_size = 10
#poly_order = 2
#F_smoothSG = procBasic.smoothingSG(F_bS, window_size, poly_order)

#fig = plot.Fdsubplot(F_bS, d, F_smoothSG, subplot_name='S-G smoothing')
