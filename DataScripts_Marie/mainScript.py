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
#qmap = extractJPK.QI()

# test plotting
#fig = plot.Fd(F, d)
#fig = plot.Ft(F, t)
#plt.show()

# test basic processing
#max_value, max_element = procBasic.max(F)

F_bS = procBasic.baselineSubtraction(F)
#fig = plot.Fdsubplot(F, d, F_bS, subplot_name='baseline subtraction')
#fig = plot.Ftsubplot(F, t, F_bS, subplot_name='baseline subtraction')
#plt.show()


window_size = 200
poly_order = 2
F_smoothSG = procBasic.smoothingSG(F_bS, window_size, poly_order)

fig = plot.Fdsubplot(F_bS, d, F_smoothSG, subplot_name='S-G smoothing')
plt.show()