# -*- coding: utf-8 -*-
"""
Created on Tue Apr 2 15:29:30 2024

@author: marie
"""

import matplotlib.pylab as plt
import numpy as np
import pandas
import afmformats

import extractJPK
import plot
import procBasic


# extract the data from all the jpk-force files in the directory 'Data'
d, F, t = extractJPK.force()

# test plotting
fig = plot.Fd(F, d)
#fig = plot.Ft(F, t)
#plt.show()

# test basic processing
max_value, max_element = procBasic.max(F)
print(max_value, max_element)
if F[max_element] == max_value:
    print(True)

#F_bS = procBasic.baselineSubtraction(F)

#window_size = 100
#poly_order = 2
#F_smoothSG = procBasic.smoothingSG(F, window_size, poly_order)

#fig = plot.Fdsubplot(F, d, F_bS)
#plt.show()