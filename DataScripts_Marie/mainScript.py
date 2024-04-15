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

# extract the force spectroscopy data from all the jpk-force files in the directory 'Data'
d, F, t = extractJPK.force()
spring_constant_list = metadata.SpringConstant()

# extract the QI data from all the jpk-qi-data files in the directory 'Data_QI'
#qmap = extractJPK.QI()

force_arr = F[3][0]

k = spring_constant_list[3]

deflection_arr = force_arr/k

z = d[3][0]

delta = z - deflection_arr

fig, ax = plt.subplots()
ax.plot(d[3][0], F[3][0], 'deepskyblue')
ax.plot(delta, F[3][0], 'orange')
ax.set(xlabel='height measured (um)', ylabel='force (nN)', title='Force-distance curve %i' % k)

plt.show()
