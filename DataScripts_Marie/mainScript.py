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
print(spring_constant_list)

# extract the QI data from all the jpk-qi-data files in the directory 'Data_QI'
#qmap = extractJPK.QI()

