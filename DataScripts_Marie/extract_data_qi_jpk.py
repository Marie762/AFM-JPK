# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:57:50 2024

@author: marie
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import afmformats
import nanite
import seaborn as sns
import pandas as pd




allfilesinfolder = os.listdir(r'Data_QI') 
must_end_in = '.jpk-qi-series'
jpk_qi_series_files = [os.path.join('Data_QI',file) for file in allfilesinfolder if file[-len(must_end_in):] == must_end_in]

# create empty list to store all the data extracted from each jpk-force file
jpk_qi_series_list = []




group = afmformats.AFMGroup("Data_QI\qi_Katerina_500nN_body_great training_150nm_pt40-data-2020.11.18-16.12.55.813-cropped-cropped.jpk-qi-data")

qmap = afmformats.afm_qmap.AFMQMap(group)
#print(qmap)


plot_qmap1 = qmap.get_qmap("data: height base point")
#print(plot_qmap1)

plot_qmap2 = qmap.get_qmap("data: piezo range")
#print(plot_qmap2)

plot_qmap3 = qmap.get_qmap("data: scan order")
#print(plot_qmap3)

#df = np.random.randn(10, 10)
#sns.heatmap(plot_qmap1[2])
sns.heatmap(plot_qmap1[2])
#sns.heatmap(plot_qmap3[2])
plt.show()