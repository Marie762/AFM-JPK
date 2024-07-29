# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:57:50 2024

@author: marie
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import axes, cm
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








#print(qmap.group[0].appr['force'])
#print(qmap.features)

plot_qmap1 = qmap.get_qmap("data: height base point")

x_data = np.around(plot_qmap1[0], decimals=3)
y_data = np.around(plot_qmap1[1], decimals=3)

print(x_data)

dataframe_qmap = pd.DataFrame(data=plot_qmap1[2], index=y_data, columns=x_data)

ax = sns.heatmap(dataframe_qmap, center=5.94)
ax.set(xlabel='x (um)', ylabel='y (um)', title='QI map')
plt.show()