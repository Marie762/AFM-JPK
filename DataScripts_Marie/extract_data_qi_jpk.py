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



allfilesinfolder = os.listdir(r'Data_QI') 
must_end_in = '.jpk-qi-series'
jpk_qi_series_files = [os.path.join('Data_QI',file) for file in allfilesinfolder if file[-len(must_end_in):] == must_end_in]

# create empty list to store all the data extracted from each jpk-force file
jpk_qi_series_list = []




group = afmformats.AFMGroup("Data_QI\qi_Katerina_500nN_body_great training_150nm_pt40-data-2020.11.18-16.12.55.813-cropped-cropped.jpk-qi-data")

qmap = afmformats.afm_qmap.AFMQMap(group)

print(qmap.features)

plot_qmap = qmap.get_qmap("data: height base point")

viridis = cm.get_cmap('viridis', 256)
print(viridis)
newcolors = viridis(np.linspace(0, 1, 256))
print(newcolors)
pink = np.array([248/256, 24/256, 148/256, 1])
print(pink)
newcolors[:25, :] = pink
print(newcolors)
newcmp = ListedColormap(newcolors)
print(newcmp)

def plot_examples(cms):
    """
    helper function to plot two colormaps
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()

plot_examples([viridis, newcmp])