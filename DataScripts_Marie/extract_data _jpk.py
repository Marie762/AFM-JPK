# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:57:50 2024

@author: marie
"""

import os
import matplotlib.pylab as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas
import seaborn as sns
import afmformats

sns.set_theme(style="whitegrid", palette="muted")

# afmformats useful commands:
	# afmformats.load_data(r"path")
	# dslist[0].columns
	# dslist[0]["force"]


# List all jpk-force files from the folder 'Data'

allfilesinfolder = os.listdir(r'Data') 
must_end_in = '.jpk-force'
jpk_force_files = [os.path.join('Data',file) for file in allfilesinfolder if file[-len(must_end_in):] == must_end_in]

# create empty list to store all the data extracted from each jpk-force file
jpk_force_data_list = []
# for loop to extract and append all the separate jpk-force data to the list jpk_force_data_list (length equal to the number of files in folder 'Data')
for i in range(len(jpk_force_files)):
	data_extract = afmformats.load_data(jpk_force_files[i])[0] 
	jpk_force_data_list.append(data_extract) 
	print(jpk_force_data_list)

# to access specific data in jpk_force_data_list: 
	# jpk_force_data_list[0].columns - returns the column names of file 1 (corresponds to element 0)
	# jpk_force_data_list[0]["force"] - returns an array with the force data of file 1 (corresponds to element 0)


# scale conversion constants
ysc = 1e9 # nN
dsc = 1e6 # microns

# create three empty lists to store the height (d), force (F), and time (t) values of each jpk-force file
d = []
F = []
t = []

# add all the height, force, and time data to separate lists, with the element corresponding to the jpk_force_data_list
for j in range(len(jpk_force_files)):
	d.append(jpk_force_data_list[j]["height (measured)"]*dsc)
	F.append(jpk_force_data_list[j]["force"]*ysc)
	t.append(jpk_force_data_list[j]["time"])

# Plotting force-distance curve and saving a png to the 'Results' folder
for k in range(len(jpk_force_files)):
    # create figure 'ax' and plot F against d (force-distance curve)
    fig, ax = plt.subplots()
    ax.plot(d[k], F[k])
    ax.set(xlabel='height measured (um)', ylabel='force (nN)', title='Force-distance curve %i' % k)
    fig.savefig('Results\Fd_' + str(k) + '.png')

# Plotting force-time curve and saving a png to the 'Results' folder
for k in range(len(jpk_force_files)):
    # create figure 'ax' and plot F against d (force-distance curve)
    fig, ax = plt.subplots()
    ax.plot(t[k], F[k])
    ax.set(xlabel='time (s)', ylabel='force (nN)', title='Force-time curve %i' % k)
    fig.savefig('Results\Ft_' + str(k) + '.png')
    #plt.show()