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


# to be able to load all files from a folder
allfilesinfolder = os.listdir(r'Data') 
must_end_in = '.jpk-force'
jpk_force_files = [os.path.join('Data',file) for file in allfilesinfolder if file[-len(must_end_in):] == must_end_in]

# create empty list
jpk_force_data_list = []
# for loop to add the all the separate data to a list with size equal to the number of files
for i in range(len(jpk_force_files)):
	data = afmformats.load_data(jpk_force_files[i])[0] # format of a dictionary
	jpk_force_data_list.append(data) # add the dictionary to a list

# to access the dictionaries in the list: list[0][0]["force"]


# scale conversion constants
xsc = 1 # none
ysc = 1e9 # nN
dsc = 1e6 # microns

d = []
F = []

for j in range(len(allfilesinfolder)):
	
	d.append(jpk_force_data_list[j]["height (measured)"]*dsc)
	F.append(jpk_force_data_list[j]["force"]*ysc)
	
for k in range(len(allfilesinfolder)):
	fig, ax = plt.subplots()
	ax.plot(d[k], F[k])

	ax.set(xlabel='height measured (um)', ylabel='force (nN)',
       title='Force-distance curve %i' % k)

	fig.savefig('Results\Fd_' + str(k) + '.png')
	# plt.show()
