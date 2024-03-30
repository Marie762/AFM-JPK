# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:57:50 2024

@author: marie
"""

import matplotlib.pylab as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid", palette="muted")

import afmformats

# extract tsv file

#data=pd.read_csv(r'C:\Users\marie\Desktop\data_P3\qi_cell1_p3_4h_1nN-data-2020.10.05-15.36.52.774_processed-2020.10.13-10.27.09.tsv',sep='\t')

# for col in data.columns:
#     print(col)


data2 = afmformats.formats.fmt_jpk.load_jpk(r'C:\Users\marie\Desktop\data2\qi_cell1_1nN_glass_4h-data-2020.08.10-13.50.53.433.jpk-qi-data')

print(data2[0].columns)

#%%
# afmformats useful commands:
	# afmformats.load_data(r"path")
	# dslist[0].columns
	# dslist[0]["force"]

#%% Extract from file

# # to load all files from a folder
# allfilesinfolder = glob.glob(r'C:\Users\marie\Desktop\data\force-save-2023.06.09-*.jpk-force')

# # create empty list
# list = []

# # for loop to add the all the separate data to a list with size equal to the number of files
# for i in range(len(allfilesinfolder) - 1):
# 	data = afmformats.load_data(allfilesinfolder[i]) # format of a dictionary
# 	list.append(data) # add the dictionary to a list

# # to access the dictionaries in the list: list[0][0]["force"]



#%%
# scale conversion constants
xsc = 1 # none
ysc = 1e9 # nN
dsc = 1e6 # microns

# create empty list for force and distance arrays
d = []
F = []

for j in range(len(allfilesinfolder) - 1):
	# add arrays to their respective lists
	d.append(list[j][0]["height (measured)"]*dsc)
	F.append(list[j][0]["force"]*ysc)
	
# to access an array in the list: d[0]
# to access one element in the array in the list: d[0][0]


#%% Plotting:
	
# for k in range(len(allfilesinfolder) - 1):
#  	fig, ax = plt.subplots()
#  	ax.plot(d[k], F[k])

#  	ax.set(xlabel='height measured (um)', ylabel='force (nN)',
#         title='Force-distance curve %i' % k)

#  	fig.savefig('Force curves on nucleus vijay\Fd_' + str(k) + '.png')
#  	#plt.show()

