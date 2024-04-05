# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:57:50 2024

@author: marie
"""
import os
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import afmformats
import nanite

sns.set_theme(style="whitegrid", palette="muted")

allfilesinfolder = os.listdir(r'Data_QI') 
must_end_in = '.jpk-qi-series'
jpk_qi_series_files = [os.path.join('Data_QI',file) for file in allfilesinfolder if file[-len(must_end_in):] == must_end_in]

# create empty list to store all the data extracted from each jpk-force file
jpk_qi_series_list = []
    
# for loop to extract and append all the separate jpk-force data to the list jpk_force_data_list (length equal to the number of files in folder 'Data')
for i in range(len(jpk_qi_series_files)):
	data_extract = afmformats.load_data(jpk_qi_series_files[i])
	jpk_qi_series_list.append(data_extract)

print(jpk_qi_series_list[0][0].columns)
F = jpk_qi_series_list[0][0]['force']
print(len(F))