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

allfilesinfolder = os.listdir(r'Data') 
must_end_in = '.jpk-force'
jpk_force_files = [os.path.join('Data',file) for file in allfilesinfolder if file[-len(must_end_in):] == must_end_in]

# create empty list to store all the data extracted from each jpk-force file
jpk_force_data_list = []
    
# for loop to extract and append all the separate jpk-force data to the list jpk_force_data_list (length equal to the number of files in folder 'Data')
for i in range(len(jpk_force_files)):
    data_extract = afmformats.load_data(jpk_force_files[i])
    jpk_force_data_list.append(data_extract)

# to access specific data in jpk_force_data_list: 
# jpk_force_data_list[0].columns - returns the column names of file 1 (corresponds to element 0)
# jpk_force_data_list[0]["force"] - returns an array with the force data of file 1 (corresponds to element 0)

# # scale conversion constants
# ysc = 1e9 # nN
# dsc = 1e6 # microns

# # create three empty lists to store the height (d), force (F), and time (t) values of each jpk-force file  
# d = []
# F = []
# t = []

# # add all the height, force, and time data to separate lists, with the element corresponding to the jpk_force_data_list
# for j in range(len(jpk_force_files)):
#     # create three empty lists to locally store the [approach, intermediate, retract] data
#     d_local = []
#     F_local = []
#     t_local = []
    
#     d_local.append(jpk_force_data_list[j][0].appr["height (measured)"]*dsc)
#     F_local.append(jpk_force_data_list[j][0].appr["force"]*ysc)
#     t_local.append(jpk_force_data_list[j][0].appr["time"])
    
#     if jpk_force_data_list[j][0].modality == 'creep compliance':
#         d_local.append(jpk_force_data_list[j][0].intr["height (measured)"]*dsc)
#         F_local.append(jpk_force_data_list[j][0].intr["force"]*ysc)
#         t_local.append(jpk_force_data_list[j][0].intr["time"])
        
#     d_local.append(jpk_force_data_list[j][0].retr["height (measured)"]*dsc)
#     F_local.append(jpk_force_data_list[j][0].retr["force"]*ysc)
#     t_local.append(jpk_force_data_list[j][0].retr["time"])

#     d.append(d_local)
#     F.append(F_local)
#     t.append(t_local)
    
# try loading raw data
#data_raw = afmformats.formats.fmt_jpk.jpk_data.load_dat_raw('Data_raw\force-save-2023.06.09-16.22.19.714.txt', 'vDeflection', 'sensitivity')  

jpk_reader = afmformats.formats.fmt_jpk.jpk_reader.JPKReader(jpk_force_files[0])
print(jpk_reader)

md = jpk_reader.get_metadata(0)

print(md)

# useful metadata:
    # 'sensitivity' (m/V)?
    # 'spring constant' (N/m)
    # 'position x' (m)
    # 'position y' (m)
    # 'speed retract' (m/s)
    # 'speed approach' (m/s)
    # 'setpoint' (N)

# other metadata:
# 'feedback mode'
# 'duration'
# 'point count'
# 'session id'
# 'instrument'
# 'software version'
# 'software'
# 'enum'
# 'path'
# 'rate retract'
# 'duration retract'
# 'segment count'
# 'imaging mode'
# 'curve id'
# 'rate approach'
# 'duration approach'
# 'date'
# 'time'
