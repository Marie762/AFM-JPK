# -*- coding: utf-8 -*-
"""
Created on Tue Apr 2 15:19:50 2024

@author: marie
"""

import os
import afmformats

def force():
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

    # scale conversion constants
    ysc = 1e9 # nN
    dsc = 1e6 # microns

    # create three empty lists to store the height (d), force (F), and time (t) values of each jpk-force file
    d = []
    F = []
    t = []
    segment = []

    # add all the height, force, and time data to separate lists, with the element corresponding to the jpk_force_data_list
    for j in range(len(jpk_force_files)):
        d.append(jpk_force_data_list[j][0]["height (measured)"]*dsc)
        F.append(jpk_force_data_list[j][0]["force"]*ysc)
        t.append(jpk_force_data_list[j][0]["time"])
        segment.append(jpk_force_data_list[j][0]["segment"])
       
    # create more empty lists to store approch and retract curves 
    d_approach = []
    F_approach = []
    t_approach = []
    d_inter = []
    F_inter = []
    t_inter = []
    d_retract = []
    F_retract = []
    t_retract = []
    
    for x in range(len(F)):
        local_d0_list = []
        local_F0_list = []
        local_t0_list = []
        local_d1_list = []
        local_F1_list = []
        local_t1_list = []
        local_d2_list = []
        local_F2_list = []
        local_t2_list = []
        
        for y in range (len(F[x])):
            if segment[x][y] == 0:
                # approach
                local_d0_list.append(d[x][y])
                local_F0_list.append(F[x][y])
                local_t0_list.append(t[x][y])
            if segment[x][y] == 1:
                # intermediate
                local_d1_list.append(d[x][y])
                local_F1_list.append(F[x][y])
                local_t1_list.append(t[x][y])
            if segment[x][y] == 2:
                # retract
                local_d2_list.append(d[x][y])
                local_F2_list.append(F[x][y])
                local_t2_list.append(t[x][y])
        
        # append local arrays to corresponding list
        d_approach.append(local_d0_list)
        F_approach.append(local_F0_list)
        t_approach.append(local_t0_list)
        d_inter.append(local_d1_list)
        F_inter.append(local_F1_list)
        t_inter.append(local_t1_list)
        d_retract.append(local_d2_list)
        F_retract.append(local_F2_list)
        t_retract.append(local_t2_list)
        
    
    return d, F, t, segment, d_approach, F_approach, t_approach, d_inter, F_inter, t_inter, d_retract, F_retract, t_retract