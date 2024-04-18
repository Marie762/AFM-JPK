# -*- coding: utf-8 -*-
"""
Created on Tue Apr 2 15:19:50 2024

@author: marie
"""

import os
import numpy as np
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

    # add all the height, force, and time data to separate lists, with the element corresponding to the jpk_force_data_list
    for j in range(len(jpk_force_files)):
        # create three empty lists to locally store the [approach, intermediate, retract] data
        d_local = []
        F_local = []
        t_local = []
        
        d_local.append(jpk_force_data_list[j][0].appr["height (measured)"]*dsc)
        F_local.append(jpk_force_data_list[j][0].appr["force"]*ysc)
        t_local.append(jpk_force_data_list[j][0].appr["time"])
        
        if jpk_force_data_list[j][0].modality == 'creep compliance':
            d_local.append(jpk_force_data_list[j][0].intr["height (measured)"]*dsc)
            F_local.append(jpk_force_data_list[j][0].intr["force"]*ysc)
            t_local.append(jpk_force_data_list[j][0].intr["time"])
            
        d_local.append(jpk_force_data_list[j][0].retr["height (measured)"]*dsc)
        F_local.append(jpk_force_data_list[j][0].retr["force"]*ysc)
        t_local.append(jpk_force_data_list[j][0].retr["time"])
    
        d.append(d_local)
        F.append(F_local)
        t.append(t_local)
    
    return d, F, t

def QI():
    allfilesinfolder = os.listdir(r'Data_QI') 
    must_end_in = '.jpk-qi-data'
    jpk_qi_data_files = [os.path.join('Data_QI',file) for file in allfilesinfolder if file[-len(must_end_in):] == must_end_in]

    # create empty list to store all the data extracted from each jpk-force file
    qmap = []
    for i in range(len(jpk_qi_data_files)):
        group = afmformats.AFMGroup(jpk_qi_data_files[i])
        qmap.append(afmformats.afm_qmap.AFMQMap(group))
    
    # scale conversion constants
    ysc = 1e9 # nN
    dsc = 1e6 # microns
    
    Q = []
    XY = []
    
    for k in range(len(qmap)):
        xy_local = []
        
        get_qmap = qmap[k].get_qmap("data: height base point")
        x_data = np.around(get_qmap[0], decimals=3)
        y_data = np.around(get_qmap[1], decimals=3)
        xy_local.append(x_data)
        xy_local.append(y_data)
        XY.append(xy_local)
        
        q_local = []
          
        d,F,t = [],[],[]

        d_cols = []
        F_cols = []
        t_cols = []

        for i in range(len(qmap[k].group)):
            d_local = []
            F_local = []
            t_local = []
            
            # qmap[1].group[i]: force-distance data i 
            d_local.append(qmap[k].group[i].appr["height (measured)"]*dsc)
            F_local.append(qmap[k].group[i].appr["force"]*ysc)
            t_local.append(qmap[k].group[i].appr["time"])
            
            d_local.append(qmap[k].group[i].retr["height (measured)"]*dsc)
            F_local.append(qmap[k].group[i].retr["force"]*ysc)
            t_local.append(qmap[k].group[i].retr["time"])
            
            d_cols.append(d_local)
            F_cols.append(F_local)
            t_cols.append(t_local)

            if len(d_cols) == len(x_data):
                d.append(d_cols)
                F.append(F_cols)
                t.append(t_cols)
                
                d_cols = []
                F_cols = []
                t_cols = []
        
        q_local.append(d)
        q_local.append(F)
        q_local.append(t)
        Q.append(q_local)
    
    return qmap, Q, XY

