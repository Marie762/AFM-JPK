# -*- coding: utf-8 -*-
"""
Created on Tue Apr 2 15:19:50 2024

@author: marie
"""

import os
from tracemalloc import start
import numpy as np
import afmformats
import time
import pickle

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

def QI(load_from_pickle = False, data_path = r'Data_QI/'):
    allfilesinfolder = os.listdir(data_path) 
    must_end_in = '.jpk-qi-data'
    jpk_qi_data_files = [os.path.join(data_path,file) for file in allfilesinfolder if file[-len(must_end_in):] == must_end_in]

    # create empty list to store all the data extracted from each jpk-force file
    qmap = []
    for i in range(len(jpk_qi_data_files)):
        start_time = time.time()
        
        if not load_from_pickle:
            group = afmformats.AFMGroup(jpk_qi_data_files[i])
            qmapped_obj = afmformats.afm_qmap.AFMQMap(group)
            
            with open(jpk_qi_data_files[i][:-len(must_end_in)] + '.pkl', "wb") as output_file:
                pickle.dump( qmapped_obj, output_file)
            qmap.append(qmapped_obj)
        else:
            with open(jpk_qi_data_files[i][:-len(must_end_in)] + '.pkl','rb') as input_file:
                qmapped_obj = pickle.load( input_file )             
            qmap.append( qmapped_obj )
        
        print(f"Importing datafile {jpk_qi_data_files[i]} took {time.time() - start_time:1f} ")
    
    # scale conversion constants
    ysc = 1e9 # nN
    dsc = 1e6 # microns
    
    if not load_from_pickle:
        XY = []
        Q = []

        for q in qmap:
            xy_local = [
                np.around(q.get_qmap("data: height base point")[0], decimals=3),
                np.around(q.get_qmap("data: height base point")[1], decimals=3),
            ]
            XY.append(xy_local)

            d = []
            F = []
            t = []

            d_local = [
                [q.group[i].appr["height (measured)"] * dsc, q.group[i].retr["height (measured)"] * dsc]
                for i in range(len(q.group))
            ]
            F_local = [
                [q.group[i].appr["force"] * ysc, q.group[i].retr["force"] * ysc]
                for i in range(len(q.group))
            ]
            t_local = [
                [q.group[i].appr["time"], q.group[i].retr["time"]]
                for i in range(len(q.group))
            ]

            # Group data to maintain the same structure as the original code
            d = [d_local[i : i + len(xy_local[0])] for i in range(0, len(d_local), len(xy_local[0]))]
            F = [F_local[i : i + len(xy_local[0])] for i in range(0, len(F_local), len(xy_local[0]))]
            t = [t_local[i : i + len(xy_local[0])] for i in range(0, len(t_local), len(xy_local[0]))]

            q_local = [d, F, t]
            Q.append(q_local)

        with open(data_path + '/qmap.pkl', "wb") as output_file:
            pickle.dump(qmap, output_file)
        with open(data_path + '/Q.pkl', "wb") as output_file:
            pickle.dump(Q, output_file)
        with open(data_path + '/Xy.pkl', "wb") as output_file:
            pickle.dump(XY, output_file)
    else: 
        with open(data_path + '/qmap.pkl', "rb") as output_file:
            qmap = pickle.load(output_file)
        with open(data_path + '/Q.pkl', "rb") as output_file:
            Q = pickle.load(output_file)
        with open(data_path + '/XY.pkl', "rb") as output_file:
            XY = pickle.load(output_file)
    
    return qmap, Q, XY

if __name__ == '__main__':
    # extract the QI data from all the jpk-qi-data files in the directory 'Data_QI'
    qmap, Q, XY = QI(load_from_pickle=True)
    import matplotlib.pyplot as plt
    from plot import QIMap
    from contactPoint import QIcontactPoint2
    
    for k in range(len(qmap)):
        d = Q[k][0]
        F = Q[k][1]
        x_data = XY[k][0]
        y_data = XY[k][1]
        # if k == 0:
        #     contact_point_height = contactPoint.QIcontactPoint1(F,d)
        #     fig = QIMap(contact_point_height, y_data, x_data, k, save='True')
            
        if k == 1:
            contact_point_height = QIcontactPoint2(F,d)
            fig = QIMap(contact_point_height, y_data, x_data, k, save='True')

    plt.show()
        