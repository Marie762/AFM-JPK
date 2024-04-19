# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 2024

@author: marie
"""
import os
import nanite
import afmformats

allfilesinfolder = os.listdir(r'Data_QI') 
must_end_in = '.jpk-qi-data'
jpk_qi_data_files = [os.path.join('Data_QI',file) for file in allfilesinfolder if file[-len(must_end_in):] == must_end_in]

i = 0
group = nanite.load_group(jpk_qi_data_files[i])
qmap = nanite.qmap.QMap(group)


for i in range(len(qmap.group)):
    idnt = qmap.group[i] # this is an instance of `nanite.Indentation`
    # apply preprocessing
    contact = qmap.feat_fit_contact_point(idnt)