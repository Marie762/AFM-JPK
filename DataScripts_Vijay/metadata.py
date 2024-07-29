# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 2024

@author: marie
"""

import os
import afmformats

# create JPKReader instances:
def JPKReaderList():
    allfilesinfolder = os.listdir(r'Data') 
    must_end_in = '.jpk-force'
    jpk_force_files = [os.path.join('Data',file) for file in allfilesinfolder if file[-len(must_end_in):] == must_end_in]

    # create empty list to store all the metadata extracted from each jpk-force file
    jpk_reader_list = []
    
    # for loop to extract and append all the separate jpk-force metadata to the list jpk_reader_list
    for i in range(len(jpk_force_files)):
        jpk_reader = afmformats.formats.fmt_jpk.jpk_reader.JPKReader(jpk_force_files[i])
        jpk_reader_list.append(jpk_reader)
    
    return jpk_reader_list

# separate functions to extract specific metadata for clarity

# useful metadata:
    # 'sensitivity' (m/V)?
    # 'spring constant' (N/m)
    # 'position x' (m)
    # 'position y' (m)
    # 'speed retract' (m/s)
    # 'speed approach' (m/s)
    # 'setpoint' (N)
    
def Sensitivity():
    sensitivity_list = []
    jpk_reader_list = JPKReaderList()
    for j in range(len(jpk_reader_list)):
        sensitivity = jpk_reader_list[j].get_metadata(0)['sensitivity'] # in m/V
        sensitivity_list.append(sensitivity)
    return sensitivity_list

def SpringConstant():
    spring_constant_list = []
    jpk_reader_list = JPKReaderList()
    for j in range(len(jpk_reader_list)):
        spring_constant = jpk_reader_list[j].get_metadata(0)['spring constant'] # in N/m
        spring_constant_list.append(spring_constant)
    return spring_constant_list

def Position():
    position_list = []
    jpk_reader_list = JPKReaderList()
    dsc = 1e6 # conversion to microns
    for j in range(len(jpk_reader_list)):
        position = []
        position_x = jpk_reader_list[j].get_metadata(0)['position x']*dsc # in micrometers
        position_y = jpk_reader_list[j].get_metadata(0)['position y']*dsc
        position.append(position_x)
        position.append(position_y)
        position_list.append(position)
    return position_list

def Speed():
    speed_list = []
    jpk_reader_list = JPKReaderList()
    dsc = 1e6 # conversion to microns
    for j in range(len(jpk_reader_list)):
        speed = []
        speed_approach = jpk_reader_list[j].get_metadata(0)['speed approach']*dsc # in micrometer/second
        speed_retract = jpk_reader_list[j].get_metadata(0)['speed retract']*dsc 
        speed.append(speed_approach)
        speed.append(speed_retract)
        speed_list.append(speed)
    return speed_list

def Setpoint():
    setpoint_list = []
    jpk_reader_list = JPKReaderList()
    for j in range(len(jpk_reader_list)):
        setpoint = jpk_reader_list[j].get_metadata(0)['setpoint'] # in N
        setpoint_list.append(setpoint)
    return setpoint_list

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
    # 'duration intermediate'
    # 'rate approach'
    # 'duration approach'
    # 'date'
    # 'time'