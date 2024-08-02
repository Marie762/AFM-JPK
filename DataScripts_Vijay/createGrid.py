# -*- coding: utf-8 -*-
"""
Created on Tue Apr 3 2024

@author: marie
"""

from metadata import Position

def grid10x10(arr):
    # create data array of 10 by 10 from an array of 100 points
    c0 = arr[:10]
    c1 = arr[10:20]
    r1 = c1[::-1]
    c2 = arr[20:30]
    c3 = arr[30:40]
    r3 = c3[::-1]
    c4 = arr[40:50]
    c5 = arr[50:60]
    r5 = c5[::-1]
    c6 = arr[60:70]
    c7 = arr[70:80]
    r7 = c7[::-1]
    c8 = arr[80:90]
    c9 = arr[90:]
    r9 = c9[::-1]
    grid_data = [c0,r1,c2,r3,c4,r5,c6,r7,c8,r9]
    # metadata
    x_position_list, y_position_list = Position()
    x_and_y_data = x_position_list[0:10] # 10x10
    return grid_data, x_and_y_data

def grid10x10_specialcase(arr):
    # create data array of 10 by 10 from an array of 60/100 points
    c0 = arr[:10]
    c1 = arr[10:20]
    r1 = c1[::-1]
    c2 = arr[20:30]
    c3 = arr[30:40]
    r3 = c3[::-1]
    c4 = arr[40:50]
    c5 = arr[50:60]
    r5 = c5[::-1]
    grid_data = [c0,r1,c2,r3,c4,r5]
    # metadata
    x_position_list, y_position_list = Position()
    x_data = x_position_list[0:6] # 10x10
    y_data = x_position_list[0:10] # 10x10
    return grid_data, x_data, y_data

def grid15x15(arr):
    # create data array of 15 by 15 from an array of 225 points
    c0 = arr[:15]
    c1 = arr[15:30]
    r1 = c1[::-1]
    c2 = arr[30:45]
    c3 = arr[45:60]
    r3 = c3[::-1]
    c4 = arr[60:75]
    c5 = arr[75:90]
    r5 = c5[::-1]
    c6 = arr[90:105]
    c7 = arr[105:120]
    r7 = c7[::-1]
    c8 = arr[120:135]
    c9 = arr[135:150]
    r9 = c9[::-1]
    c10 = arr[150:165]
    c11 = arr[165:180]
    r11 = c11[::-1]
    c12 = arr[180:195]
    c13 = arr[195:210]
    r13 = c13[::-1]
    c14 = arr[210:225]

    grid_data = [c0,r1,c2,r3,c4,r5,c6,r7,c8,r9,c10,r11,c12,r13,c14]
    # metadata
    x_position_list, y_position_list = Position()
    x_and_y_data = x_position_list[0:15] # 15x15
    return grid_data, x_and_y_data

def grid15x15_specialcase(arr):
    # create data array of 15 by 15 from an array of 225 points where it stopped at 183/225
    c0 = arr[:15]
    c1 = arr[15:30]
    r1 = c1[::-1]
    c2 = arr[30:45]
    c3 = arr[45:60]
    r3 = c3[::-1]
    c4 = arr[60:75]
    c5 = arr[75:90]
    r5 = c5[::-1]
    c6 = arr[90:105]
    c7 = arr[105:120]
    r7 = c7[::-1]
    c8 = arr[120:135]
    c9 = arr[135:150]
    r9 = c9[::-1]
    c10 = arr[150:165]
    c11 = arr[165:180]
    r11 = c11[::-1]
    
    grid_data = [c0,r1,c2,r3,c4,r5,c6,r7,c8,r9,c10,r11]
    # metadata
    x_position_list, y_position_list = Position()
    x_data = x_position_list[0:12] # 12
    y_data = x_position_list[0:15] # 15
    return grid_data, x_data, y_data


def grid20x20(arr):
    # create data array of 20 by 20 from an array of 400 points
    c0 = arr[:20]
    c1 = arr[20:40]
    r1 = c1[::-1]
    c2 = arr[40:60]
    c3 = arr[60:80]
    r3 = c3[::-1]
    c4 = arr[80:100]
    c5 = arr[100:120]
    r5 = c5[::-1]
    c6 = arr[120:140]
    c7 = arr[140:160]
    r7 = c7[::-1]
    c8 = arr[160:180]
    c9 = arr[180:200]
    r9 = c9[::-1]
    c10 = arr[200:220]
    c11 = arr[220:240]
    r11 = c11[::-1]
    c12 = arr[240:260]
    c13 = arr[260:280]
    r13 = c13[::-1]
    c14 = arr[280:300]
    c15 = arr[300:320]
    r15 = c15[::-1]
    c16 = arr[320:340]
    c17 = arr[340:360]
    r17 = c17[::-1]
    c18 = arr[360:380]
    c19 = arr[380:400]
    r19 = c19[::-1]
    grid_data = [c0,r1,c2,r3,c4,r5,c6,r7,c8,r9,c10,r11,c12,r13,c14,r15,c16,r17,c18,r19]
    # metadata
    x_position_list, y_position_list = Position()
    x_and_y_data = x_position_list[0:20] # 20x20
    return grid_data, x_and_y_data

def grid25x25(arr):
    # create data array of 25 by 25 from an array of 625 points
    c0 = arr[:25]
    c1 = arr[25:50]
    r1 = c1[::-1]
    c2 = arr[50:75]
    c3 = arr[75:100]
    r3 = c3[::-1]
    c4 = arr[100:125]
    c5 = arr[125:150]
    r5 = c5[::-1]
    c6 = arr[150:175]
    c7 = arr[175:200]
    r7 = c7[::-1]
    c8 = arr[200:225]
    c9 = arr[225:250]
    r9 = c9[::-1]
    c10 = arr[250:275]
    c11 = arr[275:300]
    r11 = c11[::-1]
    c12 = arr[300:325]
    c13 = arr[325:350]
    r13 = c13[::-1]
    c14 = arr[350:375]
    c15 = arr[375:400]
    r15 = c15[::-1]
    c16 = arr[400:425]
    c17 = arr[425:450]
    r17 = c17[::-1]
    c18 = arr[450:475]
    c19 = arr[475:500]
    r19 = c19[::-1]
    c20 = arr[500:525]
    c21 = arr[525:550]
    r21 = c21[::-1]
    c22 = arr[550:575]
    c23 = arr[575:600]
    r23 = c23[::-1]
    c24 = arr[600:]
    grid_data = [c0,r1,c2,r3,c4,r5,c6,r7,c8,r9,c10,r11,c12,r13,c14,r15,c16,r17,c18,r19,c20,r21,c22,r23,c24]
    # metadata
    x_position_list, y_position_list = Position()
    x_and_y_data = x_position_list[0:25] # 25x25
    return grid_data, x_and_y_data
