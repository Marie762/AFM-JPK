# -*- coding: utf-8 -*-
"""
Created on Tue Oct 1 2024

@author: marie
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from numpy import mean, std, percentile

# import all data from folder Data_csv
allfilesinfolder = os.listdir(r'Data_csv')
must_end_in = '.csv'
allfilesinfolder_csv = [os.path.join('Data_csv',file) for file in allfilesinfolder if file[-len(must_end_in):] == must_end_in]

data_list = []
for i in range(len(allfilesinfolder)):
            data_frame = pd.read_csv(allfilesinfolder_csv[i])
            data_list.append(data_frame)


# extract contact point, number of force drops, force drop, indentation depth, insertion force
tip_diameter_list, velocity_list, height_list, spring_constant_list, E_modulus_list = [],[],[],[],[]
insertion_events_present_list, indentation_depth_list, insertion_force_list, force_drop_list = [],[],[],[]
for i in range(len(data_list)):
    # height
    height_list.append(data_list[i]['height'].to_numpy())
    # velocity
    velocity_list.append(data_list[i]['insertion velocity'].to_numpy())
    # tip diameter
    tip_diameter_list.append(data_list[i]['tip diameter'].to_numpy())
    # spring constant
    spring_constant_list.append(data_list[i]['spring constant'].to_numpy())
    # E modulus
    E_modulus_list.append(data_list[i]['E modulus'].to_numpy())
    # indentation depth
    indentation_depth_list.append(data_list[i]['indentation depth'].to_numpy())
    # insertion force
    insertion_force_list.append(data_list[i]['insertion force'].to_numpy())  
    # force drop
    force_drop_list.append(data_list[i]['force drop'].to_numpy())
    # number of peaks
    number_of_peaks = data_list[i]['number of peaks'].to_numpy()
    insertion_events_present = []
    for j in range(len(number_of_peaks)):
        if number_of_peaks[j] == 0:
            insertion_events_present.append(0)
        elif number_of_peaks[j] > 4:
            insertion_events_present.append(5)
        else:
            insertion_events_present.append(number_of_peaks[j])
    insertion_events_present_list.append(np.array(insertion_events_present))

# concatenate into one long array
heights_conc = np.concatenate(height_list, axis=None)
velocity_conc = np.concatenate(velocity_list, axis=None)
tip_diameter_conc = np.concatenate(tip_diameter_list, axis=None)
spring_constant_conc = np.concatenate(spring_constant_list, axis=None)
print(len(spring_constant_conc), len(velocity_conc), len(tip_diameter_conc), len(heights_conc))
E_modulus_conc = np.concatenate(E_modulus_list, axis=None)
insertion_events_present_conc = np.concatenate(insertion_events_present_list, axis=None)
indentation_depth_conc = np.concatenate(indentation_depth_list, axis=None)
insertion_force_conc = np.concatenate(insertion_force_list, axis=None)
force_drop_conc = np.concatenate(force_drop_list, axis=None)


######### Stats

# velocity vs force drop/indentation depth/insertion force
# controls:

force_drop_array = []
forcedrop05, forcedrop1, forcedrop2, forcedrop5, forcedrop10, forcedrop20 = [],[],[],[],[],[]
insforce05, insforce1, insforce2, insforce5, insforce10, insforce20 = [],[],[],[],[],[]
indepth05, indepth1, indepth2, indepth5, indepth10, indepth20 = [],[],[],[],[],[]
vel_array = []
for i in range(len(heights_conc)):
    height = heights_conc[i] 
    force_drop = force_drop_conc[i]
    insertion_force = insertion_force_conc[i]
    indentation_depth = indentation_depth_conc[i]
    tip_diam = tip_diameter_conc[i] # control 2 um
    vel = velocity_conc[i] 
    sprg = spring_constant_conc[i] # control: not the microfluidic ones, so <0.5 N/m
    if height > 0.5 and tip_diam == 2 and sprg < 0.5 and force_drop != 0: # height > 0.5 and tip_diam == 2 and
        force_drop_array.append(force_drop)
        if vel == 0.5:
            forcedrop05.append(force_drop)
            insforce05.append(insertion_force)
            indepth05.append(indentation_depth)
        if vel == 1:
            forcedrop1.append(force_drop)
            insforce1.append(insertion_force)
            indepth1.append(indentation_depth)
        if vel == 2:
            forcedrop2.append(force_drop)
            insforce2.append(insertion_force)
            indepth2.append(indentation_depth)
        if vel == 5:
            forcedrop5.append(force_drop)
            insforce5.append(insertion_force)
            indepth5.append(indentation_depth)
        if vel == 10:
            forcedrop10.append(force_drop)
            insforce10.append(insertion_force)
            indepth10.append(indentation_depth)
        if vel == 20:
            forcedrop20.append(force_drop)
            insforce20.append(insertion_force)
            indepth20.append(indentation_depth)
        vel_array.append(vel)


data1 = forcedrop05
data2 = forcedrop1
data3 = forcedrop2
data4 = forcedrop5
data5 = forcedrop10
data6 = forcedrop20
print(len(data1),len(data2),len(data3),len(data4),len(data5),len(data6))

# # summarize
# print([mean(data1), std(data1), percentile(data1, 25), percentile(data1, 50), percentile(data1, 75)])
# print([mean(data2), std(data2), percentile(data2, 25), percentile(data2, 50), percentile(data2, 75)])
# print([mean(data3), std(data3), percentile(data3, 25), percentile(data3, 50), percentile(data3, 75)])
# print([mean(data4), std(data4), percentile(data4, 25), percentile(data4, 50), percentile(data4, 75)])
# print([mean(data5), std(data5), percentile(data5, 25), percentile(data5, 50), percentile(data5, 75)])
# print([mean(data6), std(data6), percentile(data6, 25), percentile(data6, 50), percentile(data6, 75)])



forcedrop_labels1 = ['0.5'] * len(forcedrop05)
forcedrop_labels2 = ['1'] * len(forcedrop1)
forcedrop_labels3 = ['2'] * len(forcedrop2)
forcedrop_labels4 = ['5'] * len(forcedrop5)
forcedrop_labels5 = ['10'] * len(forcedrop10)
forcedrop_labels6 = ['20'] * len(forcedrop20)

insforce_labels1 = ['0.5'] * len(insforce05)
insforce_labels2 = ['1'] * len(insforce1)
insforce_labels3 = ['2'] * len(insforce2)
insforce_labels4 = ['5'] * len(insforce5)
insforce_labels5 = ['10'] * len(insforce10)
insforce_labels6 = ['20'] * len(insforce20)

indepth_labels1 = ['0.5'] * len(indepth05)
indepth_labels2 = ['1'] * len(indepth1)
indepth_labels3 = ['2'] * len(indepth2)
indepth_labels4 = ['5'] * len(indepth5)
indepth_labels5 = ['10'] * len(indepth10)
indepth_labels6 = ['20'] * len(indepth20)

forcedrop_df = pd.DataFrame({"force drop": forcedrop05 + forcedrop1 +forcedrop2 + forcedrop5 + forcedrop10 + forcedrop20,
                   "velocity": forcedrop_labels1 + forcedrop_labels2 + forcedrop_labels3 + forcedrop_labels4 + forcedrop_labels5 + forcedrop_labels6})

insforce_df = pd.DataFrame({"insertion force": insforce05 + insforce1 +insforce2 + insforce5 + insforce10 + insforce20,
                   "velocity": insforce_labels1 + insforce_labels2 + insforce_labels3 + insforce_labels4 + insforce_labels5 + insforce_labels6})

indepth_df = pd.DataFrame({"indentation depth": indepth05 + indepth1 + indepth2 + indepth5 + indepth10 + indepth20,
                   "velocity": indepth_labels1 + indepth_labels2 + indepth_labels3 + indepth_labels4 + indepth_labels5 + indepth_labels6})


colors_diam_force_drop = sns.xkcd_palette(["olive green", "light olive","light yellow","goldenrod","orange","rust"])
fig, ax = plt.subplots()
sns.kdeplot(data=forcedrop_df, x="force drop", hue="velocity", common_norm=False, fill=True, bw_adjust=0.5, clip=(0, None), palette=colors_diam_force_drop)
# plt.xlim(0,0.5)
plt.title('Distribution of force drop values by velocity', fontsize=14)
plt.xlabel('Force drop (nN)', fontsize=14)
plt.ylabel('Density', fontsize=14)
fig.savefig('Results\KDE_velocity_force_drop_control_all.png', dpi=600)

fig, ax = plt.subplots()
sns.kdeplot(data=insforce_df, x="insertion force", hue="velocity", common_norm=False, fill=True, bw_adjust=0.5, clip=(0, None), palette=colors_diam_force_drop)
plt.title('Distribution of insertion force values by velocity', fontsize=14)
plt.xlabel('Insertion force (nN)', fontsize=14)
plt.ylabel('Density', fontsize=14)
fig.savefig('Results\KDE_velocity_insertion_force_control_all.png', dpi=600)

fig, ax = plt.subplots()
sns.kdeplot(data=indepth_df, x="indentation depth", hue="velocity", common_norm=False, fill=True, bw_adjust=0.5, clip=(0, None), palette=colors_diam_force_drop)
plt.title('Distribution of indentation depth values by velocity', fontsize=14)
plt.xlabel('Indentation depth (um)', fontsize=14)
plt.ylabel('Density', fontsize=14)
fig.savefig('Results\KDE_velocity_indentation_depth_control_all.png', dpi=600)



def counter(events):
    count0,count1,count2,count3,count4,count5 = 0,0,0,0,0,0
    count_tot = len(events)
    for ev in events:
        if ev == 0:
            count0 += 1
        if ev == 1:
            count1 += 1
        if ev == 2:
            count2 += 1
        if ev == 3:
            count3 += 1
        if ev == 4:
            count4 += 1
        if ev == 5:
            count5 += 1
    return count_tot, count0, count1, count2, count3, count4, count5

def countEvents(events_list):
    count0_list, count1_list, count2_list, count3_list, count4_list, count5_list, count_tot_list = [],[],[],[],[],[],[]
    for events in events_list:
        count_tot, count0, count1, count2, count3, count4, count5 = counter(events)
        count0_list.append(count0*100/count_tot)
        count1_list.append(count1*100/count_tot)
        count2_list.append(count2*100/count_tot)
        count3_list.append(count3*100/count_tot)
        count4_list.append(count4*100/count_tot)
        count5_list.append(count5*100/count_tot)
        count_tot_list.append(count_tot)
    return count0_list, count1_list, count2_list, count3_list, count4_list, count5_list, count_tot_list

##########################################################################
########### Insertion Velocity ###########################################

# #### Insertion Force
# force_drop_events_05, force_drop_events_1, force_drop_events_2, force_drop_events_5, force_drop_events_10, force_drop_events_20 = [],[],[],[],[],[]
# for i in range(len(velocity_conc)):
#     height = heights_conc[i] # control: must be on the cell so height > 1 um
#     tip_diam = tip_diameter_conc[i] # control 2 um
#     vel = velocity_conc[i] 
#     sprg = spring_constant_conc[i] # control: not the microfluidic ones, so <0.8 N/m
#     if tip_diam == 2 and height > 0.5 and sprg < 0.5 and insertion_force_conc[i] != 0:
#         if vel == 0.5:
#             force_drop_events_05.append(insertion_force_conc[i])
#         if vel == 1:
#             force_drop_events_1.append(insertion_force_conc[i])
#         if vel == 2:
#             force_drop_events_2.append(insertion_force_conc[i])
#         if vel == 5:
#             force_drop_events_5.append(insertion_force_conc[i])
#         if vel == 10:
#             force_drop_events_10.append(insertion_force_conc[i])
#         if vel == 20:
#             force_drop_events_20.append(insertion_force_conc[i])


#### Indentation Depth
# force_drop_events_05, force_drop_events_1, force_drop_events_2, force_drop_events_5, force_drop_events_10, force_drop_events_20 = [],[],[],[],[],[]
# for i in range(len(velocity_conc)):
#     height = heights_conc[i] # control: must be on the cell so height > 1 um
#     tip_diam = tip_diameter_conc[i] # control 2 um
#     vel = velocity_conc[i] 
#     sprg = spring_constant_conc[i] # control: not the microfluidic ones, so <0.8 N/m
#     if tip_diam == 2 and height > 0.5 and sprg < 0.5 and indentation_depth_conc[i] != 0:
#         if vel == 0.5:
#             force_drop_events_05.append(indentation_depth_conc[i])
#         if vel == 1:
#             force_drop_events_1.append(indentation_depth_conc[i])
#         if vel == 2:
#             force_drop_events_2.append(indentation_depth_conc[i])
#         if vel == 5:
#             force_drop_events_5.append(indentation_depth_conc[i])
#         if vel == 10:
#             force_drop_events_10.append(indentation_depth_conc[i])
#         if vel == 20:
#             force_drop_events_20.append(indentation_depth_conc[i])


#### Force drop    
force_drop_events_05, force_drop_events_1, force_drop_events_2, force_drop_events_5, force_drop_events_10, force_drop_events_20 = [],[],[],[],[],[]
for i in range(len(velocity_conc)):
    height = heights_conc[i] # control: must be on the cell so height > 1 um
    tip_diam = tip_diameter_conc[i] # control 2 um
    vel = velocity_conc[i] 
    sprg = spring_constant_conc[i] # control: not the microfluidic ones, so <0.8 N/m
    if tip_diam == 2 and height > 0.5 and sprg < 0.5 and force_drop_conc[i] != 0:
        if vel == 0.5:
            force_drop_events_05.append(force_drop_conc[i])
        if vel == 1:
            force_drop_events_1.append(force_drop_conc[i])
        if vel == 2:
            force_drop_events_2.append(force_drop_conc[i])
        if vel == 5:
            force_drop_events_5.append(force_drop_conc[i])
        if vel == 10:
            force_drop_events_10.append(force_drop_conc[i])
        if vel == 20:
            force_drop_events_20.append(force_drop_conc[i])

data0 = pd.DataFrame({'0.5': force_drop_events_05})
data1 = pd.DataFrame({'1': force_drop_events_1})
data2 = pd.DataFrame({'2': force_drop_events_2})
data3 = pd.DataFrame({'5': force_drop_events_5})
data4 = pd.DataFrame({'10': force_drop_events_10})
data5 = pd.DataFrame({'20': force_drop_events_20})

data_conc = pd.concat([data0, data1, data2, data3, data4, data5], axis=1) 

print(data_conc.head())

# # force drop plot
fig, ax = plt.subplots(figsize=(10,10))
colors_velocity_force_drop = sns.xkcd_palette(["olive green", "light olive","light yellow","goldenrod","orange","rust"])
sns.boxplot(data=data_conc, palette=colors_velocity_force_drop, linewidth=4)
# ax.set_title("Relationship between insertion velocity and force drop", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlabel(u'Velocity (\u03bcm/s)', fontsize=20)
ax.set_ylabel('Force drop (nN)', fontsize=20)
plt.ylim(0,0.35)
fig.savefig('Results\welocity_force_drop_limit.pdf', format='pdf')


# indentation depth plot
# fig, ax = plt.subplots(figsize=(9,10))
# colors_velocity_force_drop = sns.xkcd_palette(["olive green", "light olive","light yellow","goldenrod","orange","rust"])
# sns.boxplot(data=data_conc, palette=colors_velocity_force_drop)
# ax.set_title("Relationship between insertion velocity and indentation depth", fontsize=18)
# ax.set_xlabel('Velocity (um/s)', fontsize=14)
# ax.set_ylabel('Indentation depth (um)', fontsize=14)
# plt.ylim(0,3)
# fig.savefig('Results\welocity_indentation_depth_limit.png', dpi=600)


# insertion force plot
# fig, ax = plt.subplots(figsize=(9,10))
# colors_velocity_force_drop = sns.xkcd_palette(["olive green", "light olive","light yellow","goldenrod","orange","rust"])
# sns.boxplot(data=data_conc, palette=colors_velocity_force_drop)
# ax.set_title("Relationship between insertion velocity and insertion force", fontsize=18)
# ax.set_xlabel('Insertion velocity (um/s)', fontsize=14)
# ax.set_ylabel('Insertion force (nN)', fontsize=14)
# plt.ylim(0,4)
# fig.savefig('Results\welocity_insertion_force_limit.png', dpi=600)
# plt.close()



#### Number of insertion events

events_05, events_1, events_2, events_5, events_10, events_20 = [],[],[],[],[],[]
for i in range(len(velocity_conc)):
    height = heights_conc[i] # control: must be on the cell so height > 1 um
    tip_diam = tip_diameter_conc[i] # control 2 um
    vel = velocity_conc[i] 
    sprg = spring_constant_conc[i] # control: not the microfluidic ones, so <0.8 N/m
    if tip_diam == 2 and height > 0.5 and sprg < 0.5:
        if vel == 0.5:
            events_05.append(insertion_events_present_conc[i])
        if vel == 1:
            events_1.append(insertion_events_present_conc[i])
        if vel == 2:
            events_2.append(insertion_events_present_conc[i])
        if vel == 5:
            events_5.append(insertion_events_present_conc[i])
        if vel == 10:
            events_10.append(insertion_events_present_conc[i])
        if vel == 20:
            events_20.append(insertion_events_present_conc[i])

velocity_bins = ['0.5', '1', '2', '5', '10', '20']
events_vel_list = [events_05, events_1, events_2, events_5, events_10, events_20]

count0_list, count1_list, count2_list, count3_list, count4_list, count5_list, count_tot_list = countEvents(events_vel_list)

weight_counts_vel = {
                    # "0": np.array(count0_list),
                     "1": np.array(count1_list),"2": np.array(count2_list),"3": np.array(count3_list),"4": np.array(count4_list),"5": np.array(count5_list)}

fig, ax = plt.subplots()
bottom = np.zeros(6)
colors_vel = [
            # "xkcd:slate", 
              "xkcd:olive green", "xkcd:light olive","xkcd:light yellow","xkcd:goldenrod","xkcd:orange","xkcd:rust"]
for (boolean, weight_count),color in zip(weight_counts_vel.items(), colors_vel):
    p = ax.bar(velocity_bins, weight_count, 0.5, label=boolean, bottom=bottom, color=color, linewidth=4)
    bottom += weight_count
# ax.set_title("Relationship between insertion velocity and number of insertion events (%)", fontsize=18)
ax.set_xlabel(u'Insertion velocity (\u03bcm/s)', fontsize=15)
ax.set_ylabel('Percentage of total insertion events (%)', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.savefig('Results\welocity_number of insertion events_percentage_wozero.pdf', format='pdf')

