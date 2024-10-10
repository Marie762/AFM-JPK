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
from scipy import stats
from numpy import mean, std, percentile
from scipy.stats import mannwhitneyu, wilcoxon, kruskal, friedmanchisquare

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

# tip diameter vs force drop
# controls:

force_drop_array = []
forcedrop07, forcedrop1, forcedrop2 = [],[],[]
insforce07, insforce1, insforce2 = [],[],[]
indepth07, indepth1, indepth2 = [],[],[]
tip_diam_array = []
for i in range(len(heights_conc)):
    height = heights_conc[i]
    force_drop = force_drop_conc[i]
    insertion_force = insertion_force_conc[i]
    indentation_depth = indentation_depth_conc[i]
    tip_diam = tip_diameter_conc[i] # control 2 um
    vel = velocity_conc[i] # control 2 um/s
    sprg = spring_constant_conc[i] # control: not the microfluidic ones, so <0.5 N/m
    if height > 0.5 and vel == 2 and sprg < 0.5 and force_drop != 0: # height > 0.5 and vel == 2 and
        force_drop_array.append(force_drop)
        if tip_diam == 0.2:
            tip_diam = 0.7
            forcedrop07.append(force_drop)
            insforce07.append(insertion_force)
            indepth07.append(indentation_depth)
        if tip_diam == 1:
            forcedrop1.append(force_drop)
            insforce1.append(insertion_force)
            indepth1.append(indentation_depth)
        if tip_diam == 2:
            forcedrop2.append(force_drop)
            insforce2.append(insertion_force)
            indepth2.append(indentation_depth)
        tip_diam_array.append(tip_diam)



# # Lets take tip diameters 1 and 2 microns
# # and force drop data

# data1 = indepth07
# data2 = indepth1
# data3 = indepth2
# print(len(data1),len(data2),len(data3))
# # summarize
# print('data1: ', [mean(data1), std(data1), percentile(data1, 25), percentile(data1, 50), percentile(data1, 75)])
# print('data2: ', [mean(data2), std(data2), percentile(data2, 25), percentile(data2, 50), percentile(data2, 75)])
# print('data3: ', [mean(data3), std(data3), percentile(data3, 25), percentile(data3, 50), percentile(data3, 75)])


# sys.exit()
# # choose alpha = 0.05
# alpha = 0.05

# # normality test
# normal_stat, normal_p = stats.normaltest(data2)
# print('Normality statistic=%.3f, p=%.10f' % (normal_stat, normal_p))


# ##### Parametric tests
# # Two-Sample T-Test
# # Perform the t-test:
# t_stat, p_value = stats.ttest_ind(data1, data2)
# print('T statistic=%.3f, p=%.10f' % (t_stat, p_value))
# if p_value < alpha:
#     print('Different distribution (reject H0)')
# else:
#     print('Same distribution (fail to reject H0)')

# sys.exit()
# ##### Non-parametric tests
# # Mann-Whitney U test (two samples)
# print('MANN-WHITNEY U TEST')
#     # for randomly selected values X and Y from two populations,
#     # the probability of X being greater than Y is equal to the 
#     # probability of Y being greater than X.

# stat, p = mannwhitneyu(data1, data2)
# print('Statistics=%.3f, p=%.3f' % (stat, p))
# # interpret
# if p > alpha:
#     print('Same distribution (fail to reject H0)')
# else:
#     print('Different distribution (reject H0)')


# # Kruskal-Wallis H Test (more than two samples)
# print('KRUSKAL-WALLIS H TEST')
# stat,p=kruskal(data1,data2,data3)
# print('Statistics=%.3f, p=%.10f' % (stat, p))
# # interpret
# if p > alpha:
#     print('Same distribution (fail to reject H0)')
# else:
#     print('Different distribution (reject H0)')



# # Wilcoxon Signed-Rank Test 
# print('WILCOXON SIGNED-RANK TEST')
# # Doesnt work for this data, because arrays are different lengths.
# # But it will work for the number of force drops data.

# # two paired samples
# # compare samples
# stat, p = wilcoxon(data1, data2)
# print('Statistics=%.3f, p=%.3f' % (stat, p))
# # interpret
# if p > alpha:
#     print('Same distribution (fail to reject H0)')
# else:
#     print('Different distribution (reject H0)')
    
# # one sample: testing the difference between the two samples
# stat, p = wilcoxon(data1-data2)

# print('Statistics=%.3f, p=%.3f' % (stat, p))
# # interpret
# if p > alpha:
#     print('Same distribution (fail to reject H0)')
# else:
#     print('Different distribution (reject H0)')


# # Friedman Test 
# print('FRIEDMAN TEST')
# Use on paired data, so the number of force drops will work here
# # compare samples
# stat, p = friedmanchisquare(data1, data2, data3)
# print('Statistics=%.3f, p=%.3f' % (stat, p))
# # interpret
# alpha = 0.05
# if p > alpha:
#     print('Same distributions (fail to reject H0)')
# else:
#     print('Different distributions (reject H0)')


##### KDE ########################
forcedrop_labels1 = ['0.7'] * len(forcedrop07)
forcedrop_labels2 = ['1'] * len(forcedrop1)
forcedrop_labels3 = ['2'] * len(forcedrop2)

insforce_labels1 = ['0.7'] * len(insforce07)
insforce_labels2 = ['1'] * len(insforce1)
insforce_labels3 = ['2'] * len(insforce2)

indepth_labels1 = ['0.7'] * len(indepth07)
indepth_labels2 = ['1'] * len(indepth1)
indepth_labels3 = ['2'] * len(indepth2)

forcedrop_df = pd.DataFrame({"force drop": forcedrop07 + forcedrop1 +forcedrop2,
                   "tip diameter": forcedrop_labels1 + forcedrop_labels2 + forcedrop_labels3})

insforce_df = pd.DataFrame({"insertion force": insforce07 + insforce1 +insforce2,
                   "tip diameter": insforce_labels1 + insforce_labels2 + insforce_labels3})

indepth_df = pd.DataFrame({"indentation depth": indepth07 + indepth1 +indepth2,
                   "tip diameter": indepth_labels1 + indepth_labels2 + indepth_labels3})


colors_diam_force_drop = sns.xkcd_palette(["green","blue green","dark teal"])
fig, ax = plt.subplots()
sns.kdeplot(data=forcedrop_df, x="force drop", hue="tip diameter", common_norm=False, fill=True, bw_adjust=0.5, clip=(0, None), palette=colors_diam_force_drop)
plt.title('Distribution of force drop values by tip diameter', fontsize=14)
plt.xlabel('Force drop (nN)', fontsize=14)
plt.ylabel('Density', fontsize=14)
fig.savefig('Results\KDE_diameter_force_drop_control_height1.pdf', format='pdf')

fig, ax = plt.subplots()
sns.kdeplot(data=insforce_df, x="insertion force", hue="tip diameter", common_norm=False, fill=True, bw_adjust=0.5, clip=(0, None), palette=colors_diam_force_drop)
plt.title('Distribution of insertion force values by tip diameter', fontsize=14)
plt.xlabel('Insertion force (nN)', fontsize=14)
plt.ylabel('Density', fontsize=14)
fig.savefig('Results\KDE_diameter_insertion_force_control_height1.pdf', format='pdf')

fig, ax = plt.subplots()
sns.kdeplot(data=indepth_df, x="indentation depth", hue="tip diameter", common_norm=False, fill=True, bw_adjust=0.5, clip=(0, None), palette=colors_diam_force_drop)
plt.title('Distribution of indentation depth values by tip diameter', fontsize=14)
plt.xlabel('Indentation depth (um)', fontsize=14)
plt.ylabel('Density', fontsize=14)
fig.savefig('Results\KDE_diameter_indentation_depth_control_height1.pdf', format='pdf')


# sys.exit()
# x = np.array(tip_diam_array)
# y = np.array(force_drop_array)
# res = stats.linregress(x, y)
# print(f"R-squared: {res.rvalue**2:.6f}")

# text = 'slope: ' + str(res.slope)
# text2 = f"R-squared: {res.rvalue**2:.6f}"

# #["spring green","light teal","blue green"]
# fig, ax = plt.subplots()
# plt.plot(x, y, 'o', color="xkcd:light teal", label='original data')
# plt.plot(x, res.intercept + res.slope*x, "xkcd:blue green", label='fitted line')
# plt.legend()
# plt.text(3.4, 2, text)
# plt.text(3.4, 1.5, text2)
# ax.set_title("Relationship between tip diameter and force drop")
# ax.set(xlabel='Tip diameter (um)', ylabel='Force drop (nN)')
# # plt.ylim(-0.3,5.4)
# fig.savefig('Results\linregress_diameter_force_drop_control_vel_and_height.pdf', format='pdf')





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
########### Tip Diameter #################################################

### Insertion Force
# force_drop_events_02, force_drop_events_1, force_drop_events_2 = [],[],[]
# for i in range(len(tip_diameter_conc)):
#     height = heights_conc[i] # control: needs to be on a cell, so height >1 um
#     tip_diam = tip_diameter_conc[i]
#     vel = velocity_conc[i] # control 2 um/s
#     sprg = spring_constant_conc[i] # control: not the microfluidic ones, so <0.8 N/m
#     if height > 0.5 and vel == 2 and sprg < 0.5 and insertion_force_conc[i] != 0:
#         if tip_diam == 0.2:
#             force_drop_events_02.append(insertion_force_conc[i])
#         if tip_diam == 1:
#             force_drop_events_1.append(insertion_force_conc[i])
#         if tip_diam == 2:
#             force_drop_events_2.append(insertion_force_conc[i])


# ### Indentation Depth
# force_drop_events_02, force_drop_events_1, force_drop_events_2 = [],[],[]
# for i in range(len(tip_diameter_conc)):
#     height = heights_conc[i] # control: needs to be on a cell, so height >1 um
#     tip_diam = tip_diameter_conc[i]
#     vel = velocity_conc[i] # control 2 um/s
#     sprg = spring_constant_conc[i] # control: not the microfluidic ones, so <0.8 N/m
#     if height > 0.5 and vel == 2 and sprg < 0.5 and indentation_depth_conc[i] != 0:
#         if tip_diam == 0.2:
#             force_drop_events_02.append(indentation_depth_conc[i])
#         if tip_diam == 1:
#             force_drop_events_1.append(indentation_depth_conc[i])
#         if tip_diam == 2:
#             force_drop_events_2.append(indentation_depth_conc[i])


### Force drop
force_drop_events_02, force_drop_events_1, force_drop_events_2 = [],[],[]
for i in range(len(tip_diameter_conc)):
    height = heights_conc[i] # control: needs to be on a cell, so height >1 um
    tip_diam = tip_diameter_conc[i]
    vel = velocity_conc[i] # control 2 um/s
    sprg = spring_constant_conc[i] # control: not the microfluidic ones, so <0.8 N/m
    if height > 0.5 and vel == 2 and sprg < 0.5 and force_drop_conc[i] != 0:
        if tip_diam == 0.2:
            force_drop_events_02.append(force_drop_conc[i])
        if tip_diam == 1:
            force_drop_events_1.append(force_drop_conc[i])
        if tip_diam == 2:
            force_drop_events_2.append(force_drop_conc[i])

data0 = pd.DataFrame({'0.7': force_drop_events_02})
data1 = pd.DataFrame({'1': force_drop_events_1})
data2 = pd.DataFrame({'2': force_drop_events_2})

data_conc = pd.concat([data0, data1, data2], axis=1) 

print(data_conc.head())

# # force drop plot
fig, ax = plt.subplots(figsize=(10,10))
colors_diam_force_drop = sns.xkcd_palette(["spring green","light teal","blue green"])
sns.boxplot(data=data_conc,palette=colors_diam_force_drop, linewidth=4)
# ax.set_title("Relationship between tip diameter and force drop", fontsize=18)
ax.set_xlabel(u'Tip diameter (\u03bcm)', fontsize=20)
ax.set_ylabel('Force drop (nN)', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.ylim(0,2)
fig.savefig('Results\diameter_force_drop_limit.pdf', format='pdf')
plt.close()

# indentation depth plot
# fig, ax = plt.subplots(figsize=(9,10))
# colors_diam_force_drop = sns.xkcd_palette(["spring green","light teal","blue green"])
# sns.boxplot(data=data_conc,palette=colors_diam_force_drop)
# ax.set_title("Relationship between tip diameter and indentation depth", fontsize=18)
# ax.set_xlabel('Tip diameter (um)', fontsize=14)
# ax.set_ylabel('Indentation depth (um)', fontsize=14)
# ax.tick_params(axis='both', which='major', labelsize=15)
# plt.ylim(0,2.5)
# fig.savefig('Results\diameter_indentation_depth_limit.pdf', format='pdf')
# plt.close()

# insertion force plot
# fig, ax = plt.subplots(figsize=(9,10))
# colors_diam_force_drop = sns.xkcd_palette(["spring green","light teal","blue green"])
# sns.boxplot(data=data_conc,palette=colors_diam_force_drop)
# ax.set_title("Relationship between tip diameter and insertion force", fontsize=18)
# ax.set_xlabel('Tip diameter (um)', fontsize=14)
# ax.set_ylabel('Insertion force (nN)', fontsize=14)
# ax.tick_params(axis='both', which='major', labelsize=15)
# plt.ylim(0,8)
# fig.savefig('Results\diameter_insertion_force_limit.pdf', format='pdf')

#### Number of insertion events
events_02, events_1, events_2 = [],[],[]
for i in range(len(tip_diameter_conc)):
    height = heights_conc[i] # control: needs to be on a cell, so height >1 um
    tip_diam = tip_diameter_conc[i]
    vel = velocity_conc[i] # control 2 um/s
    sprg = spring_constant_conc[i] # control: not the microfluidic ones, so <0.8 N/m
    if height > 0.5 and vel == 2 and sprg < 0.5:
        if tip_diam == 0.2:
            events_02.append(insertion_events_present_conc[i])
        if tip_diam == 1:
            events_1.append(insertion_events_present_conc[i])
        if tip_diam == 2:
            events_2.append(insertion_events_present_conc[i])

tip_diameter_bins = ['0.7', '1', '2']
events_diam_list = [events_02, events_1, events_2]
print(len(events_02),len(events_1), len(events_2))    
count0_list, count1_list, count2_list, count3_list, count4_list, count5_list, count_tot_list = countEvents(events_diam_list)

weight_counts_diam = {
    # "0": np.array(count0_list),
    "1": np.array(count1_list),"2": np.array(count2_list),"3": np.array(count3_list),"4": np.array(count4_list),"5": np.array(count5_list)}

fig, ax = plt.subplots()
bottom = np.zeros(3)
colors_diam = [
    # "xkcd:grey", 
    "xkcd:sage", "xkcd:green","xkcd:spring green","xkcd:light teal","xkcd:blue green","xkcd:dark teel"]
for (boolean, weight_count),color in zip(weight_counts_diam.items(), colors_diam):
    p = ax.bar(tip_diameter_bins, weight_count, 0.5, label=boolean, bottom=bottom, color=color, linewidth=4)
    bottom += weight_count
# ax.set_title("Relationship between tip diameter and number of insertion events (%)")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_xlabel(u'Tip diameter (\u03bcm)', fontsize=15)
ax.set_ylabel('Percentage of total insertion events (%)', fontsize=15)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.savefig('Results\diameter_percentage_wozero.pdf', format='pdf')

