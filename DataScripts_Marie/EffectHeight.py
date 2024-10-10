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

# height vs force drop

force_drop_array = []
forcedrop00, forcedrop05, forcedrop1, forcedrop2, forcedrop5, forcedrop10, forcedrop20 = [],[],[],[],[],[],[]
insforce00, insforce05, insforce1, insforce2, insforce5, insforce10, insforce20 = [],[],[],[],[],[],[]
indepth00, indepth05, indepth1, indepth2, indepth5, indepth10, indepth20 = [],[],[],[],[],[],[]
for i in range(len(heights_conc)):
    height = heights_conc[i] 
    force_drop = force_drop_conc[i]
    insertion_force = insertion_force_conc[i]
    indentation_depth = indentation_depth_conc[i]
    tip_diam = tip_diameter_conc[i] # control 2 um
    vel = velocity_conc[i] # control 2 um/s
    sprg = spring_constant_conc[i] # control: not the microfluidic ones, so <0.5 N/m
    if vel == 2 and tip_diam == 2 and sprg < 0.5 and force_drop != 0: # vel == 2 and tip_diam == 2 and
        force_drop_array.append(force_drop)
        if height == 0:
            forcedrop00.append(force_drop)
            insforce00.append(insertion_force)
            indepth00.append(indentation_depth)
        if 0 < height < 1:
            forcedrop05.append(force_drop)
            insforce05.append(insertion_force)
            indepth05.append(indentation_depth)
        if 1 <= height < 2:
            forcedrop1.append(force_drop)
            insforce1.append(insertion_force)
            indepth1.append(indentation_depth)
        if 2 <= height < 3:
            forcedrop2.append(force_drop)
            insforce2.append(insertion_force)
            indepth2.append(indentation_depth)
        if 3 <= height < 4:
            forcedrop5.append(force_drop)
            insforce5.append(insertion_force)
            indepth5.append(indentation_depth)
        if 4 <= height < 5:
            forcedrop10.append(force_drop)
            insforce10.append(insertion_force)
            indepth10.append(indentation_depth)
        if 5 <= height:
            forcedrop20.append(force_drop)
            insforce20.append(insertion_force)
            indepth20.append(indentation_depth)

data0 = indepth00
data1 = indepth05
data2 = indepth1
data3 = indepth2
data4 = indepth5
data5 = indepth10
data6 = indepth20
print(len(data0), len(data1),len(data2),len(data3),len(data4),len(data5),len(data6))

# # summarize
# print([mean(data1), std(data1), percentile(data1, 25), percentile(data1, 50), percentile(data1, 75)])
# print([mean(data2), std(data2), percentile(data2, 25), percentile(data2, 50), percentile(data2, 75)])
# print([mean(data3), std(data3), percentile(data3, 25), percentile(data3, 50), percentile(data3, 75)])
# print([mean(data4), std(data4), percentile(data4, 25), percentile(data4, 50), percentile(data4, 75)])
# print([mean(data5), std(data5), percentile(data5, 25), percentile(data5, 50), percentile(data5, 75)])
# print([mean(data6), std(data6), percentile(data6, 25), percentile(data6, 50), percentile(data6, 75)])

forcedrop_labels0 = ['0'] * len(forcedrop00)
forcedrop_labels1 = ['0-1'] * len(forcedrop05)
forcedrop_labels2 = ['1-2'] * len(forcedrop1)
forcedrop_labels3 = ['2-3'] * len(forcedrop2)
forcedrop_labels4 = ['3-4'] * len(forcedrop5)
forcedrop_labels5 = ['4-5'] * len(forcedrop10)
forcedrop_labels6 = ['>5'] * len(forcedrop20)

insforce_labels0 = ['0'] * len(insforce00)
insforce_labels1 = ['0-1'] * len(insforce05)
insforce_labels2 = ['1-2'] * len(insforce1)
insforce_labels3 = ['2-3'] * len(insforce2)
insforce_labels4 = ['3-4'] * len(insforce5)
insforce_labels5 = ['4-5'] * len(insforce10)
insforce_labels6 = ['>5'] * len(insforce20)

indepth_labels0 = ['0'] * len(indepth00)
indepth_labels1 = ['0-1'] * len(indepth05)
indepth_labels2 = ['1-2'] * len(indepth1)
indepth_labels3 = ['2-3'] * len(indepth2)
indepth_labels4 = ['3-4'] * len(indepth5)
indepth_labels5 = ['4-5'] * len(indepth10)
indepth_labels6 = ['>5'] * len(indepth20)

forcedrop_df = pd.DataFrame({"force drop": forcedrop00 + forcedrop05 + forcedrop1 +forcedrop2 + forcedrop5 + forcedrop10 + forcedrop20,
                   "height": forcedrop_labels0 + forcedrop_labels1 + forcedrop_labels2 + forcedrop_labels3 + forcedrop_labels4 + forcedrop_labels5 + forcedrop_labels6})

insforce_df = pd.DataFrame({"insertion force": insforce00 + insforce05 + insforce1 +insforce2 + insforce5 + insforce10 + insforce20,
                   "height": insforce_labels0 + insforce_labels1 + insforce_labels2 + insforce_labels3 + insforce_labels4 + insforce_labels5 + insforce_labels6})

indepth_df = pd.DataFrame({"indentation depth": indepth00 + indepth05 + indepth1 + indepth2 + indepth5 + indepth10 + indepth20,
                   "height": indepth_labels0 + indepth_labels1 + indepth_labels2 + indepth_labels3 + indepth_labels4 + indepth_labels5 + indepth_labels6})


# colors_diam_force_drop = sns.xkcd_palette(["taupe","dark pink", "pink","lilac", "light blue", "periwinkle", "royal blue"])
# fig, ax = plt.subplots()
# sns.kdeplot(data=forcedrop_df, x="force drop", hue="height", common_norm=False, fill=True, bw_adjust=0.5, clip=(0, None), palette=colors_diam_force_drop)
# plt.title('Distribution of force drop values by height', fontsize=14)
# plt.xlabel('Force drop (nN)', fontsize=14)
# plt.ylabel('Density', fontsize=14)
# fig.savefig('Results\KDE_height_force_drop_control_all.png', dpi=600)

# fig, ax = plt.subplots()
# sns.kdeplot(data=insforce_df, x="insertion force", hue="height", common_norm=False, fill=True, bw_adjust=0.5, clip=(0, None), palette=colors_diam_force_drop)
# plt.title('Distribution of insertion force values by height', fontsize=14)
# plt.xlabel('Insertion force (nN)', fontsize=14)
# plt.ylabel('Density', fontsize=14)
# fig.savefig('Results\KDE_height_insertion_force_control_all.png', dpi=600)

# fig, ax = plt.subplots()
# sns.kdeplot(data=indepth_df, x="indentation depth", hue="height", common_norm=False, fill=True, bw_adjust=0.5, clip=(0, None), palette=colors_diam_force_drop)
# plt.title('Distribution of indentation depth values by height', fontsize=14)
# plt.xlabel('Indentation depth (um)', fontsize=14)
# plt.ylabel('Density', fontsize=14)
# fig.savefig('Results\KDE_height_indentation_depth_control_all.png', dpi=600)

# force_drop_array = []
# height_array = []
# for i in range(len(heights_conc)):
#     height = heights_conc[i]
#     force_drop = force_drop_conc[i]
#     tip_diam = tip_diameter_conc[i] # control 2 um
#     vel = velocity_conc[i] # control 2 um/s
#     sprg = spring_constant_conc[i] # control: not the microfluidic ones, so <0.5 N/m
#     if sprg < 0.5 and force_drop != 0: # tip_diam == 2 and vel == 2 and
#         force_drop_array.append(force_drop)
#         height_array.append(height)

# x = np.array(height_array)
# y = np.array(force_drop_array)
# res = stats.linregress(x, y)
# print(f"R-squared: {res.rvalue**2:.6f}")

# text = 'slope: ' + str(res.slope)
# text2 = f"R-squared: {res.rvalue**2:.6f}"

# fig, ax = plt.subplots()
# plt.plot(x, y, 'o', color="xkcd:periwinkle", label='original data')
# plt.plot(x, res.intercept + res.slope*x, "xkcd:royal blue", label='fitted line')
# plt.legend()
# plt.text(3.4, 2, text)
# plt.text(3.4, 1.5, text2)
# ax.set_title("Relationship between height and force drop")
# ax.set(xlabel='Height (um)', ylabel='Force drop (nN)')
# # plt.ylim(-0.3,5.4)
# fig.savefig('Results\linregress_height_force_drop_control_nocontrols.png', dpi=600)



# pears = stats.pearsonr(x, y)
# print(pears)

# rng = np.random.default_rng()
# method = stats.BootstrapMethod(method='BCa', random_state=rng)
# pears_conf = pears.confidence_interval(confidence_level=0.9, method=method)
# print(pears_conf)

# spear = stats.spearmanr(x, y)
# print(spear)

# dof = len(x)-2  # len(x) == len(y)
# dist = stats.t(df=dof)
# t_vals = np.linspace(-5, 5, 100)
# pdf = dist.pdf(t_vals)
# fig, ax = plt.subplots(figsize=(8, 5))
# def plot(ax):  # we'll reuse this
#     ax.plot(t_vals, pdf)
#     ax.set_title("Spearman's Rho Test Null Distribution")
#     ax.set_xlabel("statistic")
#     ax.set_ylabel("probability density")
# plot(ax)

# rs = spear.statistic  # original statistic
# transformed = rs * np.sqrt(dof / ((rs+1.0)*(1.0-rs)))
# pvalue = dist.cdf(-transformed) + dist.sf(transformed)
# annotation = (f'p-value={pvalue:.4f}\n(shaded area)')
# props = dict(facecolor='black', width=1, headwidth=5, headlength=8)
# _ = ax.annotate(annotation, (2.7, 0.025), (3, 0.03), arrowprops=props)
# i = t_vals >= transformed
# ax.fill_between(t_vals[i], y1=0, y2=pdf[i], color='C0')
# i = t_vals <= -transformed
# ax.fill_between(t_vals[i], y1=0, y2=pdf[i], color='C0')
# ax.set_xlim(-5, 5)
# ax.set_ylim(0, 0.5)

# plt.show()


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
########### Height ######################################################

### Insertion Force
# events_0_05, events_05_1, events_1_15, events_15_2, events_2_25, events_25_3 = [],[],[],[],[],[]
# events_3_35, events_35_4, events_4_45, events_45_5, events_5_55, events_55_6 = [],[],[],[],[],[]
# for i in range(len(heights_conc)):
#     height = heights_conc[i]
#     tip_diam = tip_diameter_conc[i] # control 2 um
#     vel = velocity_conc[i] # control 2 um/s
#     sprg = spring_constant_conc[i] # control: not the microfluidic ones, so <0.8 N/m
#     if tip_diam == 2 and vel == 2 and sprg < 0.5 and insertion_force_conc[i] != 0:
#         if height < 0.5:
#             events_0_05.append(insertion_force_conc[i])
#         if 0.5 <= height < 1:
#             events_05_1.append(insertion_force_conc[i])
#         if 1 <= height < 1.5:
#             events_1_15.append(insertion_force_conc[i])
#         if 1.5 <= height < 2:
#             events_15_2.append(insertion_force_conc[i])
#         if 2 <= height < 2.5:
#             events_2_25.append(insertion_force_conc[i])
#         if 2.5 <= height < 3:
#             events_25_3.append(insertion_force_conc[i])
#         if 3 <= height < 3.5:
#             events_3_35.append(insertion_force_conc[i])
#         if 3.5 <= height < 4:
#             events_35_4.append(insertion_force_conc[i])
#         if 4 <= height < 4.5:
#             events_4_45.append(insertion_force_conc[i])
#         if 4.5 <= height < 5:
#             events_45_5.append(insertion_force_conc[i])
#         if 5 <= height < 5.5:
#             events_5_55.append(insertion_force_conc[i])
#         if 5.5 <= height:
#             events_55_6.append(insertion_force_conc[i])


# #### Indentation Depth
# events_0_05, events_05_1, events_1_15, events_15_2, events_2_25, events_25_3 = [],[],[],[],[],[]
# events_3_35, events_35_4, events_4_45, events_45_5, events_5_55, events_55_6 = [],[],[],[],[],[]
# for i in range(len(heights_conc)):
#     height = heights_conc[i]
#     tip_diam = tip_diameter_conc[i] # control 2 um
#     vel = velocity_conc[i] # control 2 um/s
#     sprg = spring_constant_conc[i] # control: not the microfluidic ones, so <0.8 N/m
#     if tip_diam == 2 and vel == 2 and sprg < 0.5 and indentation_depth_conc[i] != 0:
#         if height < 0.5:
#             events_0_05.append(indentation_depth_conc[i])
#         if 0.5 <= height < 1:
#             events_05_1.append(indentation_depth_conc[i])
#         if 1 <= height < 1.5:
#             events_1_15.append(indentation_depth_conc[i])
#         if 1.5 <= height < 2:
#             events_15_2.append(indentation_depth_conc[i])
#         if 2 <= height < 2.5:
#             events_2_25.append(indentation_depth_conc[i])
#         if 2.5 <= height < 3:
#             events_25_3.append(indentation_depth_conc[i])
#         if 3 <= height < 3.5:
#             events_3_35.append(indentation_depth_conc[i])
#         if 3.5 <= height < 4:
#             events_35_4.append(indentation_depth_conc[i])
#         if 4 <= height < 4.5:
#             events_4_45.append(indentation_depth_conc[i])
#         if 4.5 <= height < 5:
#             events_45_5.append(indentation_depth_conc[i])
#         if 5 <= height < 5.5:
#             events_5_55.append(indentation_depth_conc[i])
#         if 5.5 <= height:
#             events_55_6.append(indentation_depth_conc[i])


#### Force drop
events_0_00, events_0_05, events_05_1, events_1_15, events_15_2, events_2_25, events_25_3 = [],[],[],[],[],[],[]
events_3_35, events_35_4, events_4_45, events_45_5, events_5_55, events_55_6 = [],[],[],[],[],[]
for i in range(len(heights_conc)):
    height = heights_conc[i]
    tip_diam = tip_diameter_conc[i] # control 2 um
    vel = velocity_conc[i] # control 2 um/s
    sprg = spring_constant_conc[i] # control: not the microfluidic ones, so <0.8 N/m
    if tip_diam == 2 and vel == 2 and sprg < 0.5 and force_drop_conc[i] != 0: # tip_diam == 2 and vel == 2 and 
        if height ==0:
            events_0_00.append(force_drop_conc[i])
        if 0 < height < 0.5:
            events_0_05.append(force_drop_conc[i])
        if 0.5 <= height < 1:
            events_05_1.append(force_drop_conc[i])
        if 1 <= height < 1.5:
            events_1_15.append(force_drop_conc[i])
        if 1.5 <= height < 2:
            events_15_2.append(force_drop_conc[i])
        if 2 <= height < 2.5:
            events_2_25.append(force_drop_conc[i])
        if 2.5 <= height < 3:
            events_25_3.append(force_drop_conc[i])
        if 3 <= height < 3.5:
            events_3_35.append(force_drop_conc[i])
        if 3.5 <= height < 4:
            events_35_4.append(force_drop_conc[i])
        if 4 <= height < 4.5:
            events_4_45.append(force_drop_conc[i])
        if 4.5 <= height < 5:
            events_45_5.append(force_drop_conc[i])
        if 5 <= height < 5.5:
            events_5_55.append(force_drop_conc[i])
        if 5.5 <= height:
            events_55_6.append(force_drop_conc[i])

data00 = pd.DataFrame({'0': events_0_00})
data0 = pd.DataFrame({'0-0.5': events_0_05})
data1 = pd.DataFrame({'0.5-1': events_05_1})
data2 = pd.DataFrame({'1-1.5': events_1_15})
data3 = pd.DataFrame({'1.5-2': events_15_2})
data4 = pd.DataFrame({'2-2.5': events_2_25})
data5 = pd.DataFrame({'2.5-3': events_25_3})
data6 = pd.DataFrame({'3-3.5': events_3_35})
data7 = pd.DataFrame({'3.5-4': events_35_4})
data8 = pd.DataFrame({'4-4.5': events_4_45})
data9 = pd.DataFrame({'4.5-5': events_45_5})
data10 = pd.DataFrame({'5-5.5': events_5_55})
data11 = pd.DataFrame({'>5.5': events_55_6})

data_conc = pd.concat([data00, data0, data1, data2, data3, data4, data5, data6, data7, data8, data9, data10], axis=1) 

print('0-0.5', len(events_0_05))
print('0.5-1', len(events_05_1))
print('1-1.5', len(events_1_15))
print('1.5-2', len(events_15_2))
print('2-2.5', len(events_2_25))
print('2.5-3', len(events_25_3))
print('3-3.5', len(events_3_35))
print('3.5-4', len(events_35_4))
print('4-4.5', len(events_4_45))
print('4.5-5', len(events_45_5))
print('5-5.5', len(events_5_55))

# # force drop plot
fig, ax = plt.subplots(figsize=(10,10))
colors_height_force_drop = sns.xkcd_palette(["beige", "taupe", "wine", "dark pink","pink","light pink", "lilac","light blue", "periwinkle","light blue", "bright blue","royal blue"])
sns.boxplot(data=data_conc, palette=colors_height_force_drop, linewidth=4)
# ax.set_title("Relationship between height and force drop", fontsize=18)
ax.set_xlabel(u'Cell height (\u03bcm)', fontsize=20)
ax.set_ylabel('Force drop (nN)', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.ylim(0,1.2)
fig.savefig('Results\height_force_drop_limit.pdf', format='pdf')




# # indentation depth plot
# fig, ax = plt.subplots(figsize=(9,10))
# colors_height_force_drop = sns.xkcd_palette(["beige", "taupe", "wine", "dark pink","pink","light pink", "lilac","periwinkle","light blue", "bright blue","royal blue"])
# sns.boxplot(data=data_conc, palette=colors_height_force_drop)
# ax.set_title("Relationship between height and indentation depth", fontsize=18)
# ax.set_xlabel('Height (um)', fontsize=14)
# ax.set_ylabel('Indentation depth (um)', fontsize=14)
# # plt.ylim(0,1.2)
# fig.savefig('Results\height_indentation_depth.pdf', format='pdf')


# insertion force plot
# fig, ax = plt.subplots(figsize=(9,10))
# colors_height_force_drop = sns.xkcd_palette(["beige", "taupe", "wine", "dark pink","pink","light pink", "lilac","periwinkle","light blue", "bright blue","royal blue"])
# sns.boxplot(data=data_conc, palette=colors_height_force_drop)
# ax.set_title("Relationship between height and insertion force", fontsize=18)
# ax.set_xlabel('Height (um)', fontsize=14)
# ax.set_ylabel('Insertion force (nN)', fontsize=14)
# # plt.ylim(0,6)
# fig.savefig('Results\height_insertion_force.pdf', format='pdf')


#### Number of insertion events
events_0_1, events_1_2, events_2_3, events_3_4, events_4_5, events_5_6 = [],[],[],[],[],[]
for i in range(len(heights_conc)):
    height = heights_conc[i]
    tip_diam = tip_diameter_conc[i] # control 2 um
    vel = velocity_conc[i] # control 2 um/s
    sprg = spring_constant_conc[i] # control: not the microfluidic ones, so <0.8 N/m
    if tip_diam == 2 and vel == 2 and sprg < 0.5:
        if height < 1:
            events_0_1.append(insertion_events_present_conc[i])
        if 1 <= height < 2:
            events_1_2.append(insertion_events_present_conc[i])
        if 2 <= height < 3:
            events_2_3.append(insertion_events_present_conc[i])
        if 3 <= height < 4:
            events_3_4.append(insertion_events_present_conc[i])
        if 4 <= height < 5:
            events_4_5.append(insertion_events_present_conc[i])
        if 5 <= height:
            events_5_6.append(insertion_events_present_conc[i])
     
height_bins = ['0-1','1-2', '2-3', '3-4', '4-5', '>5']
events_height_list = [events_0_1, events_1_2, events_2_3, events_3_4, events_4_5, events_5_6]

count0_list, count1_list, count2_list, count3_list, count4_list, count5_list, count_tot_list = countEvents(events_height_list)

weight_counts_height = {
    # "0": np.array(count0_list),
    "1": np.array(count1_list),"2": np.array(count2_list),"3": np.array(count3_list),"4": np.array(count4_list),"5": np.array(count5_list)}

fig, ax = plt.subplots()
bottom = np.zeros(6)
colors_height = [
    # "xkcd:taupe", 
    "xkcd:dark pink", "xkcd:pink","xkcd:lilac","xkcd:periwinkle","xkcd:bright blue","xkcd:royal blue"]
for (boolean, weight_count),color in zip(weight_counts_height.items(), colors_height):
    p = ax.bar(height_bins, weight_count, 0.5, label=boolean, bottom=bottom, color=color, linewidth=4)
    bottom += weight_count
# ax.set_title("Relationship between height and number of insertion events (%)", fontsize=18)
ax.set_xlabel(u'Cell height (\u03bcm)', fontsize=15)
ax.set_ylabel('Percentage of total insertion events (%)', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.savefig('Results\height_number of insertion events_percentage_wozero.pdf', format='pdf')
