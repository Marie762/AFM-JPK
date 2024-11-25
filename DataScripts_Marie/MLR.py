# -*- coding: utf-8 -*-
"""
Created on Tues Oct 15 2024

@author: marie
"""

import os
import sys
import numpy as np
import pickle
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import statsmodels.formula.api as sm
from statsmodels.regression.linear_model import OLS
from scipy.stats import chi2
from itertools import chain, combinations

##### Load the created data frame which includes cell experiments data
one_large_data_frame = pd.read_csv('Results_metadata\one_large_data_frame_edited.csv') # FD != 0
# one_large_data_frame = pd.read_csv('Results_metadata\one_large_data_frame.csv') # all data

# create a Boolean mask to replace all tipdiameters 0.2 to 0.7
mask = one_large_data_frame['tip_diameter'] == 0.2
# replace values based on the mask
one_large_data_frame.loc[mask, 'tip_diameter'] = 0.7
# one_large_data_frame = one_large_data_frame[one_large_data_frame['spring_constant'] < 0.5]


# The goal is to check whether the independent variables: height (H), tip_diameter (D), insertion_velocity (V), and/or spring_stiffness (S),
# have an effect on the variable: force_drop (FD)
H = one_large_data_frame['height'].to_numpy()
D = one_large_data_frame['tip_diameter'].to_numpy()
V = one_large_data_frame['insertion_velocity'].to_numpy()
S = one_large_data_frame['spring_constant'].to_numpy()
E = one_large_data_frame['E_modulus'].to_numpy()
FD = one_large_data_frame['force_drop'].to_numpy()
IF = one_large_data_frame['insertion_force'].to_numpy()
ID = one_large_data_frame['indentation_depth'].to_numpy()
# N = one_large_data_frame['number_of_peaks'].to_numpy()

# 1. Create new interaction variables
H_D = H*D
H_V = H*V
H_S = H*S

D_S = D*S
V_S = V*S


H_D_S = H*D*S
H_V_S = H*V*S



# Create a data frame with all data
# data = pd.DataFrame({'H': H,'D': D,'V': V,'S': S,
#                      'HxD': H_D,'HxV': H_V,'HxS': H_S,'DxS': D_S,'VxS': V_S,
#                      'HxDxS': H_D_S,'HxVxS': H_V_S})

data = pd.DataFrame({'H': H,'D': D,'V': V,'S': S,
                     'HxD': H_D,'HxV': H_V,'HxS': H_S,'DxS': D_S,'VxS': V_S})

# data = pd.DataFrame({'H': H,'D': D,'V': V,'S': S,
#                      'H_D_S': H_D_S,'H_V_S': H_V_S,
#                      'FD': FD})

# data = pd.DataFrame({'H': H,'D': D,'V': V,'S': S,
#                      'D_S': D_S,'V_S': V_S})

# data = pd.DataFrame({'H': H,'D': D,'V': V,'S': S,
#                      'H_D': H_D,'H_V': H_V,'H_S': H_S,'FD': FD}) #,'FD': FD,'IF': IF,'ID': ID

# data = pd.DataFrame({'H': H,'D': D,'V': V,'S': S}) # 'k (N/m)': S.round(decimals=3), 'Velocity': V, 'Diameter': D

#standardize the values in each column
# data = (data-data.mean())/data.std()
# target = pd.DataFrame({'FD': FD, 'IF': IF, 'ID': ID})
# target = pd.DataFrame({'FD': FD})
# target = pd.DataFrame({'IF': IF})
# target = pd.DataFrame({'ID': ID})
target = pd.DataFrame({'E': E, 'FD': FD, 'IF': IF, 'ID': ID})
data = pd.concat([data,target],axis=1)
# data.head().to_csv('Results_metadata\Variables.csv', encoding='utf-8')
# data.head().to_csv('Results_metadata\Variables_standardised.csv', encoding='utf-8')

min_val = data.min(0)
print(min_val)
sys.exit()
# 2. Look at descriptive statistics for all data
# data.describe().to_csv('Results_metadata\Descriptive_stats.csv', encoding='utf-8')
# print(data.describe())

# 3. Look at scatter plots for each variable

# sns.set_theme()
# sns.set_context("notebook",font_scale=1.5) # ,font_scale=2
# pp = sns.pairplot(data, diag_kind=None, markers='o', plot_kws={'color':'k'},height=2, aspect=1, corner=True) # "husl", "Set2", "crest", "YlOrBr","flare", "Spectral"
# pp.tick_params(axis='both', which='major',labelsize=15)
# pp.savefig('Results\pairplot_incE.png', dpi=600)
# plt.show()

# x='D'
# xlabel = u'Tip diameter (\u03bcm)' # u'Cell height (\u03bcm) x Insertion velocity (\u03bcm/s)'
# xlabel = u'Tip diameter (\u03bcm)' # u'Cell height (\u03bcm) x Insertion velocity (\u03bcm/s)'
# xlabel = u'Insertion velocity (\u03bcm/s)' # u'Cell height (\u03bcm) x Insertion velocity (\u03bcm/s)'
# xlabel = 'Spring constant (N/m)' # u'Cell height (\u03bcm) x Insertion velocity (\u03bcm/s)'
# xlabel = 'E modulus (kPa)'
# palette = sns.color_palette("Spectral", 11) # S: "Spectral", 11; D: "YlOrBr", 3; V: "Set2", 6
# hue = 'k (N/m)'

# sns.set_theme() # height vs. youngs modulus
# sc = sns.scatterplot(data, x=x, y='S',color='r') #, hue=hue, palette=palette
# # sns.move_legend(sc, loc="upper left", bbox_to_anchor=(1, 1)) 
# plt.xlabel(xlabel,fontsize=15)
# plt.ylabel('Spring constant (N/m)',fontsize=15) 
# plt.subplots_adjust(right=0.8, bottom=0.15)
# plt.tick_params(axis='both', which='major',labelsize=15)
# plt.grid(True)
# fig = sc.get_figure()
# fig.savefig('Results\scatterplot_D_S.png', dpi=600)
# plt.close()

# Stiffness just changes the rate of increase in force load
# Compare 0.01 to 0.1 to 1 N/m
# In range: 0-10 N and 0-5 microns
x_arr = np.arange(0,1000,1)
y_0_01 = 0.01*x_arr
y_0_05 = 0.05*x_arr
y_0_1 = 0.1*x_arr
y_0_5 = 0.5*x_arr
y_1 = x_arr

plt.plot(x_arr, y_0_01, color='r', label='0.01 N/m')
plt.plot(x_arr, y_0_05, color='m', label='0.05 N/m')
plt.plot(x_arr, y_0_1, color='g', label='0.1 N/m')
plt.plot(x_arr, y_0_5, color='c', label='0.5 N/m')
plt.plot(x_arr, y_1, color='b', label='1 N/m')
plt.legend(loc='lower right',fontsize=12)
plt.xlabel('Cantilever deflection (nm)',fontsize=15)
plt.ylabel('Force (nN)',fontsize=15) 
plt.ylim(0,10)
plt.subplots_adjust(right=0.8, bottom=0.15)
plt.tick_params(axis='both', which='major',labelsize=15)
plt.grid(True)

plt.savefig('Results\spring_constant_effect.png', dpi=600)
plt.close()


sys.exit()

# sns.set_theme()
# sc = sns.scatterplot(data, x=x, y='IF') #, hue=hue, palette=palette
# # sns.move_legend(sc, loc="upper left", bbox_to_anchor=(1, 1)) 
# plt.xlabel(xlabel,fontsize=15)
# plt.ylabel('Insertion force (nN)',fontsize=15) 
# plt.subplots_adjust(right=0.8, bottom=0.15)
# plt.tick_params(axis='both', which='major',labelsize=15)
# plt.grid(True)
# fig = sc.get_figure()
# fig.savefig('Results\scatterplot_E_IF.png', dpi=600)
# plt.close()


# sns.set_theme()
# sc = sns.scatterplot(data, x=x, y='FD', hue=hue, palette=palette) 
# sns.move_legend(sc, loc="upper left", bbox_to_anchor=(1, 1)) 
# plt.xlabel(xlabel,fontsize=15)
# plt.ylabel('Force drop (nN)',fontsize=15) 
# plt.subplots_adjust(right=0.8, bottom=0.15)
# plt.tick_params(axis='both', which='major',labelsize=15)
# plt.grid(True)
# fig = sc.get_figure()
# fig.savefig('Results\scatterplot_H_FD_no_microfluidics.png', dpi=600)
# plt.close()

# sns.set_theme()
# sc = sns.scatterplot(data, x=x, y='IF', hue=hue, palette=palette) 
# sns.move_legend(sc, loc="upper left", bbox_to_anchor=(1, 1)) 
# plt.xlabel(xlabel, fontsize=15)
# plt.ylabel('Insertion force (nN)',fontsize=15)
# plt.subplots_adjust(right=0.8, bottom=0.15)
# plt.tick_params(axis='both', which='major',labelsize=15)
# plt.grid(True)
# fig = sc.get_figure()
# fig.savefig('Results\scatterplot_H_IF_no_microfluidics.png', dpi=600)
# plt.close()

# sns.set_theme()
# sc = sns.scatterplot(data, x=x, y='ID', hue=hue, palette=palette) 
# sns.move_legend(sc, loc="upper left", bbox_to_anchor=(1, 1)) 
# plt.xlabel(xlabel, fontsize=15)
# plt.ylabel(u'Indentation depth (\u03bcm)',fontsize=15)
# plt.subplots_adjust(right=0.8, bottom=0.15)
# plt.tick_params(axis='both', which='major',labelsize=15)
# plt.grid(True)
# fig = sc.get_figure()
# fig.savefig('Results\scatterplot_H_ID_no_microfluidics.png', dpi=600)
# plt.close()

# 4. Calculate a correlation matrix for all variables.
# correlation_matrix = data.corr()
# # correlation_matrix = target.corr()
# correlation_matrix.to_csv('Results_metadata\correlation_matrix.csv', encoding='utf-8')
# print(correlation_matrix)

# plt.figure(figsize=(6, 3)) # figsize=(10, 5)
# heatmap = sns.heatmap(target.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
# fig = heatmap.get_figure()
# fig.savefig('Results\heatmap_correlaticn_matrix_E_outcomes.png', dpi=600)

# 5. Calculate a simple linear regression for each variable (not the interactions).

# formula_large = 'ID ~ H + D + HxS + DxS'
# mod_large = OLS.from_formula(formula_large, data=data)
# res_large = mod_large.fit()
# with open('Results_metadata\ML_ID_mymodel.csv', 'w') as fh:
#     fh.write(res_large.summary().as_text())
# print(res_large.summary())


def powerset(iterable):
    s = list(iterable)
    [",".join(map(str, comb)) for comb in combinations(s, 3)]
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

# formulas = powerset(['H','D','V','S','HxD','HxV','HxS','DxS','VxS']) # ,'HxDxS','HxVxS'
formulas = powerset(['H','D','S','HxD','HxS','DxS']) # removing V

AIC_list = [] 
BIC_list = []
formula_list = []                    
for subset in formulas:
    if len(subset) > 0:
        joined_subset = '+'.join(subset)
        formula = 'ID ~' + joined_subset
        formula_list.append(formula)
        mod = OLS.from_formula(formula, data=data)
        res = mod.fit()
        AIC_list.append(res.aic)
        BIC_list.append(res.bic)



index_list = np.arange(len(BIC_list))
print('min BIC', min(BIC_list))
min_index = BIC_list.index(min(BIC_list))
formuls = [q for q,el in enumerate(BIC_list) if el < 3680]
for i in range(len(formuls)):
    formula1 = formula_list[formuls[i]]
    print(formula1)
    # mod1 = OLS.from_formula(formula1, data=data)
    # res1 = mod1.fit()
    # print(res1.summary())
    # print( )

formula1 = formula_list[min_index]

# for subset,i in zip(formulas,index_list):
#     if len(subset) > 0:
#         if i == min_index:
#             joined_subset = '+'.join(subset)
#             formula = 'FD ~' + joined_subset
#             print(formula)


sc = plt.scatter(index_list,BIC_list,c='k',marker='.')
plt.scatter(index_list[formuls[0]],BIC_list[formuls[0]],c='m',marker='.',label=formula_list[formuls[0]])
plt.scatter(index_list[min_index],BIC_list[min_index],c='r',marker='.',label=formula_list[min_index])
plt.scatter(index_list[formuls[2]],BIC_list[formuls[2]],c='g',marker='.',label=formula_list[formuls[2]])
plt.xlabel('Formula index',fontsize=15)
plt.ylabel('BIC value',fontsize=15) 
plt.subplots_adjust(left=0.15, bottom=0.15)
plt.tick_params(axis='both', which='major',labelsize=15)
plt.grid(True)
plt.legend(loc='upper right',fontsize=12)
# plt.show()
fig = sc.get_figure()
fig.savefig('Results\scatterplot_BIC_ID_reduced.png', dpi=600)
plt.close()
sys.exit()


# formula1 = 'ID ~ H'
# formula2 = 'ID ~ D'
# formula3 = 'ID ~ V'
# formula4 = 'ID ~ S'
# formula1 = 'FD ~ H + D + HxD + HxS'
mod1 = OLS.from_formula(formula1, data=data)
res1 = mod1.fit()
with open('Results_metadata\ML_BIC_IF_reduced.csv', 'w') as fh:
    fh.write(res1.summary().as_text())
print(res1.summary())
sys.exit()

# mod2 = OLS.from_formula(formula2, data=data)
# res2 = mod2.fit()
# with open('Results_metadata\LM2_D_ID.csv', 'w') as fh:
#     fh.write(res2.summary().as_text())

# mod3 = OLS.from_formula(formula3, data=data)
# res3 = mod3.fit()
# with open('Results_metadata\LM3_V_ID.csv', 'w') as fh:
#     fh.write(res3.summary().as_text())

# mod4 = OLS.from_formula(formula4, data=data)
# res4 = mod4.fit()
# with open('Results_metadata\LM4_S_ID.csv', 'w') as fh:
#     fh.write(res4.summary().as_text())

# 6. Calculate a multiple linear regression for all variables, without interactions
# full model:
formula5 = 'ID ~ H + D + V + S'
mod5 = OLS.from_formula(formula5, data=data)
res5 = mod5.fit()
# with open('Results_metadata\LM_ID_interactions_HxS_DxS.csv', 'w') as fh:
#     fh.write(res5.summary().as_text())
res5_ll = res5.llf #calculate log-likelihood of model
print(res5_ll)

# reduced model:
formula5_r = 'ID ~ H + D'
mod5_r = OLS.from_formula(formula5_r, data=data)
res5_r = mod5_r.fit()
with open('Results_metadata\LM_ID_multiple_HVS.csv', 'w') as fh:
    fh.write(res5_r.summary().as_text())
res5_r_ll = res5_r.llf #calculate log-likelihood of model
print(res5_r_ll)

#calculate likelihood ratio Chi-Squared test statistic
LR_statistic = -2*(res5_r_ll-res5_ll)
print(LR_statistic)

#calculate p-value of test statistic using 2 degrees of freedom
p_val = chi2.sf(LR_statistic, 2)
print(p_val)


sys.exit()
# Let's see what happens if you drop out D or V or H, these have a cross correlation of 0.435
formula6 = 'FD ~ H + D + S' # dropping out V
mod6 = OLS.from_formula(formula6, data=data)
res6 = mod6.fit()
with open('Results_metadata\LM6_H_D_S_FD.csv', 'w') as fh:
    fh.write(res6.summary().as_text())

formula7 = 'FD ~ H + V + S' # dropping out D
mod7 = OLS.from_formula(formula7, data=data)
res7 = mod7.fit()
with open('Results_metadata\LM7_H_V_S_FD.csv', 'w') as fh:
    fh.write(res7.summary().as_text())
    
formula9 = 'FD ~ D + S' # dropping out H
mod9 = OLS.from_formula(formula9, data=data)
res9 = mod9.fit()
with open('Results_metadata\LM10_D_S_FD.csv', 'w') as fh:
    fh.write(res9.summary().as_text())

# Let's see what happens if you drop out both D and V
formula8 = 'FD ~ H + S' # dropping out both D and V
mod8 = OLS.from_formula(formula8, data=data)
res8 = mod8.fit()
with open('Results_metadata\LM8_H_S_FD.csv', 'w') as fh:
    fh.write(res8.summary().as_text())
    
# 7. Add in various interactions, to see what happens.
formula6_int1 = 'FD ~ D + S + D_S'
mod6_1 = OLS.from_formula(formula6_int1, data=data)
res6_1 = mod6_1.fit()
with open('Results_metadata\LM10_D_S_FD_Int_DS.csv', 'w') as fh:
    fh.write(res6_1.summary().as_text())
    
# formula6_int2 = 'FD ~ H + V + S + V_S'
# mod6_2 = OLS.from_formula(formula6_int2, data=data)
# res6_2 = mod6_2.fit()
# with open('Results_metadata\LM7_H_V_S_FD_Int_VS.csv', 'w') as fh:
#     fh.write(res6_2.summary().as_text())
    
# formula6_int3 = 'FD ~ H + V + S + H_V + V_S'
# mod6_3 = OLS.from_formula(formula6_int3, data=data)
# res6_3 = mod6_3.fit()
# with open('Results_metadata\LM7_H_V_S_FD_Int_HV_VS.csv', 'w') as fh:
#     fh.write(res6_3.summary().as_text())
    
# formula6_int4 = 'FD ~ H + V + S + H_V_S'
# mod6_4 = OLS.from_formula(formula6_int4, data=data)
# res6_4 = mod6_4.fit()
# with open('Results_metadata\LM7_H_V_S_FD_Int_HVS.csv', 'w') as fh:
#     fh.write(res6_4.summary().as_text())
    
# formula6_int5 = 'FD ~ H + V + S + H_V + V_S + H_V_S'
# mod6_5 = OLS.from_formula(formula6_int5, data=data)
# res6_5 = mod6_5.fit()
# with open('Results_metadata\LM7_H_V_S_FD_Int_HV_VS_HVS.csv', 'w') as fh:
#     fh.write(res6_5.summary().as_text())