# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:57:50 2024

@author: marie
"""

import matplotlib.pylab as plt
from matplotlib.ticker import ScalarFormatter
from scipy.signal import savgol_filter
import numpy as np
import afmformats

#%% Plotting:
	
# for k in range(len(allfilesinfolder) - 1):
#  	fig, ax = plt.subplots()
#  	ax.plot(d[k], F[k])

#  	ax.set(xlabel='height measured (um)', ylabel='force (nN)',
#         title='Force-distance curve %i' % k)

#  	fig.savefig('Force curves on nucleus vijay\Fd_' + str(k) + '.png')
#  	#plt.show()

#%%
# to find the maximum force point, which is ideally the set-point force

# find maximum value in "force" array 
k = 49

max_value = max(F[k])
max_element = np.argmax(F[k])

# plots a red point at the max force value
fig, ax = plt.subplots()
ax.plot(d[k], F[k])
ax.plot(d[k][max_element], F[k][max_element], 'ro')

ax.set(xlabel='height measured (um)', ylabel='force (nN)', title='Force-distance curve %i' % k)

#%%
# background substraction

k = 49

o = min(F[k])
F_corr = F[k] - o

fig, ax = plt.subplots()
ax.plot(d[k], F[k])
ax.plot(d[k], F_corr, 'r')

ax.set(xlabel='height measured (um)', ylabel='force (nN)', title='Force-distance curve '+ str(k) + ': baseline offset correction')

#fig.savefig('Fd_baseline_' + str(k) + '.png')

#%%
# smoothing function rolling average
# k = 49

# # pandas dataframe
# F_pd = pd.DataFrame(F[k])

# # Calculate moving average with a window size of 5
# moving_avg = F_pd.rolling(window=100).mean()

# fig, ax = plt.subplots()
# ax.plot(d[k], F[k])
# ax.plot(d[k], moving_avg, 'r')

# ax.set(xlabel='height measured (um)', ylabel='force (nN)', title='Force-distance curve '+ str(k) + ': with moving average smoothing with window 100')

# fig.savefig('Force curves on nucleus vijay\Fd_smoothed_' + str(k) + '.png')

#%%
# smoothing Savitzky-Golay filter
k = 49
window_size, poly_order = 500, 3

smoothed_data = savgol_filter(F_corr, window_size, poly_order)

fig, ax = plt.subplots()
ax.plot(t[k], F_corr)
ax.plot(t[k], smoothed_data, 'r')

ax.set(xlabel='time (s)', ylabel='force (nN)', title='Force-distance curve '+ str(k) + ': with Savitzky-Golay smoothing with window ' + str(window_size) + 'and polynomial order ' + str(poly_order))

#%%
# plotting derivative 

#dFdd = []

# for i in range(len(F[k])-1):
#  	#dF = (F[k][i+1] - F[k][i]) / (d[k][i+1] - d[k][i])
#  	dF = F[k][i+1] - F[k][i]
#  	dd = d[k][i+1] - d[k][i]
#  	dFdd.append(dF/dd)
 	
# dFdd.append(dFdd[len(F[k])-2])

dFdt = []

for i in range(len(F[k])-1):
	dF = smoothed_data[i+1] - smoothed_data[i]
	dt = t[k][i+1] - t[k][i]
	dFdt.append(dF/dt)

dFdt.append(dFdt[len(F[k])-2])


# fig, ax = plt.subplots()
# ax.plot(d[k], F_corr, 'r')
# ax.plot(d[k], dFdd, 'g')

# ax.set(xlabel='height measured (um)', ylabel='force (nN)', title='Force-distance curve %i' % k)

fig, ax = plt.subplots()
ax.plot(t[k], dFdt, 'g')

ax.set(xlabel='time (s)', ylabel='dF/dt', title='Derivative of smoothed force-distance curve '+ str(k))
#fig.savefig('Force curves on nucleus vijay\Fd_derivative_' + str(k) + '.png')



plt.show()
#%%
# find derivative  for moving avg smoothed function

# k = 49

# dFdd = []
# # for i in range(len(F[k])-1):
# # 	#dF = (F[k][i+1] - F[k][i]) / (d[k][i+1] - d[k][i])
# # 	dF = F[k][i+1] - F[k][i]
# # 	dd = d[k][i+1] - d[k][i]
# # 	dFdd.append(dF/dd)
# # 	
# # dFdd.append(dFdd[len(F[k])-2])
# naam_van_mijn_vrouw = "Marie"
# print(f"Beeldschone code hoor {naam_van_mijn_vrouw} \n -Gert")

# # fig, ax = plt.subplots()
# # ax.plot(d[k], F_corr, 'r')
# # ax.plot(d[k], dFdd, 'g')

# # ax.set(xlabel='height measured (um)', ylabel='force (nN)', title='Force-distance curve %i' % k)


# for i in range(len(F[k])-1):
#  	#dF = (F[k][i+1] - F[k][i]) / (d[k][i+1] - d[k][i])
#  	dF = moving_avg.loc[i+1].iat[0] - moving_avg.loc[i].iat[0] # .loc[i].iat[0]
#  	dd = d[k][i+1] - d[k][i]
#  	dFdd.append(dF/dd)
 	
# dFdd.append(dFdd[len(moving_avg)-2])

# fig, ax = plt.subplots()
# ax.plot(d[k], F[k])
# ax.plot(d[k], moving_avg, 'r')
# ax.plot(d[k], dFdd, 'g')

# ax.set(xlabel='height measured (um)', ylabel='force (nN)', title='Force-distance curve '+ str(k) + ': with derivative')
# #fig.savefig('Force curves on nucleus vijay\Fd_derivative_' + str(k) + '.png')



# plt.show()

