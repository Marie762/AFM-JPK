# -*- coding: utf-8 -*-
"""
Created on Tue Apr 2 15:29:30 2024

@author: marie
"""
import matplotlib.pylab as plt
import numpy as np
import extractJPK
import plot
import procBasic
import metadata

# extract the QI data from all the jpk-qi-data files in the directory 'Data_QI'
#qmap = extractJPK.QI()

# extract the force spectroscopy data from all the jpk-force files in the directory 'Data'
d, F, t = extractJPK.force()
spring_constant_list = metadata.SpringConstant()
F_bS = procBasic.baselineSubtraction(F)

#compute first 10% of data set
i=0
percentage = 10
decimal = percentage/100
length = len(F_bS[i][0])
percentage_of_length = length*decimal
round_percentage_of_length = round(percentage_of_length)

# linear fit of first 10% of dataset
m,b = np.polyfit(d[i][0][:round_percentage_of_length], F_bS[i][0][:round_percentage_of_length], 1)

print(m)
print(b)
x = d[i][0]
lin_fit = m*x + b
fig, ax = plt.subplots()
ax.plot(x, F_bS[i][0], 'deepskyblue')
ax.plot(x, lin_fit, 'orange')
ax.set(xlabel='d', ylabel='force', title='Force-distance curve %i' % i)

plt.show()

# fig, ax = plt.subplots()
# ax.plot(d[7][0], F[7][0], 'purple')
# ax.plot(d[6][0], F[6][0], 'g')
# ax.plot(d[5][0], F[5][0], 'y')
# ax.plot(d[4][0], F_bS[4][0], 'm')
# ax.plot(d[3][0], F_bS[3][0], 'deepskyblue')
# ax.plot(d[2][0], F_bS[2][0], 'orange')
# ax.plot(d[1][0], F_bS[1][0], 'r')
# ax.plot(d[0][0], F_bS[0][0], 'b')
# ax.set(xlabel='height measured (um)', ylabel='force (nN)', title='Force-distance curve %i' % k)

# fig, ax = plt.subplots()
# ax.plot(z, f, 'deepskyblue')
# ax.plot(delta, f, 'orange')
# ax.set(xlabel='z (blue) and delta (orange)', ylabel='force', title='Force-distance curve %i' % k)

# for i in range(len(F_bS)):
#     f = F_bS[i][0]
#     z = d[i][0]
#     k = spring_constant_list[i]
#     deflection = f/k
#     delta = z - deflection
        
    # fig, ax = plt.subplots()
    # ax.plot(t[i][0], f, 'deepskyblue')
    # ax.plot(-d[i][0], f, 'orange')
    # ax.set(xlabel='z (blue) and delta (orange)', ylabel='force', title='Force-distance curve %i' % k)
    
    # fig, ax = plt.subplots()
    # ax.plot(t[i][0], z, 'deepskyblue')
    # ax.plot(t[i][0], deflection, 'red')
    # ax.plot(t[i][0], delta, 'orange')
    # ax.set(xlabel='time', ylabel='z (blue) and delta (orange)', title='time-distance curve %i' % i)
    #fig.savefig('Results\dtt_' + str(i) + '.png')

#plt.show()

