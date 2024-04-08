# -*- coding: utf-8 -*-
"""
Created on Tue Apr 3 2024

@author: marie
"""

import matplotlib.pylab as plt

def Fd(F, F_approach, d_approach, F_inter, d_inter, F_retract, d_retract, save='False'):
    for k in range(len(F)):
        fig, ax = plt.subplots()
        ax.plot(d_approach[k], F_approach[k])
        ax.plot(d_inter[k], F_inter[k], 'y')
        ax.plot(d_retract[k], F_retract[k], 'g')
        ax.set(xlabel='height measured (um)', ylabel='force (nN)', title='Force-distance curve %i' % k)
        if save == 'True':
            fig.savefig('Results\Fd_' + str(k) + '.png')
    return fig

def Ft(F, F_approach, t_approach, F_inter, t_inter, F_retract, t_retract, save='False'):
    for k in range(len(F)):
        fig, ax = plt.subplots()
        ax.plot(t_approach[k], F_approach[k])
        ax.plot(t_inter[k], F_inter[k], 'y')
        ax.plot(t_retract[k], F_retract[k], 'g')
        ax.set(xlabel='time (s)', ylabel='force (nN)', title='Force-time curve %i' % k)
        if save == 'True':
            fig.savefig('Results\Ft_' + str(k) + '.png')
    return fig

def Fdsubplot(F, d, F_sub, colour='r', subplot_name='subplot', save='False'):
    for k in range(len(F)):
        fig, ax = plt.subplots()
        ax.plot(d[k], F[k])
        ax.plot(d[k], F_sub[k], colour)
        ax.set(xlabel='height measured (um)', ylabel='force (nN)', title='Force-distance curve ' + str(k) + ' with ' + subplot_name)
        if save == 'True':
            fig.savefig('Results\Ft_' + subplot_name + '_' + str(k) + '.png')
    return fig

def Ftsubplot(F, t, F_sub, colour='r', subplot_name='subplot', save='False'):
    for k in range(len(F)):
        fig, ax = plt.subplots()
        ax.plot(t[k], F[k])
        ax.plot(t[k], F_sub[k], colour)
        ax.set(xlabel='time (s)', ylabel='force (nN)', title='Force-time curve ' + str(k) + ' with ' + subplot_name)
        if save == 'True':
            fig.savefig('Results\Ft_' + subplot_name + '_' + str(k) + '.png')
    return fig

