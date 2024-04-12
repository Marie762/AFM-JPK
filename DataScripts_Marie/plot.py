# -*- coding: utf-8 -*-
"""
Created on Tue Apr 3 2024

@author: marie
"""

import matplotlib.pylab as plt

def Fd(F, d, save='False'):
    for k in range(len(F)):
        fig, ax = plt.subplots()
        ax.plot(d[k][0], F[k][0])
        ax.plot(d[k][1], F[k][1], 'y')
        ax.plot(d[k][2], F[k][2], 'g')
        ax.set(xlabel='height measured (um)', ylabel='force (nN)', title='Force-distance curve %i' % k)
        if save == 'True':
            fig.savefig('Results\Fd_' + str(k) + '.png')
    return fig

def Ft(F, t, save='False'):
    for k in range(len(F)):
        fig, ax = plt.subplots()
        ax.plot(t[k][0], F[k][0])
        ax.plot(t[k][1], F[k][1], 'y')
        ax.plot(t[k][2], F[k][2], 'g')
        ax.set(xlabel='time (s)', ylabel='force (nN)', title='Force-time curve %i' % k)
        if save == 'True':
            fig.savefig('Results\Ft_' + str(k) + '.png')
    return fig

def Fdsubplot(F, d, F_sub, colour1='r', colour2='c', colour3='m', subplot_name='subplot', save='False'):
    for k in range(len(F)):
        fig, ax = plt.subplots()
        ax.plot(d[k][0], F[k][0])
        ax.plot(d[k][1], F[k][1], 'y')
        ax.plot(d[k][2], F[k][2], 'g')
        ax.plot(d[k][0], F_sub[k][0], colour1)
        ax.plot(d[k][1], F_sub[k][1], colour2)
        ax.plot(d[k][2], F_sub[k][2], colour3)
        ax.set(xlabel='height measured (um)', ylabel='force (nN)', title='Force-distance curve ' + str(k) + ' with ' + subplot_name)
        if save == 'True':
            fig.savefig('Results\Ft_' + subplot_name + '_' + str(k) + '.png')
    return fig

def Ftsubplot(F, t, F_sub, colour1='r', colour2='c', colour3='m', subplot_name='subplot', save='False'):
    for k in range(len(F)):
        fig, ax = plt.subplots()
        ax.plot(t[k][0], F[k][0])
        ax.plot(t[k][1], F[k][1], 'y')
        ax.plot(t[k][2], F[k][2], 'g')
        ax.plot(t[k][0], F_sub[k][0], colour1)
        ax.plot(t[k][1], F_sub[k][1], colour2)
        ax.plot(t[k][2], F_sub[k][2], colour3)
        ax.set(xlabel='time (s)', ylabel='force (nN)', title='Force-time curve ' + str(k) + ' with ' + subplot_name)
        if save == 'True':
            fig.savefig('Results\Ft_' + subplot_name + '_' + str(k) + '.png')
    return fig

