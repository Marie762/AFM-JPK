# -*- coding: utf-8 -*-
"""
Created on Tue Apr 3 2024

@author: marie
"""

import matplotlib.pylab as plt
import pandas as pd

def Fd(F, d, save=False):
    for k in range(len(F)):
        fig, ax = plt.subplots()
        ax.plot(d[k][0], F[k][0], 'deepskyblue')
        ax.plot(d[k][1], F[k][1], 'orange')
        if len(F[k]) > 2:
            ax.plot(d[k][2], F[k][2], 'mediumorchid')
        ax.set(xlabel='height measured (um)', ylabel='force (nN)', title='Force-distance curve %i' % k)
        if save:
            fig.savefig('Results\Fd_' + str(k) + '.png')
    return fig

def Ft(F, t, save='False'):
    for k in range(len(F)):
        fig, ax = plt.subplots()
        ax.plot(t[k][0], F[k][0], 'deepskyblue')
        ax.plot(t[k][1], F[k][1], 'orange')
        if len(F[k]) > 2:
            ax.plot(t[k][2], F[k][2], 'mediumorchid')
        ax.set(xlabel='time (s)', ylabel='force (nN)', title='Force-time curve %i' % k)
        if save == 'True':
            fig.savefig('Results\Ft_' + str(k) + '.png')
    return fig

def Fdsubplot(F, d, F_sub, colour1='blue', colour2='orangered', colour3='indigo', subplot_name='subplot', save='False'):
    for k in range(len(F)):
        fig, ax = plt.subplots()
        ax.plot(d[k][0], F[k][0], 'deepskyblue')
        ax.plot(d[k][1], F[k][1], 'orange')
        ax.plot(d[k][0], F_sub[k][0], colour1)
        ax.plot(d[k][1], F_sub[k][1], colour2)
        if len(F[k]) > 2:
            ax.plot(d[k][2], F[k][2], 'mediumorchid')
            ax.plot(d[k][2], F_sub[k][2], colour3)
        ax.set(xlabel='height measured (um)', ylabel='force (nN)', title='Force-distance curve ' + str(k) + ' with ' + subplot_name)
        if save == 'True':
            fig.savefig('Results\Ft_' + subplot_name + '_' + str(k) + '.png')
    return fig

def Ftsubplot(F, t, F_sub, colour1='blue', colour2='orangered', colour3='indigo', subplot_name='subplot', save='False'):
    for k in range(len(F)):
        fig, ax = plt.subplots()
        ax.plot(t[k][0], F[k][0], 'deepskyblue')
        ax.plot(t[k][1], F[k][1], 'orange')
        ax.plot(t[k][0], F_sub[k][0], colour1)
        ax.plot(t[k][1], F_sub[k][1], colour2)
        if len(F[k]) > 2:
            ax.plot(t[k][2], F[k][2], 'mediumorchid')
            ax.plot(t[k][2], F_sub[k][2], colour3)
        ax.set(xlabel='time (s)', ylabel='force (nN)', title='Force-time curve ' + str(k) + ' with ' + subplot_name)
        if save == 'True':
            fig.savefig('Results\Ft_' + subplot_name + '_' + str(k) + '.png')
    return fig

def QIMap(data, ind, col, k, save='False', name = '_'):
    dataframe_qmap = pd.DataFrame(data=data, index=ind, columns=col)
    fig, ax = plt.subplots()
    # ax = sns.heatmap(dataframe_qmap)
    im = ax.imshow(dataframe_qmap, origin='lower', extent=(col[0], col[-1], ind[0], ind[-1]), interpolation='gaussian', cmap='Blues_r')  #vmin = 8.75, vmax = 10.5, interpolation='gaussian'
    fig.colorbar(im, ax=ax, label='Height (um)')
    ax.set(xlabel='x (um)', ylabel='y (um)', title='QI map ' + name + ' ' + str(k))
    if save == 'True':
        fig.savefig('Results\QIMap_' + name + str(k) + '.png')
    return fig


def FdGrid_Height(data, x_position, y_position, k, save=False, name = '_', interpolation='gaussian'):
    dataframe_qmap = pd.DataFrame(data=data, index=x_position, columns=y_position)
    fig, ax = plt.subplots()
    im = ax.imshow(dataframe_qmap, origin='lower', extent=(y_position[0], y_position[-1], x_position[0], x_position[-1]), 
                   cmap='Blues_r', interpolation=interpolation, vmin = 0, vmax =5)  #, vmin = 0, vmax = 5, vmin = 8.75, vmax = 10.5, interpolation='gaussian', cmap='Blues_r' or 'hot_r'
    fig.colorbar(im, ax=ax, label='Height (um)') # height map
    ax.set(xlabel='x (um)', ylabel='y (um)', title='Fd grid ' + name + ' ' + str(k))
    if save:
        fig.savefig('Results\AFdGrid_' + name + str(k) + '.png')
    return fig

def FdGrid_Peaks(data, x_position, y_position, k, save=False, name = '_', interpolation='gaussian'):
    dataframe_qmap = pd.DataFrame(data=data, index=x_position, columns=y_position)
    fig, ax = plt.subplots()
    im = ax.imshow(dataframe_qmap, origin='lower', extent=(y_position[0], y_position[-1], x_position[0], x_position[-1]), 
                   cmap='hot_r', interpolation=interpolation, vmin = 0, vmax = 5) #, vmin = 8.75, vmax = 10.5, interpolation='gaussian', cmap='Blues_r' or 'hot_r'
    fig.colorbar(im, ax=ax, label='Number of peaks') # number of peaks map
    ax.set(xlabel='x (um)', ylabel='y (um)', title='Fd grid ' + name + ' ' + str(k))
    if save:
        fig.savefig('Results\AFdGrid_' + name + str(k) + '_p1.png')
    return fig

def FdGrid_Indentation(data, x_position, y_position, k, save=False, name = '_', interpolation='gaussian'):
    dataframe_qmap = pd.DataFrame(data=data, index=x_position, columns=y_position)
    fig, ax = plt.subplots()
    im = ax.imshow(dataframe_qmap, origin='lower', extent=(y_position[0], y_position[-1], x_position[0], x_position[-1]), 
                   cmap='hot_r', interpolation=interpolation)  #, vmin = 0, vmax = 5, vmin = 8.75, vmax = 10.5, interpolation='gaussian', cmap='Blues_r' or 'hot_r'
    fig.colorbar(im, ax=ax, label='Indentation depth (um)') # indentation depth map
    ax.set(xlabel='x (um)', ylabel='y (um)', title='Fd grid ' + name + ' ' + str(k))
    if save:
        fig.savefig('Results\AFdGrid_' + name + str(k) + '_p1.png')
    return fig

def FdGrid_Emodulus(data, x_position, y_position, k, save=False, name = '_', interpolation='gaussian'):
    dataframe_qmap = pd.DataFrame(data=data, index=x_position, columns=y_position)
    fig, ax = plt.subplots()
    im = ax.imshow(dataframe_qmap, origin='lower', extent=(y_position[0], y_position[-1], x_position[0], x_position[-1]), 
                   cmap='Purples', vmin = 0, vmax = 20)  #, vmin = 0, vmax = 5, vmin = 8.75, vmax = 10.5, interpolation='gaussian', cmap='Blues_r' or 'hot_r'
    fig.colorbar(im, ax=ax, label='Youngs modulus (kPa)') # Youngs modulus map
    ax.set(xlabel='x (um)', ylabel='y (um)', title='Fd grid ' + name + ' ' + str(k))
    if save:
        fig.savefig('Results\AFdGrid_' + name + str(k) + '.png')
    return fig