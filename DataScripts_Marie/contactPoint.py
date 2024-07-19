# -*- coding: utf-8 -*-
"""
Created on Tue Apr 3 2024

@author: marie
"""

import copy
import time
import matplotlib.pylab as plt
import numpy as np
from procBasic import baselineSubtraction, heightCorrection

def baselineLinearFit(F, d, perc_bottom=0, perc_top=50, plot=False, saveplot=False):
    # two empty lists to store the gradients 'm' and constants 'b' of the linear fit function for each curve
    M = []
    B = []
    for i in range(len(F)):
        slice_bottom = round((perc_bottom/100)*len(F[i][0]))
        slice_top = round((perc_top/100)*len(F[i][0])) #compute first ..% of data set and round to nearest integer
        m,b = np.polyfit(d[i][0][slice_bottom:slice_top], F[i][0][slice_bottom:slice_top], 1) # linear fit of first ..% of dataset
        M.append(m) # store in lists
        B.append(b)
        
        if plot:
            x = d[i][0]
            lin_fit = m*x + b
            fig, ax = plt.subplots()
            ax.plot(d[i][0], F[i][0], 'deepskyblue', label='force-distance curve')
            ax.plot(x, lin_fit, 'orange', label='linear fit line')
            ax.plot(d[i][0][slice_bottom:slice_top], F[i][0][slice_bottom:slice_top], 'red', label='part of curve used in the linear fit')
            ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
            plt.legend(loc="upper right")
            if saveplot:
                fig.savefig('Results\Fd_baseline_linearfit_' + str(i) + '.png')
    
    return M, B

def contactPoint1(F, d, plot='False', saveplot='False', perc_bottom=0, perc_top=50):
    F_bS = baselineSubtraction(F)
    M, B = baselineLinearFit(F_bS, d, perc_bottom=perc_bottom, perc_top=perc_top)
    # empty list to store the index of the last intersection point of the F-d curve with the linear fit line 
    contact_point_list = []
    for i in range(len(F_bS)):
        difference_list = []
        for j in range(len(F_bS[i][0])):
            f = M[i]*(d[i][0][j]) + B[i] # linear fit line
            difference_squared = (F_bS[i][0][j] - f)**2 # the difference-swuared between the force value and the value of the linear fit line at each point
            difference_list.append(difference_squared)

        # argmin_val = [i for i,el in enumerate(difference_list) if abs(el) < 0.0001]
        # if len(argmin_val) != 0:
        #     argmin_val = argmin_val[-1]
        # else:
        argmin_val = [i for i,el in enumerate(difference_list) if abs(el) < 0.001]
        if len(argmin_val) != 0:
            argmin_val = argmin_val[-1]
        else:
            argmin_val = [i for i,el in enumerate(difference_list) if abs(el) < 0.01]
            if len(argmin_val) != 0:
                argmin_val = argmin_val[-1]
            else:
                argmin_val = [i for i,el in enumerate(difference_list) if abs(el) < 0.1][-1]
        
        contact_point_list.append(argmin_val)

        if plot :
            fig, ax = plt.subplots()
            ax.plot(d[i][0], F_bS[i][0], 'deepskyblue', label='force-distance extend curve')
            ax.plot(d[i][1], F_bS[i][1], 'skyblue', label='force-distance retract curve')
            ax.plot(d[i][0], M[i]*(d[i][0]) + B[i], 'orange', label='linear fit line')
            ax.plot(d[i][0][argmin_val], F_bS[i][0][argmin_val], 'ro', label='contact point estimation 1')
            ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
            plt.legend(loc="upper right")
            if saveplot :
                fig.savefig('Results\Fd_contact_point_' + str(i) + '.png')
            plt.close()
    
    return contact_point_list

def contactPoint2(F, d, plot='False', saveplot='False'):
    F_bS = baselineSubtraction(F)
    contact_point_list = []
    perc_bottom = 80
    for i in range(len(F_bS)):
        slice_bottom = round((perc_bottom/100)*len(F_bS[i][0]))
        argmin_val = np.argmin(F_bS[i][0][slice_bottom:])
        argmin_val = argmin_val + slice_bottom
        contact_point_list.append(argmin_val)

        if plot :
            fig, ax = plt.subplots()
            ax.plot(d[i][0], F_bS[i][0], 'deepskyblue', label='force-distance curve')
            ax.plot(d[i][0][argmin_val], F_bS[i][0][argmin_val], 'ro', label='contact point estimation 2')
            ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
            plt.legend(loc="upper right")
            plt.show()
            if saveplot :
                fig.savefig('Results\Fd_contact_point_' + str(i) + '.png')
    
    return contact_point_list

def contactPoint_derivative(F, D):


    for i,(f,d) in enumerate(zip(F,D)):
        f_ext, _ = f[0], f[1]
        d_ext, _ = d[0], d[1]
        

        import torch
        # d_ext is x, f_ext is y
        d_ext = -d_ext

        ## tandardise d_ext and f_ext
        d_ext = (d_ext - d_ext.min()) / (d_ext.max() - d_ext.min())
        f_ext = (f_ext - f_ext.min()) / (f_ext.max() - f_ext.min())

        x = torch.tensor(d_ext, requires_grad=True, dtype=torch.float32)
        y = torch.tensor(f_ext, requires_grad=True, dtype=torch.float32)


        loss_fn = torch.nn.L1Loss()
        learning_rate = 1e-1
        num_restarts = 15

        best_loss = float('inf')
        best_model_state = None
        overall_best_loss = float('inf')
        overall_best_model_state = None

        for i in range(num_restarts):
            model = torch.nn.Sequential(
                torch.nn.Linear(1, 1),
                torch.nn.PReLU(),
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=350, gamma=0.1) 

            best_loss = float('inf')
            best_model_state = None
        
            for _ in range(3000):
                shuffled_indices = torch.randperm(len(x))
                shuffled_indices = shuffled_indices[:len(x)//10]
                x_shuffled = x[shuffled_indices]
                y_shuffled = y[shuffled_indices]
                y_pred = model(x_shuffled[:, None])
                loss = loss_fn(y_pred, y_shuffled[:, None])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                validation_loss = loss_fn(model(x[:, None]), y[:, None])
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    best_model_state = copy.deepcopy(model.state_dict())


            if best_loss < overall_best_loss:
                overall_best_loss = best_loss
                overall_best_model_state = best_model_state
        
            print(f"Progress: {i+1}/{num_restarts}, best loss: {overall_best_loss:.3f}", end='\r')
            
        model.load_state_dict(overall_best_model_state)
        print(f"Overall best loss after {num_restarts} restarts: {overall_best_loss:.3f}")
        print(model[0].weight, model[0].bias)
        plt.plot(d_ext, f_ext, label='data')
        plt.plot(d_ext, model(x[:, None]).detach().numpy(), label='model')
        plt.legend()
        plt.show()

        time.sleep(2)
        plt.close()

def contactPoint3(F, d, plot = False, save = False, perc_bottom=0, perc_top=50, multiple=4, multiple1=3, multiple2=2): ## TODO: Turn lists of arrays into arrays
    M, B = baselineLinearFit(F, d, perc_bottom=perc_bottom, perc_top=perc_top)
    # empty list to store the index of the last intersection point of the F-d curve with the linear fit line 
    contact_point_list = []
    standard_deviation_list = []
    argmax_store = 0
    for i in range(len(F)):
        deviation_list = []
        slice_bottom = round((perc_bottom/100)*len(F[i][0]))
        slice_top = round((perc_top/100)*len(F[i][0]))
        # calculate standard deviation ## TODO: Vectorize this
        for j in range(len(F[i][0])):
            f = M[i]*(d[i][0][j]) + B[i] # linear fit line
            deviation_squared = (F[i][0][j] - f)**2 # the difference-squared between the force value and the value of the linear fit line at each point
            deviation_list.append(deviation_squared)
        standard_deviation = np.sqrt(np.sum(deviation_list[slice_bottom:slice_top])/len(F[i][0][slice_bottom:slice_top]))
        standard_deviation_list.append(standard_deviation)
        
        argmax_val = [i for i,el in enumerate(deviation_list[slice_bottom:]) if abs(np.sqrt(el)) > multiple*standard_deviation]
        if len(argmax_val) != 0:
                argmax_val = argmax_val[0] + slice_bottom
                m = multiple
        else:
            argmax_val = [i for i,el in enumerate(deviation_list[slice_bottom:]) if abs(np.sqrt(el)) > multiple1*standard_deviation]
            if len(argmax_val) != 0:
                argmax_val = argmax_val[0] + slice_bottom
                m = multiple1
            else:
                argmax_val = [i for i,el in enumerate(deviation_list[slice_bottom:]) if abs(np.sqrt(el)) > multiple2*standard_deviation]
                if len(argmax_val) != 0:
                    argmax_val = argmax_val[0] + slice_bottom
                    m = multiple2
                else:
                    argmax_val = argmax_store
        
        argmax_store = argmax_val
        
        # argmin_val = [i for i,el in enumerate(deviation_list[:argmax_val]) if abs(el) < 0.0001]
        # if len(argmin_val) != 0:
        #         argmin_val = argmin_val[-1]
        # else:
        # argmin_val = [i for i,el in enumerate(deviation_list[:argmax_val]) if abs(el) < 0.001]
        # if len(argmin_val) != 0:
        #     argmin_val = argmin_val[-1]
        # else:
        argmin_val = [i for i,el in enumerate(deviation_list[:argmax_val]) if abs(el) < 0.01]
        if len(argmin_val) != 0:
            argmin_val = argmin_val[-1]
        else:
            argmin_val = argmax_val
            print(i, argmax_val)
                
        contact_point_list.append(argmin_val)

        print(f"Force-distance curve {i}: contact point estimation 3: {argmin_val} x standard deviation: {m}", end='\r')
        if plot :
            fig, ax = plt.subplots()
            ax.plot(d[i][0], F[i][0], 'deepskyblue', label='force-distance curve')
            ax.plot(d[i][0][slice_bottom:slice_top], F[i][0][slice_bottom:slice_top], 'm', label='percentage of curve used')
            ax.plot(d[i][0], M[i]*(d[i][0]) + B[i], 'orange', label='linear fit line')
            ax.plot(d[i][0][argmin_val], F[i][0][argmin_val], 'ro', label='contact point estimation 3')
            ax.plot(d[i][0][argmax_val], F[i][0][argmax_val], 'go', label= '%i x standard deviation' % m)
            ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
            plt.legend(loc="upper right")
            if save :
                fig.savefig('Results\Fd_contact_point_' + str(i) + '.png')
            plt.close()
    return contact_point_list

def QIcontactPoint1(F, d, perc_bottom=0, perc_top=50):
    contact_point_height = []
    for m in range(len(F)):
        contact_point_height_cols = []
        contact_point_list = contactPoint1(F[m],d[m],perc_bottom=perc_bottom, perc_top=perc_top)
        
        for n in range(len(F[m])):
            contact_point_height_cols.append(d[m][n][0][contact_point_list[n]])
        
        contact_point_height.append(contact_point_height_cols)
    
    return contact_point_height

def QIcontactPoint2(F,d):
    contact_point_height = []
    for m in range(len(F)):
        contact_point_height_cols = []
        F_bS = baselineSubtraction(F[m])
        d_hC = heightCorrection(d[m])
        contact_point_list = contactPoint2(F_bS,d_hC)
        
        for n in range(len(F[m])):
            contact_point_height_cols.append(d[m][n][0][contact_point_list[n]])
        
        contact_point_height.append(contact_point_height_cols)
    
    return contact_point_height

def QIcontactPoint3(F, d, perc_bottom=0, perc_top=50):
    contact_point_height = []
    for m in range(len(F)):
        contact_point_height_cols = []
        F_bS = baselineSubtraction(F[m])
        d_hC = heightCorrection(d[m])
        contact_point_list = contactPoint3(F_bS,d_hC,perc_bottom=perc_bottom, 
                                           perc_top=perc_top) # , multiple=35, multiple1=25, multiple2=3
        
        for n in range(len(F[m])):
            contact_point_height_cols.append(d[m][n][0][contact_point_list[n]])
        
        contact_point_height.append(contact_point_height_cols)
    
    return contact_point_height
