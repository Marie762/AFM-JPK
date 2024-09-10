# -*- coding: utf-8 -*-
"""
Created on Tue Apr 3 2024

@author: marie
"""

import copy
import time
import matplotlib.pylab as plt
import numpy as np
import pickle
from procBasic import baselineSubtraction, heightCorrection
from plot import Fd

def baselineLinearFit(F, d, perc_bottom=0, perc_top=50, plot=False, save=False):
    # two empty lists to store the gradients 'm' and constants 'b' of the linear fit function for each curve
    M = []
    B = []
    for i in range(len(F)):
        slice_bottom = round((perc_bottom/100)*len(F[i][0]))
        slice_top = round((perc_top/100)*len(F[i][0])) #compute first ..% of data set and round to nearest integer
        m,b = np.polyfit(d[i][0][slice_bottom:slice_top], F[i][0][slice_bottom:slice_top], 1) # linear fit of first ..% of dataset
        M.append(m) # store in lists
        B.append(b)
        
        if plot :
            x = d[i][0]
            lin_fit = m*x + b
            fig, ax = plt.subplots()
            ax.plot(d[i][0], F[i][0], 'deepskyblue', label='force-distance curve')
            ax.plot(x, lin_fit, 'orange', label='linear fit line')
            ax.plot(d[i][0][slice_bottom:slice_top], F[i][0][slice_bottom:slice_top], 'red', label='part of curve used in the linear fit')
            ax.set(xlabel='distance (um)', ylabel='force (nN)', title='Force-distance curve %i' % i)
            plt.legend(loc="upper right")
            if save:
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
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

def contactPoint_ruptures(F, D):
    contact_point_fit = []
    
    for i, (f, d) in enumerate(zip(F, D)):
        f_ext, _ = f[0], f[1]
        d_ext, _ = d[0], d[1]
        
        # d_ext is x, f_ext is y
        d_ext = -d_ext

        # Standardize d_ext and f_ext
        d_ext_min = d_ext.min()
        d_ext_max = d_ext.max()
        d_ext = (d_ext - d_ext_min) / (d_ext_max - d_ext_min)
        f_ext = (f_ext - f_ext.min()) / (f_ext.max() - f_ext.min())
        
        # Prepare data for change point detection
        data = np.column_stack([d_ext, f_ext])

        # Apply KernelCPD with RBF kernel for change point detection
        model = "l1"  # "l2", "rbf" normal, cosine, mahalanobis, linear
        # algo = rpt.Pelt(model=model).fit(f_ext)
        # algo = rpt.Window(width=40, model=model).fit(data)
        # algo = rpt.Binseg(model='linear').fit(data)
        algo = rpt.Dynp(model='normal').fit(f_ext)
        result = algo.predict(n_bkps=1)

        # Get the change point
        if result:
            change_point_index = result[0]  # Assuming we are only interested in the first change point
            change_point_x = d_ext[change_point_index]
            
            # De-normalize to get original x value
            change_point_real = -((change_point_x * (d_ext_max - d_ext_min)) + d_ext_min)
            contact_point_fit.append(change_point_real)

            # Plotting the original curve and detected change point
            plt.figure()
            plt.plot(d_ext, f_ext, 'deepskyblue', label='force-distance curve')
            plt.axvline(x=change_point_x, color='red', linestyle='--', label='Detected Change Point')
            plt.xlabel('distance (normalized)')
            plt.ylabel('force (normalized)')
            plt.title(f'Change Point Detection for Curve {i}')
            plt.legend()
            plt.savefig(f'Results\Fd_contact_point_ruptures_{i}.png')
            # plt.show()

            print(f"Change point for dataset {i}: {change_point_real}")
        else:
            print(f"No change point detected for dataset {i}")
            contact_point_fit.append(None)
        
        plt.close()
        
    return contact_point_fit

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
    contact_point_fit = []
    for i,(f,d) in enumerate(zip(F,D)):
        f_ext, _ = f[0], f[1]
        d_ext, _ = d[0], d[1]
        
        # d_ext is x, f_ext is y
        d_ext = -d_ext

        ## standardise d_ext and f_ext
        d_ext_min = d_ext.min()
        d_ext_max = d_ext.max()
        d_ext = (d_ext - d_ext_min) / (d_ext_max - d_ext_min)
        f_ext = (f_ext - f_ext.min()) / (f_ext.max() - f_ext.min())
        

        ## Fit piecewise linear model scipy
        from scipy.optimize import curve_fit

        # def piecewise_linear_power(x, x0, y0, k1, a, b, c):
        #     return np.piecewise(x, [x < x0], 
        #                         [lambda x: k1 * x + y0 - k1 * x0,
        #                         lambda x: a + b*np.exp(x - x0)**c])
        
        def piecewise_linear_polynomial(x, x0, y0, k1, a8, a7, a6, a5, a4, a3, a2, a1, a0):
            return np.piecewise(x, [x < x0], 
                                [lambda x: k1 * x + y0 - k1 * x0,
                                lambda x: a2 * (x - x0)**2 + a1 * (x - x0) + y0])
            
        # def piecewise_linear_polynomial(x, x0, x1, y0, k1, a2, a1, k3):
        #     return np.piecewise(x, [x < x0, (x >= x0) & (x <= x1), x > x1], 
        #                         [lambda x: k1 * x + y0 - k1 * x0, # first linear segment
        #                         lambda x: a2 * (x - x0)**2 + a1 * (x - x0) + y0, # quadratic segment
        #                         lambda x: k3 * (x - x1) + (a2 * (x1 - x0)**2 + a1 * (x1 - x0) + y0)])  # Second linear segment

        
        # def piecewise_linear_polynomial_exponential(x, x0, y0, k1, a0, a1, a2, a3, a4, b, c):
        #     def poly_exp_segment(x, x0, a0, a1, a2, a3, a4, b, c):
        #         # Polynomial + Exponential function
        #         return a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4 + b * np.exp(c * (x - x0))
            
        #     return np.piecewise(x, 
        #                         [x < x0, x >= x0], 
        #                         [lambda x: k1 * x + (y0 - k1 * x0),  # Linear segment
        #                         lambda x: poly_exp_segment(x, x0, a0, a1, a2, a3, a4, b, c) + (y0 - poly_exp_segment(x0, x0, a0, a1, a2, a3, a4, b, c))])  # Polynomial-Exponential segment with offset

        
        # def piecewise_linear_polynomial(x, x0, y0, k1, a1, a0):
        #     def polynomial_segment(x, x0, a1, a0):
        #         # Polynomial of degree 7
        #         return (a1 * (x - x0) + a0)
            
        #     return np.piecewise(x, 
        #                         [x < x0, x >= x0], 
        #                         [lambda x: k1 * x + (y0 - k1 * x0),  # Linear segment
        #                         lambda x: polynomial_segment(x, x0, a1, a0) + y0])  # Polynomial segment
        
        # # Define the piecewise linear-logarithmic function
        # def piecewise_linear_logarithmic_cubic(x, x0, y0, k1, a3, a2, a1, a0):
        #     def log_polynomial_segment(x, x0, a3, a2, a1, a0):
        #         # Polynomial of degree 2 for ln(y)
        #         return a3 * (x - x0)**3 + a2 * (x - x0)**2 + a1 * (x - x0) + a0
            
        #     return np.piecewise(x, 
        #                 [x < x0, x >= x0], 
        #                 [lambda x: k1 * x + (y0 - k1 * x0),  # Linear segment
        #                  lambda x: np.exp(log_polynomial_segment(x, x0, a3, a2, a1, a0)) + (y0 - np.exp(log_polynomial_segment(x0, x0, a3, a2, a1, a0)))])  # Logarithmic segment with offset
        
                
        def fit_piecewise_linear_power(d_ext, f_ext, initial_guesses):
            best_p = None
            best_e = np.inf  # Set initial error to a large value
            
            for guess in initial_guesses:
                try:
                    p, _ = curve_fit(piecewise_linear_polynomial, d_ext, f_ext, p0=guess)
                    residuals = f_ext - piecewise_linear_polynomial(d_ext, *p)
                    ss_res = np.sum(residuals**2)
                    if ss_res < best_e:
                        best_p = p
                        best_e = ss_res
                except RuntimeError:
                    continue
            return best_p

        # Generate initial guesses for the breakpoints
        initial_guesses = []
        num_guesses = 6  # Number of initial guesses
        x0_candidates = np.linspace(d_ext.min(), d_ext.max(), num_guesses)
        #x1_candidates = np.linspace(d_ext.min(), d_ext.max(), num_guesses)

        for x0 in x0_candidates:
            y0_guess = f_ext[np.abs(d_ext - x0).argmin()]  # Estimate y0 based on closest point
            k1_guess = (f_ext[-1] - f_ext[0]) / (d_ext[-1] - d_ext[0])  # Initial slope guess for linear part
            a2_guess = np.random.uniform(-1, 1)  # Stochastic initial guess for 7th-order coefficient
            b2_guess = np.random.uniform(-1, 1)  # Stochastic initial guess for 6th-order coefficient
            d2_guess = np.random.uniform(-1, 1)  # Stochastic initial guess for 5th-order coefficient
            e2_guess = np.random.uniform(-1, 1)  # Stochastic initial guess for 4th-order coefficient
            f2_guess = np.random.uniform(-1, 1)  # Stochastic initial guess for 3rd-order coefficient
            g2_guess = np.random.uniform(-1, 1)  # Stochastic initial guess for 2nd-order coefficient
            h2_guess = np.random.uniform(-1, 1)  # Stochastic initial guess for linear term
            b_shift_guess = np.random.uniform(0.5, 1.0)  # Stochastic shift for the 7th-order term
            c_shift_guess = np.random.uniform(0.5, 1.0)  # Stochastic shift for the 6th-order term
            initial_guesses.append([x0, y0_guess, k1_guess, a2_guess, b2_guess, d2_guess, e2_guess, f2_guess, g2_guess, h2_guess, b_shift_guess, c_shift_guess])
        best_p = fit_piecewise_linear_power(d_ext, f_ext, initial_guesses)
        
        # # Generate initial guesses for the breakpoints
        # initial_guesses = []
        # num_guesses = 6  # Number of initial guesses
        # x0_candidates = np.linspace(d_ext.min(), d_ext.max(), num_guesses)
        # x1_candidates = np.linspace(d_ext.min(), d_ext.max(), num_guesses)

        # for x0 in x0_candidates:
        #     for x1 in x1_candidates:
        #         if x1 > x0:  # Ensure x1 is always greater than x0
        #             y0_guess = f_ext[np.abs(d_ext - x0).argmin()]  # Estimate y0 based on closest point
        #             k1_guess = (f_ext[-1] - f_ext[0]) / (d_ext[-1] - d_ext[0])  # Initial slope guess for linear part
        #             a2_guess = 0  # Initial quadratic coefficient for quadratic part
        #             a1_guess = k1_guess  # Initial linear coefficient for quadratic part
        #             k3_guess = k1_guess # Initial linear coefficient for the second linear part
        #             # Append guesses for the parameters: [x0, x1, y0, k1, a2, b2, k3]
        #             initial_guesses.append([x0, x1, y0_guess, k1_guess, a2_guess, a1_guess, k3_guess])

        # best_p = fit_piecewise_linear_quadratic_linear(d_ext, f_ext, initial_guesses)

        if best_p is not None:
            xd = np.linspace(d_ext.min(), d_ext.max(), 1000)
            plt.plot(d_ext, f_ext, 'deepskyblue', label='force-distance curve')
            plt.plot(xd, piecewise_linear_polynomial(xd, *best_p), label='piecewise linear-powerlaw', color='red')
            plt.scatter([best_p[0]], [piecewise_linear_polynomial(best_p[0], *best_p)], color='green', s=100, zorder=5, label='Change Point')  
            plt.legend(loc="upper right")
            plt.xlabel('distance (um)')
            plt.ylabel('force (nN)')
            plt.title('Force-distance curve %i with Piecewise Linear-Power law Regression' % i)
            plt.savefig('Results\Fd_contact_point_GL14_' + str(i) + '.png')
            plt.close()

            # Print change points
            change_point1 = -((best_p[0]*(d_ext_max - d_ext_min)) + d_ext_min)
            
            print(i, " Change Point:", change_point1)
            contact_point_fit.append(change_point1)
        else:
            print("No valid fit found.")
            contact_point_fit.append(0)

        time.sleep(0.1)
        plt.close()
        
    return contact_point_fit


def contactPoint_evaluation(F, d, contact_point_list):
    k = 1
    date = '2024.07.18'
    data_path = r'StoredValues/' 
    with open(data_path + '/real_contact_point_list_'+ date + '_grid_' + str(k) + '.pkl', "rb") as output_file:
        real_contact_point_list = pickle.load(output_file)
    with open(data_path + '/real_contact_point_height_'+ date + '_grid_' + str(k) + '.pkl', "rb") as output_file:
        real_contact_point_height = pickle.load(output_file)
    
    number_of_points_correct = 0
    error_squared_list = []
    
    lower_bound_list, upper_bound_list = [], []
    
    for i in range(len(contact_point_list)):
        real_height = real_contact_point_height[i]
        estimated_height = d[i][0][contact_point_list[i]]
        
        # calculate upper and lower bounds of real height
        margin = 0.1*real_height # 5% error allowed
        real_height_L = real_height - margin # lower bound
        real_height_U = real_height + margin # upper bound
        
        lower_bound_list.append(real_height_L)
        upper_bound_list.append(real_height_U)
        
        # evaluate estimated height
        if real_height_L <= estimated_height <= real_height_U:
            number_of_points_correct = number_of_points_correct + 1 

        error_squared_list.append(np.abs(real_height - estimated_height))
    
    # percentage of points correct  
    percentage_of_points_correct = (number_of_points_correct/len(contact_point_list))*100
    
    # Root mean square error (RMSE) calculation
    sum_error_squared = np.mean(error_squared_list)
    RMS_error = sum_error_squared
    
    # plot values
    Fd(F, d, real_contact_point_list, contact_point_list, lower_bound_list, upper_bound_list, save=True)
    # print values
    print('Number of points correct: ', number_of_points_correct)
    print('Percentage of points correct: %.2f ' % percentage_of_points_correct)
    print('MAD: ', RMS_error)
        
    return number_of_points_correct, percentage_of_points_correct, RMS_error

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
        
        argmin_val = [i for i,el in enumerate(deviation_list[:argmax_val]) if abs(el) < 0.0001]
        if len(argmin_val) != 0:
                argmin_val = argmin_val[-1]
        else:
            argmin_val = [i for i,el in enumerate(deviation_list[:argmax_val]) if abs(el) < 0.001]
            if len(argmin_val) != 0:
                argmin_val = argmin_val[-1]
            else:
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
