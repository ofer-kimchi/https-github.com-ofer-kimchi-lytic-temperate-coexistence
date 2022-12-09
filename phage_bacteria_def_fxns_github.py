#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 13:04:03 2022

@author: Ofer
"""
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import scipy.integrate
import seaborn as sns
from scipy.fft import fft, fftfreq


def avg_value(x, y):
    if len(x.shape) == 2:
        # x is matrix of size num_s x num_t
        # y is a vector of size num_t
        y_zeroed = y - min(y)  # which is also t_vec[0]
        
        x_rescaled = (y_zeroed[1: ] - y_zeroed[: -1]) * (
            (x[:, 1:] + x[:, :-1]) / 2)  # the avg x over this time window
        return(np.sum(x_rescaled, 1) / y_zeroed[-1])
    if len(x.shape) == 3:
        # x is matrix of size num_s1 x num_s2 x num_t
        # y is a vector of size num_t
        y_zeroed = y - min(y)  # which is also t_vec[0]
        
        x_rescaled = (y_zeroed[1: ] - y_zeroed[: -1]) * (
            (x[:, :, 1:] + x[:, :, :-1]) / 2)  # the avg x over this time window
        return(np.sum(x_rescaled, 2) / y_zeroed[-1])
        

def flatten(l): #From StackExchange
    #Flatten a list -- given a list of sublists, concatenate the sublists into one long list
    return([item for sublist in l for item in sublist])


def moving_average(arr, window):
    return np.convolve(arr, np.ones(window), 'same') / window  
    # same leads to boundary effects but same length


def plot_fft(max_time, curves_to_plot, labels, dt=1, plot_every=1, plot_start_index=0,
             avg_window=0, xlog=True, ylog=True, opacity=1):
    # Needs to get inputs from Euler method -- constant timestep
    
    max_time -= plot_start_index * dt
    num_timesteps = int(max_time / dt)
    xf = fftfreq(num_timesteps, dt)[:num_timesteps//2]
    
    for xlogscale in [xlog]:
        for ylogscale in [ylog]:
            if xlogscale:
                sample = np.unique(np.logspace(0, np.log10(num_timesteps/2 - 1),  # we use xf[1:][sample] since xf[0] == 0
                                     int((num_timesteps/2-1)/plot_every), 
                                     endpoint=False, dtype=int))
                sample = np.concatenate(((0,), sample))
            else:
                sample = np.linspace(0, num_timesteps/2 - 1,  # we use xf[1:][sample] since xf[0] == 0
                                     int((num_timesteps/2-1)/plot_every), 
                                     endpoint=False, dtype=int)
            plt.figure()
            for e, curve in enumerate(curves_to_plot):
                yf = 2.0/num_timesteps * np.abs(fft(
                    curve[plot_start_index:])[0:num_timesteps//2])
                if avg_window > 0:
                    yf_sample = []
                    yf = moving_average(yf, avg_window)
                    for e2 in range(len(sample) - 1):
                        yf_sample += [np.mean(yf[1:][sample[e2]: sample[e2 + 1]])]
                    yf_sample = np.array(yf_sample + [yf[-1]])
                else:
                    yf_sample = yf[1:][sample]
                plt.plot(xf[1:][sample], yf_sample,
                         linestyle='-', label=labels[e])
                
            #plt.plot([1/25, 1/25], [1e-8, 1e-2], ':k')
            plt.xlabel('Frequency')
            plt.ylabel('Fourier transform')
            if xlogscale:
                plt.xscale('log')
            if ylogscale:
                plt.yscale('log')
            plt.legend(bbox_to_anchor=(1.05, 1))
            plt.show()
            

# =============================================================================
# Define simulation
# =============================================================================
def get_diff_eqs(params):
    """
    alpha = growth rate; beta = burst size; delta = death rate; lam = induction rate; 
    r = branching ratio (r1 is branching ratio for single lysogens; r2 for double); 
    r3 is the ostensible branching ratio for triple lysogens, but we set the growth rate of those to zero.
        The result is that a phage that infects a double lysogen sensitive to it dies with probability r3_j (and undergoes lysis otherwise)
    c = rate of prophage -> cryptic prophage (c1 is rate for single lysogens; c2 for double); 
    K = carrying capacity
    
    b_i is a vector of length N_b
    p_j is a vector of length N_p
    l_ij is a matrix of size N_b x N_p
    l_ij1j2 is a tensor of size N_b x N_p x N_p
    """
    
    
    (alpha_b, delta_b, beta, delta_p, alpha_l, delta_l, alpha_l2, delta_l2, lam_ijk, 
     c1, c2, K, r1_jk, r2_jk, r3_jk, gamma_ijk, phage_mask, lysogen_mask, double_lysogen_mask, 
     phage_die_when_infecting_lysogen, triple_lysogens_displace_double, 
     only_phage_die_with_multiple_lys, log_space) = params

    # If triple_lysogens_displace_double==True, when a double lysogen is infected and 
    # is meant to turn lysogenic, it will displace either of the current lysogens with 50% probability each
    # (And I verified that this indeed works correctly)
    
    l_2l_survive = only_phage_die_with_multiple_lys * (alpha_l2 == 0) # do lysogens that are tried to turn into double lysogens survive?
    l2_3l_survive = only_phage_die_with_multiple_lys # do double lysogens that are tried to turn into triple lysogens survive?

    def log_space_modify_derivative(num, den, log_space=log_space):
        if not log_space:  # if we're just writing the diff-eqs in linear space, no change
            return(num)
        # To convert df/dt to dlog(f)/dt, we need to divide by f. 
        # But if f is zero, we don't want to do that (since df/dt will also be 0, and we just want the result to be 0)
        
        den_mask = den > 0  #np.where(den > 0)[0]
        num[den_mask] /= den[den_mask]
        return(num)
        
    def log_space_modify_vars(ml_b_i, ml_p_jk, ml_l_ijk, ml_l_ij1k1j2k2, log_space=log_space):
        # ml_ stands for "maybe log"
        if not log_space:
            return(ml_b_i, ml_p_jk, ml_l_ijk, ml_l_ij1k1j2k2)
        exp_log_b_i = np.exp(ml_b_i)
        exp_log_p_jk = np.exp(ml_p_jk)
        exp_log_l_ijk = np.exp(ml_l_ijk)
        exp_log_l_ij1k1j2k2 = np.exp(ml_l_ij1k1j2k2)
        return(exp_log_b_i, exp_log_p_jk, exp_log_l_ijk, exp_log_l_ij1k1j2k2)
    

    def carrying_capacity(b_i, l_ijk, l_ij1k1j2k2):
        return((1 - (np.sum(b_i) + np.sum(l_ijk) + np.sum(l_ij1k1j2k2)) / K))
    
    def dB_dt(ml_b_i, ml_p_jk, ml_l_ijk, ml_l_ij1k1j2k2):
        b_i, p_jk, l_ijk, l_ij1k1j2k2 = log_space_modify_vars(
            ml_b_i, ml_p_jk, ml_l_ijk, ml_l_ij1k1j2k2)
        return(log_space_modify_derivative(
            alpha_b * b_i * carrying_capacity(b_i, l_ijk, l_ij1k1j2k2)
            - np.sum(gamma_ijk * np.expand_dims(p_jk, 0), (-1, -2)) * b_i 
            - delta_b * b_i 
            + c1 * np.sum(l_ijk, (1, 2)),
            b_i))
    
    def dB_dt_Kinf(ml_b_i, ml_p_jk, ml_l_ijk, ml_l_ij1k1j2k2):  # dB/dt if K is set to infinity
        b_i, p_jk, l_ijk, l_ij1k1j2k2 = log_space_modify_vars(
            ml_b_i, ml_p_jk, ml_l_ijk, ml_l_ij1k1j2k2)
        return(log_space_modify_derivative(
            alpha_b * b_i 
            - np.sum(gamma_ijk * p_jk, (-1, -2)) * b_i 
            - delta_b * b_i 
            + c1 * np.sum(l_ijk, (1, 2)),
            b_i))
    
    def dP_dt(ml_b_i, ml_p_jk, ml_l_ijk, ml_l_ij1k1j2k2):
        b_i, p_jk, l_ijk, l_ij1k1j2k2 = log_space_modify_vars(
            ml_b_i, ml_p_jk, ml_l_ijk, ml_l_ij1k1j2k2)
        return(log_space_modify_derivative(
            phage_mask * (
                (beta * (1 - r1_jk) - r1_jk) * p_jk * np.sum(np.expand_dims(b_i, (1, 2)) * gamma_ijk, 0) +
                (beta * (1 - r2_jk) - r2_jk) * p_jk * np.sum(np.expand_dims(np.sum(l_ijk, (1, 2)), (1, 2)) * gamma_ijk, 0) +
                (beta * (1 - r3_jk) - r3_jk) * p_jk * np.sum(np.expand_dims(np.sum(l_ij1k1j2k2, (1, 2, 3, 4)), (1, 2)) * gamma_ijk, 0) +
                - delta_p * p_jk +
                beta * np.sum(lam_ijk * (l_ijk + np.sum(l_ij1k1j2k2, (-2, -1)) + np.sum(l_ij1k1j2k2, (1, 2))), 0) + 
                - (beta * (1 - r2_jk) - r2_jk + phage_die_when_infecting_lysogen) * p_jk * np.sum(
                    gamma_ijk * np.expand_dims(np.sum(l_ijk, -1), 2), 0) + 
                - (beta * (1 - r3_jk) - r3_jk + phage_die_when_infecting_lysogen) * p_jk * np.sum(
                    gamma_ijk * np.expand_dims((np.sum(l_ij1k1j2k2, (2, 3, 4)) + np.sum(l_ij1k1j2k2, (1, 2, 4))), 2), 0)),
            p_jk))

    def dL_dt(ml_b_i, ml_p_jk, ml_l_ijk, ml_l_ij1k1j2k2):
        b_i, p_jk, l_ijk, l_ij1k1j2k2 = log_space_modify_vars(
            ml_b_i, ml_p_jk, ml_l_ijk, ml_l_ij1k1j2k2)
        return(log_space_modify_derivative(
            lysogen_mask * (alpha_l * l_ijk * carrying_capacity(b_i, l_ijk, l_ij1k1j2k2)
            + gamma_ijk * np.expand_dims(b_i, (1, 2)) * np.expand_dims(p_jk * r1_jk, 0)
            - np.expand_dims(np.sum(gamma_ijk * np.expand_dims(p_jk * (1 - r2_jk)**l_2l_survive, 0), (1, 2)), (1, 2)) * l_ijk 
            + np.expand_dims(np.sum(gamma_ijk * np.expand_dims(p_jk * (1 - r2_jk)**l_2l_survive, 0), -1), 2) * l_ijk  
            # prev line corrects line before: lysogens are resistant to corresponding phage
            - (c1 + lam_ijk + delta_l) * l_ijk
            + c2 * (np.sum(l_ij1k1j2k2, (-2, -1)) + np.sum(l_ij1k1j2k2, (1, 2)))),
            l_ijk))

    
    def dL_dt_Kinf(ml_b_i, ml_p_jk, ml_l_ijk, ml_l_ij1k1j2k2):  # dL/dt if K is set to infinity
        b_i, p_jk, l_ijk, l_ij1k1j2k2 = log_space_modify_vars(
            ml_b_i, ml_p_jk, ml_l_ijk, ml_l_ij1k1j2k2)
        return(log_space_modify_derivative(
            lysogen_mask * (alpha_l * l_ijk 
            + gamma_ijk * np.expand_dims(b_i, (1, 2)) * np.expand_dims(p_jk * r1_jk, 0)
            - np.expand_dims(np.sum(gamma_ijk * np.expand_dims(p_jk * (1 - r2_jk)**l_2l_survive, 0), (1, 2)), (1, 2)) * l_ijk 
            + np.expand_dims(np.sum(gamma_ijk * np.expand_dims(p_jk * (1 - r2_jk)**l_2l_survive, 0), -1), 2) * l_ijk  
            # prev line corrects line before: lysogens are resistant to corresponding phage
            - (c1 + lam_ijk + delta_l) * l_ijk
            + c2 * (np.sum(l_ij1k1j2k2, (-2, -1)) + np.sum(l_ij1k1j2k2, (1, 2)))),
            l_ijk))


    def dL2_dt(ml_b_i, ml_p_jk, ml_l_ijk, ml_l_ij1k1j2k2):
        b_i, p_jk, l_ijk, l_ij1k1j2k2 = log_space_modify_vars(
            ml_b_i, ml_p_jk, ml_l_ijk, ml_l_ij1k1j2k2)
        return(log_space_modify_derivative(
            double_lysogen_mask * (
                alpha_l2 * l_ij1k1j2k2 * carrying_capacity(b_i, l_ijk, l_ij1k1j2k2)
                + (np.expand_dims(gamma_ijk * np.expand_dims(p_jk * r2_jk, 0), (3, 4)) * np.expand_dims(l_ijk, (1, 2)) +
                   np.expand_dims(gamma_ijk * np.expand_dims(p_jk * r2_jk, 0), (1, 2)) * np.expand_dims(l_ijk, (3, 4)))
                - np.expand_dims(np.sum(gamma_ijk * np.expand_dims(p_jk * (1 - r3_jk)**l2_3l_survive, 0), (1, 2)), (1, 2, 3, 4)) * l_ij1k1j2k2
                + (np.expand_dims(np.sum(gamma_ijk * np.expand_dims(p_jk * (1 - r3_jk)**l2_3l_survive, 0), -1), (2, 3, 4)) + 
                   np.expand_dims(np.sum(gamma_ijk * np.expand_dims(p_jk * (1 - r3_jk)**l2_3l_survive, 0), -1), (1, 2, 4))) * l_ij1k1j2k2
                - (2 * c1 + np.expand_dims(lam_ijk, (3, 4)) + np.expand_dims(lam_ijk, (1, 2)) + delta_l2) * l_ij1k1j2k2
                + 0.5 * triple_lysogens_displace_double * (
                    np.expand_dims(gamma_ijk * np.expand_dims(p_jk * r3_jk, 0), (3, 4)) * (
                        np.expand_dims(np.sum(l_ij1k1j2k2, (3, 4)) + np.sum(l_ij1k1j2k2, (1, 2)), (1, 2)) - np.expand_dims(np.sum(l_ij1k1j2k2, 2), 2))
                    + np.expand_dims(gamma_ijk * np.expand_dims(p_jk * r3_jk, 0), (1, 2)) * (
                        np.expand_dims(np.sum(l_ij1k1j2k2, (3, 4)) + np.sum(l_ij1k1j2k2, (1, 2)), (3, 4)) - np.expand_dims(np.sum(l_ij1k1j2k2, 4), 4))
                )), 
            l_ij1k1j2k2))
    # I checked that the last two lines indeed simply redistribute the double lysogens,
    # and don't create or destroy anything (when combined with the lines where the phage kill the double lysogens)

    def dL2_dt_Kinf(ml_b_i, ml_p_jk, ml_l_ijk, ml_l_ij1k1j2k2):
        b_i, p_jk, l_ijk, l_ij1k1j2k2 = log_space_modify_vars(
            ml_b_i, ml_p_jk, ml_l_ijk, ml_l_ij1k1j2k2)
        return(log_space_modify_derivative(
            double_lysogen_mask * (
                alpha_l2 * l_ij1k1j2k2
                + (np.expand_dims(gamma_ijk * np.expand_dims(p_jk * r2_jk, 0), (3, 4)) * np.expand_dims(l_ijk, (1, 2)) +
                   np.expand_dims(gamma_ijk * np.expand_dims(p_jk * r2_jk, 0), (1, 2)) * np.expand_dims(l_ijk, (3, 4)))
                - np.expand_dims(np.sum(gamma_ijk * np.expand_dims(p_jk * (1 - r3_jk)**l2_3l_survive, 0), (1, 2)), (1, 2, 3, 4)) * l_ij1k1j2k2
                + (np.expand_dims(np.sum(gamma_ijk * np.expand_dims(p_jk * (1 - r3_jk)**l2_3l_survive, 0), -1), (2, 3, 4)) + 
                   np.expand_dims(np.sum(gamma_ijk * np.expand_dims(p_jk * (1 - r3_jk)**l2_3l_survive, 0), -1), (1, 2, 4))) * l_ij1k1j2k2
                - (2 * c1 + np.expand_dims(lam_ijk, (3, 4)) + np.expand_dims(lam_ijk, (1, 2)) + delta_l2) * l_ij1k1j2k2
                + 0.5 * triple_lysogens_displace_double * (
                    np.expand_dims(gamma_ijk * np.expand_dims(p_jk * r3_jk, 0), (3, 4)) * (
                        np.expand_dims(np.sum(l_ij1k1j2k2, (3, 4)) + np.sum(l_ij1k1j2k2, (1, 2)), (1, 2)) - np.expand_dims(np.sum(l_ij1k1j2k2, 2), 2))
                    + np.expand_dims(gamma_ijk * np.expand_dims(p_jk * r3_jk, 0), (1, 2)) * (
                        np.expand_dims(np.sum(l_ij1k1j2k2, (3, 4)) + np.sum(l_ij1k1j2k2, (1, 2)), (3, 4)) - np.expand_dims(np.sum(l_ij1k1j2k2, 4), 4))
                )),
            l_ij1k1j2k2))

    return(dB_dt, dB_dt_Kinf, dP_dt, dL_dt, dL_dt_Kinf, dL2_dt, dL2_dt_Kinf)


def phage_lysogen_simulation(b0, p0, l0, l20, t_init, t_max, Kinf_pop_floor, euler_method, dt_tols,
                             params, save_every=1, t_eval=None):
        
    dt, rtol, atol = dt_tols
    
    (alpha_b, delta_b, beta, delta_p, alpha_l, delta_l, alpha_l2, delta_l2, lam_ijk, 
     c1, c2, K, r1_jk, r2_jk, r3_jk, gamma_ijk, phage_mask, lysogen_mask, double_lysogen_mask, 
     phage_die_when_infecting_lysogen, triple_lysogens_displace_double,
     only_phage_die_with_multiple_lys, log_space) = params
    
    if log_space:  # need to convert variables to log-space
        b0 = np.log(b0)
        p0 = np.log(p0)
        l0 = np.log(l0)
        l20 = np.log(l20)

    dB_dt, dB_dt_Kinf, dP_dt, dL_dt, dL_dt_Kinf, dL2_dt, dL2_dt_Kinf = get_diff_eqs(params)
    Kinf, pop_floor = Kinf_pop_floor
    
    if log_space:
        pop_floor = np.log(pop_floor)
        pop_floor_set = -100  #-np.inf
    else:
        pop_floor_set = 0
        
    not_bool_phage_mask = ~np.array(phage_mask, dtype=bool)
    not_bool_lysogen_mask = ~np.array(lysogen_mask, dtype=bool)
   
    N_b = len(b0)
    N_p, N_r = p0.shape
    
    start = time.time()
    if euler_method:
        t_vec = np.arange(t_init, t_max + dt, dt)
        N_t = len(t_vec)
        
        # save_every = int(max(1, 1/dt))
        N_save = int(N_t / save_every)
        if dt < save_every:
            N_save += 1
        
        b_t = np.zeros((N_b, N_save))
        p_t = np.zeros((N_p, N_r, N_save))
        l_t = np.zeros((N_b, N_p, N_r, N_save))
        l2_t = np.zeros((N_b, N_p, N_r, N_p, N_r, N_save))
        t_vec_save = np.zeros(N_save)
        
        b = b0
        p = p0
        l = l0
        l2 = l20
        
        for t_e, t in enumerate(t_vec):
            b_store = b
            p_store = p
            l_store = l
            l2_store = l2
            
            if t_e % save_every == 0:
                b_t[:, t_e // save_every] = b_store
                p_t[:, :, t_e // save_every] = p_store
                l_t[:, :, :, t_e // save_every] = l_store
                l2_t[:, :, :, :, :, t_e // save_every] = l2_store
                t_vec_save[t_e // save_every] = t
                   
            if Kinf:
                b = b + dB_dt_Kinf(b_store, p_store, l_store, l2_store) * dt
                l = l + dL_dt_Kinf(b_store, p_store, l_store, l2_store) * dt
                l2 = l2 + dL2_dt_Kinf(b_store, p_store, l_store, l2_store) * dt
            else:
                b = b + dB_dt(b_store, p_store, l_store, l2_store) * dt
                l = l + dL_dt(b_store, p_store, l_store, l2_store) * dt
                l2 = l2 + dL2_dt(b_store, p_store, l_store, l2_store) * dt
            p = p + dP_dt(b_store, p_store, l_store, l2_store) * dt
            
            b[b < pop_floor] = pop_floor_set  # impose non-negativity
            p[p < pop_floor] = pop_floor_set  # impose non-negativity
            l[l < pop_floor] = pop_floor_set  # impose non-negativity
            l2[l2 < pop_floor] = pop_floor_set  # impose non-negativity
            
            if alpha_l2 == 0:
                l2[l2 > pop_floor] = pop_floor_set
                
            p[not_bool_phage_mask] = pop_floor_set  # p = p * phage_mask  # doesn't work if we're in log-space
            l[not_bool_lysogen_mask] = pop_floor_set  # l = l * lysogen_mask  # doesn't work if we're in log-space

            if (t_e) % 200000 == 0 and t > 0:
                print('Completed t = ' + str(t) + ' out of ' + str(t_max - t_init) + 
                      ' and time elapsed = ' + str(time.time() - start))
        t_vec = t_vec_save
        
    else:
        def phage_ODEs(t, z, Kinf):
            z[z < pop_floor] = pop_floor_set  # impose non-negativity
            
            b_t = z[:N_b]
            p_t = z[N_b:N_b + N_p * N_r].reshape((N_p, N_r))
            l_t = z[N_b + N_p * N_r: N_b + N_p * N_r + N_b * N_p * N_r].reshape((N_b, N_p, N_r))
            l2_t = z[N_b + N_p * N_r + N_b * N_p * N_r:].reshape((N_b, N_p, N_r, N_p, N_r))
                 
            if alpha_l2 == 0:
                l2_t[l2_t > pop_floor] = pop_floor_set
            p_t[not_bool_phage_mask] = pop_floor_set  #p_t = p_t * phage_mask  # doesn't work if we're in log-space
            l_t[not_bool_lysogen_mask] = pop_floor_set  #l_t = l_t * lysogen_mask  # doesn't work if we're in log-space

            if Kinf:
                dzdt = np.concatenate((
                    dB_dt_Kinf(b_t, p_t, l_t, l2_t), 
                    dP_dt(b_t, p_t, l_t, l2_t).reshape(N_p * N_r), 
                    dL_dt_Kinf(b_t, p_t, l_t, l2_t).reshape(N_b * N_p * N_r),
                    dL2_dt_Kinf(b_t, p_t, l_t, l2_t).reshape(N_b * N_p * N_r * N_p * N_r)))
            else:
                dzdt = np.concatenate((
                    dB_dt(b_t, p_t, l_t, l2_t), 
                    dP_dt(b_t, p_t, l_t, l2_t).reshape(N_p * N_r),
                    dL_dt(b_t, p_t, l_t, l2_t).reshape(N_b * N_p * N_r),
                    dL2_dt(b_t, p_t, l_t, l2_t).reshape(N_b * N_p * N_r * N_p * N_r)))
            return(dzdt)
        
        
        def phage_Sol(tSpan, params, b0, p0, l0, l20, Kinf, t_eval):
            
            initial_conditions = np.concatenate((b0, 
                                                 p0.reshape(N_p * N_r), 
                                                 l0.reshape(N_b * N_p * N_r), 
                                                 l20.reshape(N_b * N_p * N_r * N_p * N_r)))
            # solve ODE
            odeSol = scipy.integrate.solve_ivp(
                lambda tSpan, z: phage_ODEs(tSpan, z, Kinf),
                tSpan, initial_conditions, method = 'DOP853', vectorized=False,  # DOP853  
                # Radau is much slower, doesn't appear to give qualitatively different results. RK45 is slightly faster
                rtol=rtol, atol=atol,  # default 1e-3 and 1e-6
                t_eval=t_eval,
                # jac = lambda tSpan,z: phage_Jac(tSpan,z,params)  # slows down the calculation??
                )
            
            z = odeSol.y
            t = odeSol.t
            
            b_t = z[:N_b, :]
            p_t = z[N_b:N_b + N_p * N_r, :].reshape((N_p, N_r, len(t)))
            l_t = z[N_b + N_p * N_r: N_b + N_p * N_r + N_b * N_p * N_r, :].reshape((N_b, N_p, N_r, len(t)))
            l2_t = z[N_b + N_p * N_r + N_b * N_p * N_r:, :].reshape((N_b, N_p, N_r, N_p, N_r, len(t)))
            # l_t = np.zeros((N_b, N_p, b_t.shape[1]))
            # for i in range(b_t.shape[1]):
            #     l_t[:, :, i] = z[N_b + N_p:, i].reshape((N_b, N_p))
                
            return t, b_t, p_t, l_t, l2_t
    
        save_every = 1  # don't change
        t_vec, b_t, p_t, l_t, l2_t = phage_Sol([t_init, t_max], [], b0, p0, l0, l20, Kinf, t_eval)
        print('Time elapsed = ' + str(time.time() - start))
    
    # print('Time elapsed = ' + str(time.time() - start))
    
    if log_space:  # need to convert variables back to linear space
        b_t = np.exp(b_t)
        p_t = np.exp(p_t)
        l_t = np.exp(l_t)
        l2_t = np.exp(l2_t)
        
    return(t_vec, b_t, p_t, l_t, l2_t)

    
def get_maxes_and_mins_indices(v):
   from scipy.signal import argrelextrema
   return(argrelextrema(v, np.greater)[0], argrelextrema(v, np.less)[0])


def get_maxes_and_mins_values(v):
    max_indices, min_indices = get_maxes_and_mins_indices(v)
    if len(max_indices) > len(min_indices):
        max_indices = max_indices[1:]
    elif len(min_indices) > len(max_indices):
        min_indices = min_indices[1:]
    return(v[max_indices], v[min_indices])


def get_nearby_maxes(v1, v2):
    # For each max of v1, find the next max of v2. If there are two, keep nearest pair
    max_indices_1, min_indices_1 = get_maxes_and_mins_indices(v1)
    max_indices_2, min_indices_2 = get_maxes_and_mins_indices(v2)

    max_indices_1_to_keep = []
    max_indices_2_to_keep = []
    for mi1 in max_indices_1:
        max_indices_2 = max_indices_2[max_indices_2 >=mi1]
        if not len(max_indices_2):
            break
        mi2 = max_indices_2[np.argmin((max_indices_2 - mi1))]
        if len(max_indices_2_to_keep) and mi2 == max_indices_2_to_keep[-1]:  
            # then we have two maxes in v1 before one in v2; keep later one
            max_indices_1_to_keep[-1] = mi1
        else:
            max_indices_1_to_keep += [mi1]
            max_indices_2_to_keep += [mi2]
        
    return(v1[max_indices_1_to_keep], v2[max_indices_2_to_keep])


def which_val_is_largest(vs):
    # Given a list of vs, return an array of which v has the largest element at each point
    return([np.argmax(vs[:, t]) for t in range(len(vs[0, :]))])


def make_title(params, Kinf_pop_floor):
    (alpha_b, delta_b, beta, delta_p, alpha_l, delta_l, alpha_l2, delta_l2, 
     lam_ijk, c1, c2, K, r1_jk, r2_jk, r3_jk, gamma_ijk, phage_mask, lysogen_mask, double_lysogen_mask, 
     phage_die_when_infecting_lysogen, triple_lysogens_displace_double,
     only_phage_die_with_multiple_lys, log_space) = params
    Kinf, pop_floor = Kinf_pop_floor
    
    N_b, N_p, N_r = gamma_ijk.shape
    lam = np.max(lam_ijk)
    
    title_1 = (r'$N_b = $' + str(N_b) + '; ' + r'$N_p = $' + str(N_p) + '; ' + r'$N_r = $' + str(N_r) + '; ' + 
               r'$\lambda = $' + str(lam) + '; c1 = ' + str(c1) + '; c2 = ' + str(c2) + '; '
               r'$\frac{\alpha_l}{\alpha_b}$ = ' + str(alpha_l / alpha_b) + '; ' + 'pop_floor = ' + str(pop_floor))
    title_2 = (r'$\frac{\alpha_{l2}}{\alpha_l}$ = ' + str(alpha_l2 / alpha_l) + '; ' +
               r'$\delta_b = $' + str(delta_b) + '; ' + 
               r'$\delta_p = $' + str(delta_p) + '; ' +
               r'$\beta = $' + str(beta) )
    if Kinf:
        title_2 += '; K = ' + r'$\infty$'
    else:
        title_2 += '; K = ' + str(K)
    title = title_1 + '\n' + title_2
    if len(np.unique(gamma_ijk)) == 1:
        title += '; ' + r'$\gamma = $' + str(np.max(gamma_ijk))
    
    title +=  ('; ' + r'$q_1 = $' + str(phage_die_when_infecting_lysogen) + '; ' + 
               r'$q_2 = $' + str(triple_lysogens_displace_double) + '; ' + 
               r'$q_3 = $' + str(only_phage_die_with_multiple_lys) + '; ' + 
               'log_space_' + str(log_space))
    return(title)





# def simulation_with_mutations(
#         b0, p0, l0, l20, t_init, t_max, Kinf_pop_floor, euler_method, dt_tols,
#         params, p_lytic, p_lys, p_inc, p_dec, num_mutations, save_every=1):
    
#     t_vec, b_t, p_t, l_t, l2_t = phage_lysogen_simulation(
#         b0, p0, l0, l20, t_init, t_max, Kinf_pop_floor, euler_method, dt_tols,
#         params, save_every=save_every)
#     t_vecs = [t_vec]
#     b_ts = [b_t]
#     p_ts = [p_t]
#     l_ts = [l_t]
#     l2_ts = [l2_t]
#     num_new_species_vec = []
    
#     (alpha_b, delta_b, beta, delta_p, alpha_l, delta_l, alpha_l2, delta_l2, 
#               lam_ijk, c1, c2, K, r1_jk, r2_jk, r3_jk, gamma_ijk, phage_mask, lysogen_mask, double_lysogen_mask, 
#               phage_die_when_infecting_lysogen, triple_lysogens_displace_double,
#               only_phage_die_with_multiple_lys, log_space) = params
#     r1_jks = [r1_jk]

#     for n in range(num_mutations):        
#         new_r1_jk, p0, l0, l20, lam_ijk, num_new_species = branching_ratio_mutations(
#             r1_jk, p_t[:, :, -1], avg_value(p_t, t_vec),
#             p_lytic, p_lys, p_inc, p_dec, 
#             l_t[:, :, :, -1], l2_t[:, :, :, :, :, -1], lam_ijk)
#         b0 = b_t[:, -1]
#         t_init_prev = t_init
#         t_init = t_max
#         t_max = t_max + t_max - t_init_prev
        
#         r1_jk = new_r1_jk
#         r2_jk = new_r1_jk
#         r3_jk = new_r1_jk
#         params = [alpha_b, delta_b, beta, delta_p, alpha_l, delta_l, alpha_l2, delta_l2, 
#                   lam_ijk, c1, c2, K, r1_jk, r2_jk, r3_jk, gamma_ijk, phage_mask, lysogen_mask, double_lysogen_mask, 
#                   phage_die_when_infecting_lysogen, triple_lysogens_displace_double, 
#                   only_phage_die_with_multiple_lys, log_space]
        
#         if len(p0) != 0:
#             t_vec, b_t, p_t, l_t, l2_t = phage_lysogen_simulation(
#                 b0, p0, l0, l20, t_init, t_max, Kinf_pop_floor, euler_method, dt_tols,
#                 params, save_every=save_every)
#             t_vecs += [t_vec]
#             b_ts += [b_t]
#             p_ts += [p_t]
#             l_ts += [l_t]
#             l2_ts += [l2_t]
#         else:
#             t_vecs += [np.array([t_vec[-1]])]
#             b_ts += [np.array([b0])]  # .transpose() for all of the following?
#             p_ts += [np.array([p0])]      
#             l_ts += [np.array([l0])]      
#             l2_ts += [np.array([l20])]      
#         r1_jks += [new_r1_jk]
#         num_new_species_vec += [num_new_species]
        
#         if len(p0) == 0:  # no more phage
#             break        

#     return(t_vecs, b_ts, p_ts, l_ts, l2_ts, r1_jks, num_new_species_vec)


# def branching_ratio_mutations(r_jk, p_fin, p_mean, p_lytic, p_lys, p_inc, p_dec, 
#                               l_fin, l2_fin, lam_ijk):
# # =============================================================================
# #     p_fin is the phage population at the timepoint of mutation
# #     p_mean is the mean phage population since the last mutation
# #     p_lytic, p_lys, p_inc, p_dec are the probabilities that a phage will go lytic,
# #     lysogenic, increase r, and decrease r, respectively.
# # =============================================================================    
#     r_new_lys = 0.2    # initialize all new lysogens with r=r_new_lys
#     r_inc = 0.1  # if r increases, increase it by this increment
#     r_dec = 0.1  # if r decreases, decrease it by this increment
#     max_poss_num_rs_per_phage = 21   # have no more than this number of branching ratios per phage
    
#     #init_cst_pop = True  # if True always initialize the same # of phage; else, initialize proportional to prob
#     init_pop = 1e-6  # population to initialize after mutation for phage
#     init_pop_l = 0  # population to initialize after mutation for single lysogens
#     init_pop_l2 = 0  # population to initialize after mutation for double lysogens
    

#     N_b, N_p, N_r = l_fin.shape

#     rand_nums = np.random.random((N_p, N_r, 4))
#     which_p_go_lytic = (rand_nums[:, :, 0] * p_mean < p_lytic) * (r_jk > 0)
#     which_p_go_lysogenic = (rand_nums[:, :, 1] * p_mean < p_lys) * (r_jk == 0)
#     which_p_inc = (rand_nums[:, :, 2] * p_mean < p_inc) * (r_jk > 0) * (r_jk <= 1 - r_inc)  # can't increase further than 1
#     which_p_dec = (rand_nums[:, :, 3] * p_mean < p_dec) * (r_jk > r_dec)  # can't decrease further than r_dec
    
#     num_new_species = [np.sum(which_p_go_lytic), np.sum(which_p_go_lysogenic),
#                        np.sum(which_p_inc), np.sum(which_p_dec)]
    
#     new_r_jk = np.ones((N_p, max_poss_num_rs_per_phage)) * -1  # initialize to -1
#     new_p = np.zeros((N_p, max_poss_num_rs_per_phage))  # the new phage populations
#     max_num_rs_per_phage = 0
    
#     for j in range(N_p):
#         # Make the list of which branching ratios we'll have for each phage
#         curr_rs_per_phage = r_jk[j, :][p_fin[j, :] > 0]  # remove rs that don't have a phage population attached
#         new_rs_per_phage = []
#         if np.any(which_p_go_lytic[j, :]):
#             new_rs_per_phage += [0]
#         if np.any(which_p_go_lysogenic[j, :]):
#             new_rs_per_phage += [r_new_lys]
#         for r in range(N_r):
#             if which_p_inc[j, r]:
#                 new_rs_per_phage += [np.round(r_jk[j, r] + r_inc, 2)]  # round to avoid floating point errors
#             if which_p_dec[j, r]:
#                 new_rs_per_phage += [np.round(r_jk[j, r] - r_dec, 2)]
        
#         new_rs_per_phage = np.unique(new_rs_per_phage)  # also sorts them
#         rs_per_phage = np.unique(list(curr_rs_per_phage) + list(new_rs_per_phage))  # also sorts them
#         if len(rs_per_phage) > max_poss_num_rs_per_phage:
#             rs_per_phage = rs_per_phage[:max_poss_num_rs_per_phage]
#         num_rs = len(rs_per_phage)
#         if num_rs > max_num_rs_per_phage:
#             max_num_rs_per_phage = num_rs
#         new_r_jk[j, :num_rs] = rs_per_phage
        
#         # Get the phage population for each branching ratio
#         for e, r in enumerate(rs_per_phage):
#             if r in r_jk[j, :]:
#                 new_p[j, e] = p_fin[j, np.where(r_jk[j, :] == r)[0][0]]
#             else:
#                 new_p[j, e] = init_pop
    
#     new_r_jk = new_r_jk[:, :max_num_rs_per_phage]
#     new_p = new_p[:, :max_num_rs_per_phage]
    
#     new_l = np.zeros((N_b, N_p, max_num_rs_per_phage))
#     new_l2 = np.zeros((N_b, N_p, max_num_rs_per_phage, N_p, max_num_rs_per_phage))
    
#     for i in range(N_b):
#         for j in range(N_p):
#             for e, r in enumerate(new_r_jk[j, :]):
#                 if r in r_jk[j, :]:
#                     new_l[i, j, e] = l_fin[i, j, np.where(r_jk[j, :] == r)[0][0]]
#                 elif r > 0:
#                     new_l[i, j, e] = init_pop_l
#                 else:
#                     new_l[i, j, e] = 0
                
#                 for j2 in range(j):  # l2 is zero if j2 >= j1
#                     for e2, r2 in enumerate(new_r_jk[j2, :]):
#                         if r in r_jk[j, :] and r2 in r_jk[j2, :]:
#                             new_l2[i, j, e, j2, e2] = l2_fin[i, j, np.where(r_jk[j, :] == r)[0][0], 
#                                                              j2, np.where(r_jk[j2, :] == r2)[0][0]]
#                         elif r > 0 and r2 > 0:
#                             new_l2[i, j, e, j2, e2] = init_pop_l2
#                         else:
#                             new_l2[i, j, e, j2, e2] = 0

#     lam = np.max(lam_ijk)
#     new_lam_ijk = np.ones(np.shape(new_l)) * lam * (np.expand_dims(new_r_jk, 0) > 0)
#     return(new_r_jk, new_p, new_l, new_l2, new_lam_ijk, num_new_species)
    
#     # num_new_rs_per_phage = np.sum(which_p_go_lytic + which_p_go_lysogenic + which_p_inc + which_p_dec, -1)
#     # num_curr_rs_per_phage = np.sum(p_fin > 0, -1)
#     # num_rs = min(20, np.max(num_new_rs_per_phage + num_curr_rs_per_phage)) 
    