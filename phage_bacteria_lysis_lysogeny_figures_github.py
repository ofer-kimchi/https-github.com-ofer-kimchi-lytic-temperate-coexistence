#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:35:56 2022

@author: Ofer
"""
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import seaborn as sns
import pandas as pd
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy import stats

import matplotlib
matplotlib.rc('font', **{'size' : 22})

save_figs = False

from phage_bacteria_def_fxns_github import ( 
    phage_lysogen_simulation, make_title, flatten)


def other_vars_from_params(N_b, N_p, N_r, r1_jk, lam, gamma, c1, alpha_l2, p0_rand=True, 
                           p0_init=1e-3, l0_init=1e-3, b0_init=1e-3, l20_init=1e-3, norm_p0_by_Nr=True):
    # Don't allow lysogens or double lysogens to exist for obligate lytic phage
    double_lysogen_mask = np.expand_dims(np.tile(np.tri(N_p) - np.eye(N_p), (N_b, 1, 1)), (2, 4))  # zeros out if j1 <= j2
    # double_lysogen_mask *= (alpha_l2 > 0)
    double_lysogen_mask = np.ones((N_b, N_p, N_r, N_p, N_r)) * double_lysogen_mask
    double_lysogen_mask *= (np.expand_dims(r1_jk, (0, 1, 2)) > 0)
    double_lysogen_mask *= (np.expand_dims(r1_jk, (0, 3, 4)) > 0)

    lysogen_mask = np.ones((N_b, N_p, N_r))
    lysogen_mask *= (np.expand_dims(r1_jk, 0) > 0)
    
    phage_mask = np.ones((N_p, N_r))
    for j in range(N_p):
        for k in range(1, N_r):
            if r1_jk[j, k] == r1_jk[j, k-1]:
                phage_mask[j, k] = 0
                lysogen_mask[:, j, k] = 0
    
    r2_jk = r1_jk # branching ratio for double lysogen
    r3_jk = r1_jk
    lam_ijk = np.ones((N_b, N_p, N_r)) * lam * (np.expand_dims(r1_jk, 0) > 0)
    gamma_ijk = np.ones((N_b, N_p, N_r)) * gamma  # * 2 * np.random.random((N_b, N_p, N_r))  # 
    
    b0 = np.ones(N_b) * (c1 > 0) * b0_init # * np.random.random(N_b) # * 1 / (1 * N_b)  # bacterial Population size  # (1 * N_b could be K * N_b if desired)
    if p0_rand != 0:
        np.random.seed(p0_rand)
        p0 = np.random.random((N_p, N_r)) * p0_init # * (1 + np.random.random(N_p) * 0.5) # * np.random.random(N_p) * 0.02 
    else:
        p0 = np.ones((N_p, N_r)) * p0_init       
    if norm_p0_by_Nr:
        p0 = p0 / np.expand_dims(np.sum(phage_mask, 1), 1)
    l0 = np.ones((N_b, N_p, N_r)) * l0_init #* np.expand_dims(r1_jk > 0, 0)  # lysogen Population size
    l20 = l20_init * (alpha_l2 > 0) * np.ones((N_b, N_p, N_r, N_p, N_r)) * (  # double lysogen Population size
        double_lysogen_mask)
    
    p0 = p0 * phage_mask
    l0 = l0 * lysogen_mask
    
    return(double_lysogen_mask, phage_mask, lysogen_mask, r2_jk, r3_jk, lam_ijk, gamma_ijk, b0, p0, l0, l20)


def dt_tol_params(euler_method, t_max, use_t_eval=True, t_eval_dt=0.1):
    rtol = 1e-5#1e-8  # min is 2.3e-14
    atol = 1e-9 #1e-15
    if euler_method:
        dt = 0.001
        save_every = int(max(1, 1/dt / 5))
        t_eval = None  # Doesn't have any effect
    else:
        save_every = 1
        if use_t_eval:
            dt = t_eval_dt
            t_eval = np.arange(0, t_max + dt/2, dt)
        else:
            dt = 0
            t_eval = None  # None means use points chosen by the solver; not None doesn't work (gives wrong answers like non-zero l2).            
    return(dt, save_every, t_eval, rtol, atol)


lytic_linestyle = '--'
temperate_linestyles = {0.1: ':', 0.2: '-', 0.3: '-.', 0.4:':', 1.: ':'}
linestyles = {0: lytic_linestyle}
linestyles.update(temperate_linestyles)

all_solid_temperate_linestyles = {0.: lytic_linestyle, 0.1: '-', 0.2: '-', 0.3: '-', 0.4:'-', 1.: '-'}

temp_lyt = {0.: 'OL', 0.2: 'T'}
for i in [0.1, 0.2, 0.3, 0.4, 1.]:
    temp_lyt[i] = temp_lyt[0.2]

def p_label(j_print, k_print, r, include_T_OL=False, include_r=False):
    label = r'$P_{{{0},{1}}}$'.format(j_print, k_print)
    if include_T_OL:
        label +=  '; ' + temp_lyt[r]
    if include_r and r > 0:
        label += '; ' + r'$f=$' + "{:.1f}".format(r)
    return(label)

def l_label(j_print, k_print, r, include_T_OL=False, include_r=False):
    label = r'$L_{{{0},{1}}}$'.format(j_print, k_print)
    if include_T_OL:
        label +=  '; ' + temp_lyt[r]
    if include_r and r > 0:
        label += '; ' + r'$f=$' + "{:.1f}".format(r)
    return(label)

#colors = ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f', '#bf5b17']
colors = ['#1E88E5', '#FFC107', '#D81B60', '#004D40', '#501ACA', '#65C598']
dark_colors = ['#0B3B65', '#8C6902', '#7D0E36', '#03B99A', '#8F6AE2']
double_lysogen_colors = [[''] * 4, ['#1E88E5'] + [''] * 3, ['#D81B60', '#FFC107'] + [''] * 2, 
                         ['#004D40', '#501ACA', '#65C598', '']]
# =============================================================================
# Parameters that don't change
# =============================================================================

N_b = 1  # number of initial bacterial species.
K = 1000  # carrying capacity; also affects initial conditions for bacteria
Kinf = True  # should we set K to infinity? (if it's true, we disregard the value of K set above)

pop_floor = 1e-20  # if Population decreases below this number, set it to 0
Kinf_pop_floor = [Kinf, pop_floor]

alpha_b = 1.05  # growth rate for bacteria 
alpha_l = 1 # alpha_b * 0.95  # growth rate for lysogen
alpha_l2 = alpha_l * 0#.95  # growth rate for lysogen

doubling_time_lys = np.log(2) / alpha_l  # doubling time for lysogens

beta = 22#12  # burst size

delta_b = 0  # 0.001  # death rate for bacteria
delta_p = 1e0  # death rate for phage
delta_l = delta_b  # death rate for lysogen
delta_l2 = delta_b  # death rate for lysogen

lam = 1e-4  # induction rate  # 1e-5
c1 = 0#1e-5  # 1e-5  # rate of prophage -> cryptic prophage for single lysogen
c2 = 0#1e-5  # rate of prophage -> cryptic prophage for double lysogen

gamma = 1  #0.8 #* 2  #0.42323773

phage_die_when_infecting_lysogen = True
triple_lysogens_displace_double = False

only_phage_die_with_multiple_lys = True
log_space = False

def save_svg_and_png(file_name, trials=False):
    folder_name = 'Figures/NovRev/'
    if only_phage_die_with_multiple_lys:
        folder_name += 'only_phage_die_with_multiple_lys/'
    else:
        folder_name += 'old_'
    if trials:
        folder_name += 'trials/'
    plt.savefig(folder_name + 'SVGs/' + file_name + '.svg', bbox_inches='tight')          
    plt.savefig(folder_name + 'PNGs/' + file_name + '.png', bbox_inches='tight')          

#%%
# =============================================================================
# =============================================================================
# =============================================================================
# # #                   Begin main text figures
# =============================================================================
# =============================================================================
# =============================================================================

#%%
# =============================================================================
# Show long time coexistence of multiple species
# =============================================================================

t_max = 1000
euler_method = False

N_p = 3  # number of initial phage species
N_r = 1  # number of branching ratios for each phage species
p0_rand = 1

r1_jk = np.round(np.ones((N_p, N_r)) *  # branching ratio for single lysogen
    np.expand_dims(np.linspace(0.2, 0.2, N_r), 0)  # all phages have same r's
    #np.linspace(0., 0.7, N_p * N_r).reshape((N_p, N_r))  # all different r's
    #np.array([0, 0.2]).reshape((N_p, N_r))  # something we specify
    , 6)

double_lysogen_mask, phage_mask, lysogen_mask, r2_jk, r3_jk, lam_ijk, gamma_ijk, b0, p0, l0, l20 = other_vars_from_params(
    N_b, N_p, N_r, r1_jk, lam, gamma, c1, alpha_l2, p0_rand=p0_rand, p0_init=1e-7, l0_init=1e-2)
dt, save_every, t_eval, rtol, atol = dt_tol_params(euler_method, t_max, use_t_eval=True)
dt_tols = [dt, rtol, atol]

params = [alpha_b, delta_b, beta, delta_p, alpha_l, delta_l, alpha_l2, delta_l2, 
          lam_ijk, c1, c2, K, r1_jk, r2_jk, r3_jk, gamma_ijk, phage_mask, lysogen_mask, double_lysogen_mask,
          phage_die_when_infecting_lysogen, triple_lysogens_displace_double,
          only_phage_die_with_multiple_lys, log_space]

title = make_title(params, Kinf_pop_floor)

t_vec, b_t, p_t, l_t, l2_t = phage_lysogen_simulation(
    b0, p0, l0, l20, 0, t_max, Kinf_pop_floor, euler_method, dt_tols, params, save_every, t_eval)

if only_phage_die_with_multiple_lys:
    t_start = 6000
else:
    t_start = 7300
t_end = t_start + int(700 * doubling_time_lys)

if t_start > len(t_vec):
    t_start = 0
    t_end = len(t_vec)

plt.figure()
for j in range(N_p):
    for k in range(N_r):
        if phage_mask[j, k]:
            plt.plot(t_vec[t_start:t_end], p_t[j, k, t_start:t_end], linewidth=4,
                     linestyle=linestyles[r1_jk[j, k]], color=colors[j],
                     label=#p_label(j+1, k+1, r1_jk[j, k], include_T_OL=True, include_r=False)
                     p_label(j+1, r1_jk[j, k], r1_jk[j, k], include_T_OL=False, include_r=False)
                     )
    plt.plot(t_vec[t_start:t_end], np.sum(l_t[0, j, :, t_start:t_end], 0), linewidth=1,
             color=colors[j], label=r'$L_{0}$'.format(j+1), alpha=0.5)
# Really make blue pop
j = 0
plt.plot(t_vec[t_start:t_end], p_t[j, k, t_start:t_end], linewidth=4,
         linestyle=linestyles[r1_jk[j, k]], color=colors[j],
         )

plt.yscale('log')
plt.xlabel('Time (generations)')
plt.ylabel('Population')
plt.yticks([1e0, 1e-2, 1e-4])
plt.ylim([5e-6, 2e1])
#plt.ylim([1e-6, 1e2])    

plt.xticks(np.arange(t_start/10, (t_end + 1)/10, 10 * doubling_time_lys), [])
ax = plt.gca()
scalebar = AnchoredSizeBar(ax.transData,
                           10 * doubling_time_lys, '10', loc='lower right', 
                           pad=0.5,
                           color='black',
                           frameon=False,
                           label_top=True
                           )
ax.add_artist(scalebar)
         
leg = plt.legend(fontsize=13, bbox_to_anchor=(1.05, 1))  # loc='lower right')
for legobj in leg.legendHandles:
    if legobj.get_linestyle() != '-':
        if legobj.get_linestyle() == '--':
            legobj.set_linewidth(2.5)  # 2.
        else:
            legobj.set_linewidth(2.5)
if save_figs:
    save_svg_and_png('coexistence_temp')
plt.show()


#%%
# =============================================================================
# Show long time lytic/lysogenic coexistence for 1 strain
# =============================================================================

t_max = 1000
euler_method = False

N_p = 3  # number of initial phage species
N_r = 2  # number of branching ratios for each phage species
p0_rand = 1

r1_jk = np.round(np.ones((N_p, N_r)) *  # branching ratio for single lysogen
    #np.expand_dims(np.linspace(0., 0.2, N_r), 0)  # all phages have same r's
    #np.linspace(0., 0.7, N_p * N_r).reshape((N_p, N_r))  # all different r's
    np.array([[0.2, 0.], [0.2] * 2, [0.2] * 2]).reshape((N_p, N_r))  # something we specify
    , 6)

double_lysogen_mask, phage_mask, lysogen_mask, r2_jk, r3_jk, lam_ijk, gamma_ijk, b0, p0, l0, l20 = other_vars_from_params(
    N_b, N_p, N_r, r1_jk, lam, gamma, c1, alpha_l2, p0_rand=p0_rand, p0_init=1e-3, l0_init=1e-2)
dt, save_every, t_eval, rtol, atol = dt_tol_params(euler_method, t_max, use_t_eval=True)
dt_tols = [dt, rtol, atol]

params = [alpha_b, delta_b, beta, delta_p, alpha_l, delta_l, alpha_l2, delta_l2, 
          lam_ijk, c1, c2, K, r1_jk, r2_jk, r3_jk, gamma_ijk, phage_mask, lysogen_mask, double_lysogen_mask,
          phage_die_when_infecting_lysogen, triple_lysogens_displace_double,
          only_phage_die_with_multiple_lys, log_space]

title = make_title(params, Kinf_pop_floor)

t_vec, b_t, p_t, l_t, l2_t = phage_lysogen_simulation(
    b0, p0, l0, l20, 0, t_max, Kinf_pop_floor, euler_method, dt_tols, params, save_every, t_eval)


if only_phage_die_with_multiple_lys:
    t_start = 6400  #8000  # to show lytic phage not varying too much
    if alpha_l == 1 and lam == 1e-4:
        t_start = 7800
else:
    t_start = 7100  #8000  # to show lytic phage not varying too much

t_end = t_start + int(700 * doubling_time_lys)

plt.figure()
for j in range(N_p):
    for k in [1, 0]:
        if phage_mask[j, k]:
            plt.plot(t_vec[t_start:t_end], p_t[j, k, t_start:t_end], linewidth=4,
                     linestyle=linestyles[r1_jk[j, k]], color=colors[j],
                     label=#r'$P_{{{0},{1}}}$'.format(j+1, k+1) + '; ' + temp_lyt[r1_jk[j, k]]
                     p_label(j+1, r1_jk[j, k], r1_jk[j, k], include_T_OL=False, include_r=False)
                     )
    plt.plot(t_vec[t_start:t_end], np.sum(l_t[0, j, :, t_start:t_end], 0), linewidth=1,
             color=colors[j], label=r'$L_{0}$'.format(j+1), alpha=0.5)
# Really make blue pop
j = 0
for k in [1, 0]:
    if phage_mask[j, k]:
        plt.plot(t_vec[t_start:t_end], p_t[j, k, t_start:t_end], linewidth=4,
                 linestyle=linestyles[r1_jk[j, k]], color=colors[j],
                 )

plt.yscale('log')
plt.xlabel('Time (generations)')
plt.ylabel('Population')
plt.yticks([1e0, 1e-2, 1e-4])
if t_start==8000:
    plt.ylim([2e-6, 2e1]) 
elif t_start == 9100:
    plt.ylim([8e-6, 2e1]) 
else:
    plt.ylim([5e-6, 2e1])
plt.xticks(np.arange(t_start/10, (t_end + 1)/10, 10 * doubling_time_lys), [])
ax = plt.gca()
scalebar = AnchoredSizeBar(ax.transData,
                           10 * doubling_time_lys, '10', loc='lower right', 
                           pad=0.5,
                           color='black',
                           frameon=False,
                           label_top=True
                           )
ax.add_artist(scalebar)
   
leg = plt.legend(fontsize=13, bbox_to_anchor=(1.05, 1))  #loc='lower right')
for legobj in leg.legendHandles:
    if legobj.get_linestyle() != '-':
        if legobj.get_linestyle() == '--':
            legobj.set_linewidth(2.5)  # 2.
        else:
            legobj.set_linewidth(2.5)

if save_figs:
    save_svg_and_png('coexistence_Np3_Nr1')
plt.show()


# Zoom into how temperate and lytic strains differ; esp. highlight how lytic has higher growth potential
if only_phage_die_with_multiple_lys:
    t_sim_start = 5110
    if alpha_l == 1 and lam == 1e-4:
        t_sim_start = 7800
else:
    t_sim_start = 9110
t_max_2 = 500
t_eval_2 = np.arange(0, t_max_2 + 0.1/2, 0.1)

p0_2 = p_t[:, :, t_sim_start]
p0_2[0, :] = p0_2[0, 0]  # start temperate & lytic strains at same point

t_vec_2, b_t_2, p_t_2, l_t_2, l2_t_2 = phage_lysogen_simulation(
    b_t[:, t_sim_start], p0_2, l_t[:, :, :, t_sim_start], l2_t[:, :, :, :, :, t_sim_start], 
    0, t_max_2, Kinf_pop_floor, euler_method, dt_tols, params, save_every, t_eval_2)



t_start = 0  #8000  # to show lytic phage not varying too much
if only_phage_die_with_multiple_lys:
    t_end = t_start + int(70 * doubling_time_lys) + 1
else:
    t_end = t_start + int(50 * doubling_time_lys) + 1

plt.figure()
for j in range(N_p):
    for k in [1, 0]:
        if phage_mask[j, k]:
            plt.plot(t_vec_2[t_start:t_end], p_t_2[j, k, t_start:t_end], linewidth=4,
                     linestyle=linestyles[r1_jk[j, k]], color=colors[j],
                     label=#(r'$P_{{{0},{1}}}$'.format(j+1, k+1) + 
                           # '; ' #+ r'$r=$' + "{:.1f}".format(r1_jk[j, k])
                           # + temp_lyt[r1_jk[j, k]])
                           p_label(j+1, r1_jk[j, k], r1_jk[j, k], include_T_OL=False, include_r=False)
                     )
    plt.plot(t_vec_2[t_start:t_end], np.sum(l_t_2[0, j, :, t_start:t_end], 0), linewidth=1,
             color=colors[j], label=r'$L_{0}$'.format(j+1), alpha=0.5)
# Really make blue pop
j = 0
for k in [1, 0]:
    if phage_mask[j, k]:
        plt.plot(t_vec_2[t_start:t_end], p_t_2[j, k, t_start:t_end], linewidth=4,
                 linestyle=linestyles[r1_jk[j, k]], color=colors[j],
                 )


# times_for_slope = [20, 30]
    
# plt.plot(t_vec_2[times_for_slope[0]: times_for_slope[1]], 
#           # p_t_2[0, 0, times_for_slope[0]: times_for_slope[1]] * 
#           1e-5 * np.exp((-delta_p + beta * gamma * np.sum(
#               l_t_2[0, 1:, :, times_for_slope[0]: times_for_slope[1]], (0, 1))) * 
#               t_vec_2[times_for_slope[0]: times_for_slope[1]]), 
#           linewidth=2, color='k', linestyle=linestyles[0])
# plt.plot(t_vec_2[times_for_slope[0]: times_for_slope[1]], 
#           p0_2[0, 1] * np.exp((1 - r1_jk[0, 1]) * beta * gamma * np.sum(
#               l_t_2[0, 1:, :, times_for_slope[0]: times_for_slope[1]], (0, 1)) - delta_p), 
#           linewidth=2, color='k', linestyle=linestyles[r1_jk[0, 1]])


plt.yscale('log')
plt.xlabel('Time (generations)')
plt.ylabel('Population')
plt.yticks([1e0, 1e-2, 1e-4])
plt.ylim([5e-6, 2e1])
plt.xticks(np.arange(t_start/10, (t_end + 1)/10, 1 * doubling_time_lys), [])
ax = plt.gca()
scalebar = AnchoredSizeBar(ax.transData,
                           1 * doubling_time_lys, '1', loc='lower right', 
                           pad=0.5,
                           color='black',
                           frameon=False,
                           label_top=True
                           )
ax.add_artist(scalebar)
   
leg = plt.legend(fontsize=13, bbox_to_anchor=(1.05, 1))  #loc='lower right')
for legobj in leg.legendHandles:
    if legobj.get_linestyle() != '-':
        if legobj.get_linestyle() == '--':
            legobj.set_linewidth(2.5)  # 2.
        else:
            legobj.set_linewidth(2.5)

if save_figs:
    save_svg_and_png('lytic_opportunity')
plt.show()


#%%
# =============================================================================
# Show long time lytic/lysogenic coexistence for several strains, including bunching effects
# =============================================================================

t_max = 1100
euler_method = False

N_p = 3  # number of initial phage species
N_r = 4  # number of branching ratios for each phage species
p0_rand = 3

for diff_l0s in [True, False]:
    fig_title_append = diff_l0s * '_diff_l0s'

    # r1s = np.array([0., 0.1, 0.2, 0.3])
    r1s = np.array([0., 0.2, 0.3, 0.4])
    
    r1_jk = np.round(np.ones((N_p, N_r)) *  # branching ratio for single lysogen
        np.expand_dims(r1s, 0)  # all phages have same r's
        #np.linspace(0., 0.7, N_p * N_r).reshape((N_p, N_r))  # all different r's
        #np.array([0, 0.1, 0.2]).reshape((N_p, N_r))  # something we specify
        , 6)
    
    double_lysogen_mask, phage_mask, lysogen_mask, r2_jk, r3_jk, lam_ijk, gamma_ijk, b0, p0, l0, l20 = other_vars_from_params(
        N_b, N_p, N_r, r1_jk, lam, gamma, c1, alpha_l2, p0_rand=p0_rand, p0_init=1e-3, l0_init=1e-2)
    
    if diff_l0s:
        np.random.seed(1)
        l0 *= np.exp(7 * np.random.random(l0.shape)) / 100
        
    dt, save_every, t_eval, rtol, atol = dt_tol_params(euler_method, t_max, use_t_eval=True)
    dt_tols = [dt, rtol, atol]
    
    params = [alpha_b, delta_b, beta, delta_p, alpha_l, delta_l, alpha_l2, delta_l2, 
              lam_ijk, c1, c2, K, r1_jk, r2_jk, r3_jk, gamma_ijk, phage_mask, lysogen_mask, double_lysogen_mask,
              phage_die_when_infecting_lysogen, triple_lysogens_displace_double,
              only_phage_die_with_multiple_lys, log_space]
    
    title = make_title(params, Kinf_pop_floor)
    
    t_vec, b_t, p_t, l_t, l2_t = phage_lysogen_simulation(
        b0, p0, l0, l20, 0, t_max, Kinf_pop_floor, euler_method, dt_tols, params, save_every, t_eval)
    
    
    if not only_phage_die_with_multiple_lys:
        if np.array_equal(r1s, np.array([0., 0.1, 0.2, 0.3])):
            if p0_rand == 1:
                t_start = 4900 #4700
                t_end = 5192 #5311  
            elif p0_rand == 2:
                t_start = 6300 #4900 #4700
                t_end = 6800#5192 #5311
        elif np.array_equal(r1s, np.array([0., 0.2, 0.3, 0.4])):
            if p0_rand == 2:
                t_start = 6550
                t_end = 7250
        plt.figure()
        for j in [1, 2, 0]:
            k_list = [1, 2, 3, 0]
            for k in k_list: #range(N_r):
                if phage_mask[j, k]:
                    plt.plot(t_vec[t_start:t_end], p_t[j, k, t_start:t_end], linewidth=4 - 3 * (j > 0),
                             linestyle=linestyles[r1_jk[j, k]], color=colors[j],
                             # label=(r'$P_{{{0},{1}}}$'.format(j+1, k+1) + 
                             #        '; ' #+ r'$r=$' + "{:.1f}".format(r1_jk[j, k]
                             #        + temp_lyt[r1_jk[j, k]]
                             #        + '; ' + r'$r=$' + "{:.1f}".format(r1_jk[j, k])), 
                             alpha=1 * 0.55**(j))  # j=0 has alpha=1, j=1 (yellow) has larger alpha than j=2 (red)
            plt.plot(t_vec[t_start:t_end], np.sum(l_t[0, j, :, t_start:t_end], 0), linewidth=1 - 0.5 * (j > 0),
                     color=colors[j], #label=r'$L_{0}$'.format(j+1), 
                     alpha=0.5 * (j==0) + 0.25*(j > 0))
        
        plt.yscale('log')
        plt.xlabel('Time (generations)')
        plt.ylabel('Population')
        # plt.yticks([1e0, 1e-3, 1e-6])
        # plt.ylim([0.5e-8, 2e1])
        plt.yticks([1e0, 1e-4, 1e-8])
        plt.ylim([2e-10, 5e1])
        
        
        plt.xticks(np.arange(t_start/10, (t_end + 1)/10, 10 * doubling_time_lys), [])
        ax = plt.gca()
        scalebar = AnchoredSizeBar(ax.transData,
                                   10 * doubling_time_lys, '10', loc='lower right', 
                                   pad=0.5,
                                   color='black',
                                   frameon=False,
                                   label_top=True
                                   )
        ax.add_artist(scalebar)
        
        # Make legend labels
        j = 0
        k_list = range(N_r)  #[1, 2, 3, 0]  # k_list = range(N_r)
        for k in k_list:
            if phage_mask[j, k]:
                if k_list == [1, 2, 3, 0]:
                    k_mod = [4, 1, 2, 3]
                else:
                    k_mod = [1, 2, 3, 4]
                # label = (r'$P_{{c,{0}}}$'.format(k_mod[k]) + 
                #           '; '
                #           + temp_lyt[r1_jk[j, k]])
                # if r1_jk[j, k] > 0.:
                #     label += '; ' + r'$f=$' + "{:.1f}".format(r1_jk[j, k])
                label = p_label('c', r1_jk[j, k], r1_jk[j, k], include_T_OL=False, include_r=False)
                    
                plt.plot([t_vec[t_end]] * 2, [1e-8] * 2, color='k', 
                         linestyle=linestyles[r1_jk[j, k]], 
                         label=label, linewidth=0
                         )
        plt.plot([t_vec[t_end]] * 2, [1e-8] * 2, color='k', 
                 linestyle='-', 
                 label=r'$L_{c}$', linewidth=0
                 )
        for j in range(N_p):
            plt.plot([t_vec[t_end]] * 2, [1e-8] * 2, color=colors[j],
                     linestyle='-', linewidth=0, label=r'$c={0}$'.format(j+1))
        
        leg = plt.legend(fontsize=13, bbox_to_anchor=(1.05,1))#, handlelength=2) #loc='best') #
        for legobj in leg.legendHandles:
            if legobj.get_linestyle() != '-':
                if legobj.get_linestyle() == '--':
                    legobj.set_linewidth(2.5)  # 2.
                else:
                    legobj.set_linewidth(2.5)
            elif legobj.get_label() == r'$L_{c}$':
                legobj.set_linewidth(1)
            elif legobj.get_label() == r'$c=2$' or legobj.get_label() == r'$c=3$':
                legobj.set_linewidth(1.5)
            else:
                legobj.set_linewidth(4)
        
        if save_figs:
            save_svg_and_png('bunching_Np3_Nr4' + fig_title_append)
        plt.show()
    
    
    
    if np.array_equal(r1s, np.array([0., 0.1, 0.2, 0.3])):
        if p0_rand == 1:
            t_start = 5350 #4700
            t_end = 5800 #5311  
        elif p0_rand == 2:
            t_start = 7850
            t_end = 8250
    elif np.array_equal(r1s, np.array([0., 0.2, 0.3, 0.4])):
        if p0_rand == 2:
            t_start = 8300
            t_end = t_start + int(700 * doubling_time_lys) #8811
    elif np.array_equal(r1s, np.array([0.2, 0.3, 0.4, 0.])):
        if p0_rand == 2:
            t_start = 8500
            t_end = 9000
    if only_phage_die_with_multiple_lys:
        t_start = 2400
        t_end = t_start + int(700 * doubling_time_lys) #8811
        if alpha_l == 1 and lam == 1e-4:
            t_start = 9000 #2400
            t_end = t_start + int(1100 * doubling_time_lys) #8811
        
    plt.figure()# figsize=(8, 6))
    for j in [1, 2, 0]:
        k_list = [1, 2, 3, 0]  # k_list = range(N_r)
        for k in k_list:
            if phage_mask[j, k]:
                if k_list == [1, 2, 3, 0]:
                    k_mod = [4, 1, 2, 3]
                else:
                    k_mod = [1, 2, 3, 4]
    
                label = ''
                # label = (r'$P_{{{0},{1}}}$'.format(j+1, k_mod[k]) + 
                #          '; '
                #          + temp_lyt[r1_jk[j, k]])
                # if r1_jk[j, k] > 0.:
                #     label += '; ' + r'$r=$' + "{:.1f}".format(r1_jk[j, k])
    
                plt.plot(t_vec[t_start:t_end], p_t[j, k, t_start:t_end], linewidth=4 - 2 * (j > 0),
                         linestyle=linestyles[r1_jk[j, k]], color=colors[j],
                         label=label, 
                         alpha=1 * 0.55**(j))
                
                if diff_l0s and r1_jk[j, k] > 0:
                    plt.plot(t_vec[t_start:t_end], l_t[0, j, k, t_start:t_end], linewidth=1 - 0.5 * (j > 0),
                             linestyle=linestyles[r1_jk[j, k]], color=colors[j], #label=r'$L_{0}$'.format(j+1), 
                             alpha=0.5 * (j==0) + 0.25*(j > 0))
        if not diff_l0s:
            plt.plot(t_vec[t_start:t_end], np.sum(l_t[0, j, :, t_start:t_end], 0), linewidth=1 - 0.5 * (j > 0),
                     color=colors[j], #label=r'$L_{0}$'.format(j+1), 
                     alpha=0.5 * (j==0) + 0.25*(j > 0))
    
    plt.yscale('log')
    plt.xlabel('Time (generations)')
    plt.ylabel('Population')
    if alpha_l != 1 and lam != 1e-4:
        plt.yticks([1e0, 1e-4, 1e-8])
        plt.ylim([2e-10, 5e1])
    else:
        # plt.yticks([1e0, 1e-4, 1e-8])
        # plt.ylim([2e-9, 5e1])
        plt.yticks([1e0, 1e-2, 1e-4, 1e-6])
        plt.ylim([2e-8, 5e1])

    
    plt.xticks(np.arange(t_start/10, (t_end + 1)/10, 10 * doubling_time_lys), [])
    ax = plt.gca()
    scalebar = AnchoredSizeBar(ax.transData,
                               10 * doubling_time_lys, '10', loc='lower right', 
                               pad=0.5,
                               color='black',
                               frameon=False,
                               label_top=True
                               )
    ax.add_artist(scalebar)
    
    # Make legend labels
    j = 0
    k_list = range(N_r) #[1, 2, 3, 0]
    for k in k_list:
        if phage_mask[j, k]:
            if k_list == [1, 2, 3, 0]:
                k_mod = [4, 1, 2, 3]
            else:
                k_mod = [1, 2, 3, 4]
            # label = (r'$P_{{c,{0}}}$'.format(k_mod[k]) + 
            #           '; '
            #           + temp_lyt[r1_jk[j, k]])
            # if r1_jk[j, k] > 0.:
            #     label += '; ' + r'$f=$' + "{:.1f}".format(r1_jk[j, k])
            label = p_label('c', r1_jk[j, k], r1_jk[j, k], include_T_OL=False, include_r=False)
            
            plt.plot([t_vec[t_end]] * 2, [1e-8] * 2, color='k', 
                     linestyle=linestyles[r1_jk[j, k]], 
                     label=label, linewidth=0
                     )
            if diff_l0s and r1_jk[j, k] > 0:
                plt.plot([t_vec[t_end]] * 2, [1e-8] * 2, linewidth=0,
                         linestyle=linestyles[r1_jk[j, k]], color='k',
                         label=l_label('c', r1_jk[j, k], r1_jk[j, k], include_T_OL=False, include_r=False))
    if not diff_l0s:
        plt.plot([t_vec[t_end]] * 2, [1e-8] * 2, color='k', 
                 linestyle='-', 
                 label=r'$L_{c}$', linewidth=0
                 )
    for j in range(N_p):
        plt.plot([t_vec[t_end]] * 2, [1e-8] * 2, color=colors[j],
                 linestyle='-', linewidth=0, label=r'$c={0}$'.format(j+1))
    
    leg = plt.legend(fontsize=13, bbox_to_anchor=(1.05,1), ncol=1+diff_l0s)#, handlelength=2) #loc='best') #
    for legobj in leg.legendHandles:
        if legobj.get_linestyle() != '-':
            if legobj.get_linestyle() == '--':
                legobj.set_linewidth(2.5)  # 2.
            elif legobj.get_label() in [r'$L_{c}$'] + [
                    l_label('c', r1, r1, include_T_OL=False, include_r=False) for r1 in r1s]:
                legobj.set_linewidth(1)
            else:
                legobj.set_linewidth(2.5)
        elif legobj.get_label() in [r'$L_{c}$'] + [
                l_label('c', r1, r1, include_T_OL=False, include_r=False) for r1 in r1s]:
            legobj.set_linewidth(1)
        elif legobj.get_label() == r'$c=2$' or legobj.get_label() == r'$c=3$':
            legobj.set_linewidth(1.5)
        else:
            legobj.set_linewidth(4)
            
    if save_figs:
        save_svg_and_png('bunching_withlyt_Np3_Nr4' + fig_title_append)
    plt.show()
    
    
    
    # Make figure of ratio of phage to lysogen populations
    
    if only_phage_die_with_multiple_lys:
        if alpha_l == 1 and lam == 1e-4:  # show the same data as previously
            t_start = t_start #2400
            t_end = t_end
        else:
            t_start = 5000#4500
            t_end = t_start + int(900 * doubling_time_lys)
    else:
        t_start = 8300
        t_end = t_start + int(900 * doubling_time_lys)
    
    plt.figure()# figsize=(8, 6))
    for j in [1, 2, 0]:
        k_list = [1, 2, 3]  # k_list = range(N_r)
        for k in k_list:
            if phage_mask[j, k]:
                if k_list == [1, 2, 3]:
                    k_mod = [1, 2, 3]
                else:
                    k_mod = [1, 2, 3, 4]
    
                label = ''
                plt.plot(t_vec[t_start:t_end], 
                         p_t[j, k, t_start:t_end] / l_t[0, j, k, t_start:t_end], 
                         linewidth=4 - 2 * (j > 0), 
                         linestyle=linestyles[r1_jk[j, k]], 
                         color=colors[j],
                         label=label, 
                         alpha=1 * 0.55**(j))
    plt.yscale('log')
    plt.xlabel('Time (generations)')
    plt.ylabel('Population ratio')
    plt.yticks([1e-6, 1e-3, 1e0, 1e3, 1e6])
    plt.ylim([2e-7, 5e7])
    plt.xticks(np.arange(t_start/10, (t_end + 1)/10, 10 * doubling_time_lys), [])
    ax = plt.gca()
    scalebar = AnchoredSizeBar(ax.transData,
                               10 * doubling_time_lys, '10', loc='lower right', 
                               pad=0.5,
                               color='black',
                               frameon=False,
                               label_top=True
                               )
    ax.add_artist(scalebar)
    
    # Make legend labels
    j = 0
    k_list = [1, 2, 3]  # k_list = range(N_r)
    for k in k_list:
        if phage_mask[j, k]:
            if k_list == [1, 2, 3, 0]:
                k_mod = [4, 1, 2, 3]
            else:
                k_mod = [1, 2, 3, 4]
            # label = (r'$P_{{c,{0}}} / L_{{c, {0}}}$'.format(k_mod[k]))
            # if r1_jk[j, k] > 0.:
            #     label += '; ' + r'$f=$' + "{:.1f}".format(r1_jk[j, k])            
            label = (r'$P_{{c,{0}}} / L_{{c,{0}}}$'.format(r1_jk[j, k]))
            
            plt.plot([t_vec[t_end]] * 2, [1e-8] * 2, color='k', 
                     linestyle=linestyles[r1_jk[j, k]], 
                     label=label, linewidth=0
                     )
    for j in range(N_p):
        plt.plot([t_vec[t_end]] * 2, [1e-8] * 2, color=colors[j],
                 linestyle='-', linewidth=0, label=r'$c={0}$'.format(j+1))
    
    leg = plt.legend(fontsize=13, bbox_to_anchor=(1.05,1))#, handlelength=2) #loc='best') #
    for legobj in leg.legendHandles:
        if legobj.get_linestyle() != '-':
            if legobj.get_linestyle() == '--':
                legobj.set_linewidth(2.5)  # 2.
            else:
                legobj.set_linewidth(2.5)
        elif legobj.get_label() == r'$L_{c}$':
            legobj.set_linewidth(1)
        elif legobj.get_label() == r'$c=2$' or legobj.get_label() == r'$c=3$':
            legobj.set_linewidth(1.5)
        else:
            legobj.set_linewidth(4)
            
    for with_pred_line in [False, True]:
        if with_pred_line:
            plt.plot(t_vec[t_start:t_end],
                     beta * lam / (delta_p #+ gamma * np.sum(l_t[0, 0, :, t_start:t_end], 0)
                                   * np.array([1 for _ in t_vec[t_start:t_end]])
                                   ), 
                     '--k')#, label=r'$b \gamma / \delta$')
    
        if save_figs:
            save_svg_and_png('bunching_frac_Np3_Nr4_pred' + fig_title_append)
    plt.show()


#%%
# =============================================================================
# Show long time coexistence of multiple species -- less fluctuations with more species
# =============================================================================

t_max = 1000
euler_method = False

N_p = 6  # number of initial phage species
N_r = 2  # number of branching ratios for each phage species
p0_rand = 1

only_one_lyt = True

if only_one_lyt:
    r1_jk = np.round(
        np.concatenate(
            (np.array([[0.2, 0.]]), 
             np.ones((N_p - 1, N_r)) * 0.2),
            0)
        , 6)
else:
    r1_jk = np.round(
        np.concatenate(
            (np.array([[0.2, 0.]]), 
             np.ones((N_p - 1, N_r)) * [0.2, 0.]),
            0)
        , 6)
    

double_lysogen_mask, phage_mask, lysogen_mask, r2_jk, r3_jk, lam_ijk, gamma_ijk, b0, p0, l0, l20 = other_vars_from_params(
    N_b, N_p, N_r, r1_jk, lam, gamma, c1, alpha_l2, p0_rand=p0_rand, p0_init=1e-3, l0_init=1e-2)
dt, save_every, t_eval, rtol, atol = dt_tol_params(euler_method, t_max, use_t_eval=True)
dt_tols = [dt, rtol, atol]

params = [alpha_b, delta_b, beta, delta_p, alpha_l, delta_l, alpha_l2, delta_l2, 
          lam_ijk, c1, c2, K, r1_jk, r2_jk, r3_jk, gamma_ijk, phage_mask, lysogen_mask, double_lysogen_mask,
          phage_die_when_infecting_lysogen, triple_lysogens_displace_double,
          only_phage_die_with_multiple_lys, log_space]

title = make_title(params, Kinf_pop_floor)

t_vec, b_t, p_t, l_t, l2_t = phage_lysogen_simulation(
    b0, p0, l0, l20, 0, t_max, Kinf_pop_floor, euler_method, dt_tols, params, save_every, t_eval)


if only_phage_die_with_multiple_lys:
    t_start = 9409
    if alpha_l == 1 and lam == 1e-4:
        t_start = 5000
else:
    t_start = 9400    
t_end = t_start + int(700 * doubling_time_lys)

plt.figure()#figsize=(8,6))
for j in range(min(N_p, len(colors))):
    for k in [1,0]: #range(N_r):
        if phage_mask[j, k]:
            plt.plot(t_vec[t_start:t_end], p_t[j, k, t_start:t_end], linewidth=4,#- 1.5*(j>0),
                     linestyle=linestyles[r1_jk[j, k]], color=colors[j],
                     label=#(r'$P_{{{0},{1}}}$'.format(j+1, k+1) + 
                           # '; ' #+ r'$r=$' + "{:.1f}".format(r1_jk[j, k])
                           # + temp_lyt[r1_jk[j, k]]), 
                           p_label(j+1, r1_jk[j, k], r1_jk[j, k], include_T_OL=False, include_r=False),
                     alpha=0.7 * 0.5**(j>0))
    plt.plot(t_vec[t_start:t_end], np.sum(l_t[0, j, :, t_start:t_end], 0), linewidth=1,
             color=colors[j], label=r'$L_{0}$'.format(j+1), alpha=0.5)
for j in range(1):  # really make the obligate lytic phage pop
    for k in range(N_r):
        if phage_mask[j, k]:
            plt.plot(t_vec[t_start:t_end], p_t[j, k, t_start:t_end], linewidth=4 - 1.5*(j>0),
                     linestyle=linestyles[r1_jk[j, k]], color=colors[j],
                     alpha=0.7 * 0.75**(j>0))
plt.yscale('log')
plt.xlabel('Time (generations)')
plt.ylabel('Population')
plt.yticks([1e0, 1e-2, 1e-4]) 
plt.ylim([5e-6, 2e1])

#plt.yticks([1e0, 1e-4, 1e-8]) # plt.yticks([1e1, 1e-2, 1e-5])
#plt.ylim([2e-10, 5e1]) # plt.ylim([1e-6, 1e2])

plt.xticks(np.arange(t_start/10, (t_end + 1)/10, 10 * doubling_time_lys), [])
ax = plt.gca()
scalebar = AnchoredSizeBar(ax.transData,
                           10 * doubling_time_lys, '10', loc='lower right', 
                           pad=0.5,
                           color='black',
                           frameon=False,
                           label_top=True
                           )
ax.add_artist(scalebar)


leg = plt.legend(fontsize=11, bbox_to_anchor=(1.05, 1)) #loc = 'lower left', ncol=6)
for legobj in leg.legendHandles:
    if legobj.get_linestyle() != '-':
        if legobj.get_linestyle() == '--':
            legobj.set_linewidth(2.5)  # 2.
        else:
            legobj.set_linewidth(2.5)

if save_figs:
    save_svg_and_png('coexistence_Np6')
plt.show()


#%%
# =============================================================================
# Show fluctuations as a function of number of species: one class has both OL and T; rest are only T
# =============================================================================

N_r = 2  # number of branching ratios for each phage species
N_ps_to_test = range(2, 7)
t_max = int(2000 * doubling_time_lys)
num_trials = 50  # index 94 of N_p==8 has lytic strain go extinct
random_r1s = False

pop_floor = 1e-40  # if Population decreases below this number, set it to 0
Kinf_pop_floor = [Kinf, pop_floor]

euler_method = False

stds_lys = dict()
stds_lyt = dict()
stds_log_lys = dict()
stds_log_lyt = dict()
min_pops_lys = dict()
min_pops_lyt = dict()
avg_pops_lys = dict()
avg_pops_lyt = dict()
avg_log_pops_lys = dict()
avg_log_pops_lyt = dict()
all_pops_lys = dict()
all_pops_lyt = dict()
all_total_pops = dict()
total_pop_std = dict()
all_total_log_pops = dict()
stds_log_lyt_lys = dict()
stds_lyt_lys = dict()
min_pops_lyt_lys = dict()
avg_pops_lyt_lys = dict()
avg_log_pops_lyt_lys = dict()
all_pops_lyt_lys = dict()
for e, N_p in enumerate(N_ps_to_test):
    stds_lys[N_p] = []
    stds_lyt[N_p] = []
    stds_log_lys[N_p] = []
    stds_log_lyt[N_p] = []
    min_pops_lys[N_p] = []
    min_pops_lyt[N_p] = []
    avg_pops_lys[N_p] = []
    avg_pops_lyt[N_p] = []
    avg_log_pops_lys[N_p] = []
    avg_log_pops_lyt[N_p] = []
    all_pops_lys[N_p] = []
    all_pops_lyt[N_p] = []
    all_total_pops[N_p] = []
    total_pop_std[N_p] = []
    all_total_log_pops[N_p] = []
    stds_log_lyt_lys[N_p] = []
    stds_lyt_lys[N_p] = []
    min_pops_lyt_lys[N_p] = []
    avg_pops_lyt_lys[N_p] = []
    avg_log_pops_lyt_lys[N_p] = []
    all_pops_lyt_lys[N_p] = []

for e2, N_p in enumerate(N_ps_to_test):
    print('Np = ' + str(N_p) + '; e2 = ' + str(e2))
    for trial in range(num_trials):
        if trial > 0 and trial % 10 == 0:
            print('trial = ' + str(trial))
        
        p0_rand = trial + 1
        np.random.seed(p0_rand)
        
        if N_r == 2:
            r1_jk = np.round(
                np.concatenate(
                    (np.array([[0.2, 0.]]), 
                     np.ones((N_p - 1, N_r)) * 0.2 * (1 - random_r1s) + 
                     np.tile(np.random.random((N_p - 1, 1)) * 0.1 + 0.15, N_r) * random_r1s,
                     ),
                    0)
                , 6)    
        else:
            r1_jk = np.round(
                np.ones((N_p, N_r)) * 0.2 * (1 - random_r1s) + 
                np.tile(np.random.random((N_p, 1)) * 0.1 + 0.15, N_r) * random_r1s,
                6)    
        
        double_lysogen_mask, phage_mask, lysogen_mask, r2_jk, r3_jk, lam_ijk, gamma_ijk, b0, p0, l0, l20 = other_vars_from_params(
            N_b, N_p, N_r, r1_jk, lam, gamma, c1, alpha_l2, p0_rand=p0_rand, p0_init=1e-3, l0_init=1e-2)
        dt, save_every, t_eval, rtol, atol = dt_tol_params(euler_method, t_max, use_t_eval=True)
        dt_tols = [dt, 1e-3, 1e-7]  # [dt, rtol, atol]
        
        # Add a little heterogeneity to gamma and lambda
        gamma_var = 0 #1e-1  # how much variability to set for gammas
        gamma_ijk *= (1 + gamma_var * (-1 + 2 * np.random.random(gamma_ijk.shape)))
        lam_log10_var = 0 #1e-1
        lam_ijk *= 10**(lam_log10_var * (-1 + 2 * np.random.random(lam_ijk.shape)))
        r1_var = 0 #5e-1
        r1_jk *= (1 + r1_var * (-1 + 2 * np.random.random(r1_jk.shape)))

        
        params = [alpha_b, delta_b, beta, delta_p, alpha_l, delta_l, alpha_l2, delta_l2, 
                  lam_ijk, c1, c2, K, r1_jk, r2_jk, r3_jk, gamma_ijk, phage_mask, lysogen_mask, double_lysogen_mask,
                  phage_die_when_infecting_lysogen, triple_lysogens_displace_double,
                  only_phage_die_with_multiple_lys, log_space]
        
        t_vec, b_t, p_t, l_t, l2_t = phage_lysogen_simulation(
            b0, p0, l0, l20, 0, t_max, Kinf_pop_floor, euler_method, dt_tols, params, save_every, t_eval)
        
        t_start = np.argmin(np.abs(t_vec - t_max/2))  # (t_max - t_max/2)))  # t_max/2))) #400
        t_end = len(t_vec)
        
        stds_log_lys[N_p] += list(np.std(np.log(p_t[1 - (N_r==1):, 0, t_start:t_end]), axis=1))  # the first temperate phage is different
        stds_lys[N_p] += list(np.std(p_t[1 - (N_r==1):, 0, t_start:t_end], axis=1))  # the first temperate phage is different
        min_pops_lys[N_p] += list(np.min(p_t[1 - (N_r==1):, 0, t_start:t_end], axis=1))
        avg_pops_lys[N_p] += list(np.mean(p_t[1 - (N_r==1):, 0, t_start:t_end], axis=1))
        avg_log_pops_lys[N_p] += list(np.mean(np.log(p_t[1 - (N_r==1):, 0, t_start:t_end]), axis=1))
        all_pops_lys[N_p] += list(p_t[1 - (N_r==1):, 0, t_start:t_end])
        all_total_pops[N_p] += [np.mean(np.sum(p_t[:, :, t_start:t_end], axis=(0, 1)))]  #[np.sum(np.mean(p_t[:, :, t_start:t_end], axis=2))]
        total_pop_std[N_p] += [np.std(np.sum(p_t[:, :, t_start:t_end], axis=(0, 1)))]  #[np.sum(np.mean(p_t[:, :, t_start:t_end], axis=2))]
        all_total_log_pops[N_p] += [np.mean(np.log(np.sum(p_t[:, :, t_start:t_end], axis=(0, 1))))]  
        if N_r > 1:
            stds_log_lyt[N_p] += [np.std(np.log(p_t[0, 1, t_start:t_end]))]
            stds_lyt[N_p] += [np.std(p_t[0, 1, t_start:t_end])]
            min_pops_lyt[N_p] += [np.min(p_t[0, 1, t_start:t_end])]
            avg_pops_lyt[N_p] += [np.mean(p_t[0, 1, t_start:t_end])]
            avg_log_pops_lyt[N_p] += [np.mean(np.log(p_t[0, 1, t_start:t_end]))]
            all_pops_lyt[N_p] += [p_t[0, 1, t_start:t_end]]

            stds_log_lyt_lys[N_p] += [np.std(np.log(p_t[0, 0, t_start:t_end]))]
            stds_lyt_lys[N_p] += [np.std(p_t[0, 0, t_start:t_end])]
            min_pops_lyt_lys[N_p] += [np.min(p_t[0, 0, t_start:t_end])]
            avg_pops_lyt_lys[N_p] += [np.mean(p_t[0, 0, t_start:t_end])]
            avg_log_pops_lyt_lys[N_p] += [np.mean(np.log(p_t[0, 0, t_start:t_end]))]
            all_pops_lyt_lys[N_p] += [p_t[0, 0, t_start:t_end]]
        
        
        # Check plot 
        num_gen_to_plot = 600
        t_start_plot = len(t_vec) - int(num_gen_to_plot * doubling_time_lys) * 10
        t_end_plot = len(t_vec)
        plt.figure(figsize=(8,6))
        for j in range(min(N_p, 6)):
            for k in range(N_r)[::-1]:
                if phage_mask[j, k]:
                    plt.plot(t_vec[t_start_plot:t_end_plot], p_t[j, k, t_start_plot:t_end_plot], linewidth=4,#- 1.5*(j>0),
                             linestyle=linestyles[r1_jk[j, k]], 
                             color=colors[j],
                             label=p_label(j+1, r1_jk[j, k], r1_jk[j, k]), 
                             alpha=0.7 * 0.5**(j>0))
            plt.plot(t_vec[t_start_plot:t_end_plot], np.sum(l_t[0, j, :, t_start_plot:t_end_plot], 0), linewidth=1,
                     color=colors[j], label=r'$L_{0}$'.format(j+1), alpha=0.5)
        j = 0  # make j=0 results pop more
        for k in range(N_r)[::-1]:
            if phage_mask[j, k]:
                plt.plot(t_vec[t_start_plot:t_end_plot], p_t[j, k, t_start_plot:t_end_plot], linewidth=4,#- 1.5*(j>0),
                         linestyle=linestyles[r1_jk[j, k]], 
                         color=colors[j],
                         alpha=0.7 * 0.5**(j>0))
        plt.yscale('log')
        plt.xlabel('Time (generations)')
        plt.ylabel('Population')
        plt.yticks([1e0, 1e-4, 1e-8]) # plt.yticks([1e1, 1e-2, 1e-5])
        plt.ylim([2e-10, 5e1]) # plt.ylim([1e-6, 1e2])
        #plt.xticks(np.arange(t_start_plot/10, (t_end_plot + 1)/10, 10 * doubling_time_lys), [])
        plt.xticks([i * doubling_time_lys for i in 
                    [int(t_max / doubling_time_lys) + 1 - num_gen_to_plot, 
                     int(t_max / doubling_time_lys) + 1 - int(num_gen_to_plot / 2), 
                     int(t_max / doubling_time_lys) + 1]], 
                   [int(t_max / doubling_time_lys) + 1 - num_gen_to_plot, 
                    int(t_max / doubling_time_lys) + 1 - int(num_gen_to_plot / 2), 
                    int(t_max / doubling_time_lys) + 1])
        # ax = plt.gca()
        # scalebar = AnchoredSizeBar(ax.transData,
        #                            10 * doubling_time_lys, '10', loc='lower left', 
        #                            pad=0.5,
        #                            color='black',
        #                            frameon=False,
        #                            label_top=True
        #                            )
        # ax.add_artist(scalebar)
        leg = plt.legend(fontsize=13, bbox_to_anchor=(1.05, 1)) #loc='lower right') #ncol=2, 
        for legobj in leg.legendHandles:
            if legobj.get_linestyle() != '-':
                if legobj.get_linestyle() == '--':
                    legobj.set_linewidth(2.5)  # 2.
                else:
                    legobj.set_linewidth(2.5)
        if save_figs:
            save_svg_and_png('Np_' + str(N_p) + '_trial_' + str(trial), trials=True)
        plt.show()




# Make data into DataFrame
all_N_ps_orig = flatten([[N_p] * (N_p - 1 + (N_r == 1)) * num_trials for N_p in N_ps_to_test])
all_stds_log_lys_orig = flatten([stds_log_lys[N_p] for N_p in N_ps_to_test])
all_stds_lys_orig = flatten([stds_lys[N_p] for N_p in N_ps_to_test])
all_min_pops_lys_orig = flatten([min_pops_lys[N_p] for N_p in N_ps_to_test])
all_avg_pops_lys_orig = flatten([avg_pops_lys[N_p] for N_p in N_ps_to_test])
all_avg_log_pops_lys_orig = flatten([avg_log_pops_lys[N_p] for N_p in N_ps_to_test])
all_all_pops_lys_orig = flatten([all_pops_lys[N_p] for N_p in N_ps_to_test])

all_N_ps_lyt_orig = flatten([[N_p] * num_trials for N_p in N_ps_to_test])
all_total_pops_orig = flatten([all_total_pops[N_p] for N_p in N_ps_to_test])
total_pop_std_orig = flatten([total_pop_std[N_p] for N_p in N_ps_to_test])
all_total_log_pops_orig = flatten([all_total_log_pops[N_p] for N_p in N_ps_to_test])

if N_r > 1:
    all_stds_log_lyt_orig = flatten([stds_log_lyt[N_p] for N_p in N_ps_to_test])
    all_stds_lyt_orig = flatten([stds_lyt[N_p] for N_p in N_ps_to_test])
    all_min_pops_lyt_orig = flatten([min_pops_lyt[N_p] for N_p in N_ps_to_test])
    all_avg_pops_lyt_orig = flatten([avg_pops_lyt[N_p] for N_p in N_ps_to_test])
    all_avg_log_pops_lyt_orig = flatten([avg_log_pops_lyt[N_p] for N_p in N_ps_to_test])
    all_all_pops_lyt_orig = flatten([all_pops_lyt[N_p] for N_p in N_ps_to_test])
    all_stds_log_lyt_lys_orig = flatten([stds_log_lyt_lys[N_p] for N_p in N_ps_to_test])
    all_stds_lyt_lys_orig = flatten([stds_lyt_lys[N_p] for N_p in N_ps_to_test])
    all_min_pops_lyt_lys_orig = flatten([min_pops_lyt_lys[N_p] for N_p in N_ps_to_test])
    all_avg_pops_lyt_lys_orig = flatten([avg_pops_lyt_lys[N_p] for N_p in N_ps_to_test])
    all_avg_log_pops_lyt_lys_orig = flatten([avg_log_pops_lyt_lys[N_p] for N_p in N_ps_to_test])
    all_all_pops_lyt_lys_orig = flatten([all_pops_lyt_lys[N_p] for N_p in N_ps_to_test])

remove_zeros = N_r > 1
if remove_zeros: # Remove trajectories that included exctinction
    all_N_ps = [all_N_ps_orig[e] for e in range(len(all_N_ps_orig)) 
                if all_min_pops_lys_orig[e] > 0]
    all_stds_log_lys = [all_stds_log_lys_orig[e] for e in range(len(all_N_ps_orig)) 
                if all_min_pops_lys_orig[e] > 0]
    all_stds_lys = [all_stds_lys_orig[e] for e in range(len(all_N_ps_orig)) 
                if all_min_pops_lys_orig[e] > 0]
    all_min_pops_lys = [all_min_pops_lys_orig[e] for e in range(len(all_N_ps_orig)) 
                if all_min_pops_lys_orig[e] > 0]
    all_avg_pops_lys = [all_avg_pops_lys_orig[e] for e in range(len(all_N_ps_orig)) 
                if all_min_pops_lys_orig[e] > 0]
    all_avg_log_pops_lys = [all_avg_log_pops_lys_orig[e] for e in range(len(all_N_ps_orig)) 
                if all_min_pops_lys_orig[e] > 0]
    all_all_pops_lys = [all_all_pops_lys_orig[e] for e in range(len(all_N_ps_orig)) 
                if all_min_pops_lys_orig[e] > 0]
    all_all_total_pops = [all_total_pops_orig[e] for e in range(len(all_N_ps_lyt_orig)) 
                if all_min_pops_lyt_orig[e] > 0 and all_min_pops_lyt_lys_orig[e] > 0]
    all_total_pops_std = [total_pop_std_orig[e] for e in range(len(all_N_ps_lyt_orig)) 
                if all_min_pops_lyt_orig[e] > 0 and all_min_pops_lyt_lys_orig[e] > 0]
    all_all_total_log_pops = [all_total_log_pops_orig[e] for e in range(len(all_N_ps_lyt_orig)) 
                if all_min_pops_lyt_orig[e] > 0 and all_min_pops_lyt_lys_orig[e] > 0]
    all_N_ps_lyt = [all_N_ps_lyt_orig[e] for e in range(len(all_N_ps_lyt_orig)) 
                if all_min_pops_lyt_orig[e] > 0 and all_min_pops_lyt_lys_orig[e] > 0]
    if N_r > 1:
        all_stds_log_lyt = [all_stds_log_lyt_orig[e] for e in range(len(all_N_ps_lyt_orig)) 
                    if  all_min_pops_lyt_orig[e] > 0 and all_min_pops_lyt_lys_orig[e] > 0]
        all_stds_lyt = [all_stds_lyt_orig[e] for e in range(len(all_N_ps_lyt_orig)) 
                    if  all_min_pops_lyt_orig[e] > 0 and all_min_pops_lyt_lys_orig[e] > 0]
        all_min_pops_lyt = [all_min_pops_lyt_orig[e] for e in range(len(all_N_ps_lyt_orig)) 
                    if all_min_pops_lyt_orig[e] > 0 and all_min_pops_lyt_lys_orig[e] > 0]
        all_avg_pops_lyt = [all_avg_pops_lyt_orig[e] for e in range(len(all_N_ps_lyt_orig)) 
                    if all_min_pops_lyt_orig[e] > 0 and all_min_pops_lyt_lys_orig[e] > 0]
        all_avg_log_pops_lyt = [all_avg_log_pops_lyt_orig[e] for e in range(len(all_N_ps_lyt_orig)) 
                    if all_min_pops_lyt_orig[e] > 0 and all_min_pops_lyt_lys_orig[e] > 0]
        all_all_pops_lyt = [all_all_pops_lyt_orig[e] for e in range(len(all_N_ps_lyt_orig)) 
                    if all_min_pops_lyt_orig[e] > 0 and all_min_pops_lyt_lys_orig[e] > 0]
        all_stds_log_lyt_lys = [all_stds_log_lyt_lys_orig[e] for e in range(len(all_N_ps_lyt_orig)) 
                    if  all_min_pops_lyt_orig[e] > 0 and all_min_pops_lyt_lys_orig[e] > 0]
        all_stds_lyt_lys = [all_stds_lyt_lys_orig[e] for e in range(len(all_N_ps_lyt_orig)) 
                    if  all_min_pops_lyt_orig[e] > 0 and all_min_pops_lyt_lys_orig[e] > 0]
        all_min_pops_lyt_lys = [all_min_pops_lyt_lys_orig[e] for e in range(len(all_N_ps_lyt_orig)) 
                    if all_min_pops_lyt_orig[e] > 0 and all_min_pops_lyt_lys_orig[e] > 0]
        all_avg_pops_lyt_lys = [all_avg_pops_lyt_lys_orig[e] for e in range(len(all_N_ps_lyt_orig)) 
                    if all_min_pops_lyt_orig[e] > 0 and all_min_pops_lyt_lys_orig[e] > 0]
        all_avg_log_pops_lyt_lys = [all_avg_log_pops_lyt_lys_orig[e] for e in range(len(all_N_ps_lyt_orig)) 
                    if all_min_pops_lyt_orig[e] > 0 and all_min_pops_lyt_lys_orig[e] > 0]
        all_all_pops_lyt_lys = [all_all_pops_lyt_lys_orig[e] for e in range(len(all_N_ps_lyt_orig)) 
                    if all_min_pops_lyt_orig[e] > 0 and all_min_pops_lyt_lys_orig[e] > 0]


else:
    all_N_ps = all_N_ps_orig
    all_stds_log_lys = all_stds_log_lys_orig
    all_stds_lys = all_stds_lys_orig
    all_min_pops_lys = all_min_pops_lys_orig
    all_avg_pops_lys = all_avg_pops_lys_orig
    all_avg_log_pops_lys = all_avg_log_pops_lys_orig
    all_all_pops_lys = all_all_pops_lys_orig
    all_all_total_pops = all_total_pops_orig
    all_total_pops_std = total_pop_std_orig
    all_all_total_log_pops = all_total_log_pops_orig
    all_N_ps_lyt = all_N_ps_lyt_orig
    if N_r > 1:
        all_avg_pops_lyt = all_avg_pops_lyt_orig
        all_avg_log_pops_lyt = all_avg_log_pops_lyt_orig
        all_all_pops_lyt = all_all_pops_lyt_orig
        all_stds_lyt = all_stds_lyt_orig
        all_stds_log_lyt = all_stds_log_lyt_orig
        all_min_pops_lyt = all_min_pops_lyt_orig
        all_avg_pops_lyt_lys = all_avg_pops_lyt_lys_orig
        all_avg_log_pops_lyt_lys = all_avg_log_pops_lyt_lys_orig
        all_all_pops_lyt_lys = all_all_pops_lyt_lys_orig
        all_stds_lyt_lys = all_stds_lyt_lys_orig
        all_stds_log_lyt_lys = all_stds_log_lyt_lys_orig
        all_min_pops_lyt_lys = all_min_pops_lyt_lys_orig


std_offset = -2.5
min_pops_offset = -20

std_data_lys = pd.DataFrame({'N_p': all_N_ps, 
                             'std_log_lys': all_stds_log_lys, 
                             'std_lys': all_stds_lys, 
                             'min_pop_lys': all_min_pops_lys, 
                             'log_std_lys': np.log(all_stds_lys), 
                             'log_min_pop_lys': np.log(all_min_pops_lys),
                             'log_std_lys_off': np.log(all_stds_lys) - std_offset,
                             'log_min_pop_lys_off': np.log(all_min_pops_lys) - min_pops_offset,
                             'avg_pops_lys': all_avg_pops_lys, 
                             'min_over_avg_pop_lys': [i/j for i, j in zip(all_min_pops_lys, all_avg_pops_lys)], 
                             'log_min_over_avg_pop_lys': [np.log(i/j) for i, j in zip(all_min_pops_lys, all_avg_pops_lys)], 
                             'avg_cov_lys': [i/j for i, j in zip(all_stds_lys, all_avg_pops_lys)], 
                             'avg_log_pops_lys': all_avg_log_pops_lys, 
                             'avg_cov_log_lys': [i/j for i, j in zip(all_stds_log_lys, all_avg_log_pops_lys)], 
                             #'all_pops_lys': all_all_pops_lys, 
                             })
if N_r > 1:
    std_data_lyt = pd.DataFrame({'N_p': all_N_ps_lyt, 
                                 'std_log_lyt': all_stds_log_lyt, 
                                 'std_lyt': all_stds_lyt, 
                                 'min_pop_lyt': all_min_pops_lyt, 
                                 'log_std_lyt': np.log(all_stds_lyt), 
                                 'log_min_pop_lyt': np.log(all_min_pops_lyt),
                                 'log_std_lyt_off': np.log(all_stds_lyt) - std_offset,
                                 'log_min_pop_lyt_off': np.log(all_min_pops_lyt) - min_pops_offset,
                                 'avg_pops_lyt': all_avg_pops_lyt, 
                                 'min_over_avg_pop_lyt': [i/j for i, j in zip(all_min_pops_lyt, all_avg_pops_lyt)], 
                                 'log_min_over_avg_pop_lyt': [np.log(i/j) for i, j in zip(all_min_pops_lyt, all_avg_pops_lyt)], 
                                 'avg_cov_lyt': [i/j for i, j in zip(all_stds_lyt, all_avg_pops_lyt)], 
                                 'avg_cov_log_lyt': [i/j for i, j in zip(all_stds_log_lyt, all_avg_log_pops_lyt)], 
                                 'avg_log_pops_lyt': all_avg_log_pops_lyt, 
                                 'std_log_lyt_lys': all_stds_log_lyt_lys, 
                                 'std_lyt_lys': all_stds_lyt_lys, 
                                 'min_pop_lyt_lys': all_min_pops_lyt_lys, 
                                 'log_std_lyt_lys': np.log(all_stds_lyt_lys), 
                                 'log_min_pop_lyt_lys': np.log(all_min_pops_lyt_lys),
                                 'log_std_lyt_lys_off': np.log(all_stds_lyt_lys) - std_offset,
                                 'log_min_pop_lyt_lys_off': np.log(all_min_pops_lyt_lys) - min_pops_offset,
                                 'avg_pops_lyt_lys': all_avg_pops_lyt_lys, 
                                 'avg_pops_lyt_plus_lyt_lys': [i+j for i, j in zip(all_avg_pops_lyt, all_avg_pops_lyt_lys)], 
                                 'min_over_avg_pop_lyt_lys': [i/j for i, j in zip(all_min_pops_lyt_lys, all_avg_pops_lyt_lys)], 
                                 'log_min_over_avg_pop_lyt_lys': [np.log(i/j) for i, j in zip(all_min_pops_lyt_lys, all_avg_pops_lyt_lys)], 
                                 'avg_cov_lyt_lys': [i/j for i, j in zip(all_stds_lyt_lys, all_avg_pops_lyt_lys)], 
                                 'avg_log_pops_lyt_lys': all_avg_log_pops_lyt_lys, 
                                 'avg_cov_log_lyt_lys': [i/j for i, j in zip(all_stds_log_lyt_lys, all_avg_log_pops_lyt_lys)], 
                                 'all_total_pops': all_all_total_pops, 
                                 'all_pops_std': all_total_pops_std, 
                                 'all_cov': [i/j for i, j in zip(all_total_pops_std, all_all_total_pops)], 
                                 'all_total_log_pops': all_all_total_log_pops, 
                                 # 'all_pops_lyt': all_all_pops_lyt, 
                                 })
else:
    std_data_lyt = pd.DataFrame({'N_p': all_N_ps_lyt, 
                                 'all_total_pops': all_all_total_pops, 
                                 'all_pops_std': all_total_pops_std, 
                                 'all_cov': [i/j for i, j in zip(all_total_pops_std, all_all_total_pops)], 
                                 'all_total_log_pops': all_all_total_log_pops, 
                                 })
# =============================================================================
# Plot population means
# =============================================================================
# plt.figure()
# # plt.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i]['all_total_pops']) for i in N_ps_to_test], 
# #          color=colors[3], linestyle=linestyles[0.], 
# #          linewidth=3.5, label=temp_lyt[0.])#r'$r=0.0$')  # marker='o', 
# plt.plot([e for e, Np in enumerate(N_ps_to_test)], 
#          [(Np / (Np - 1)) * (alpha_l - lam) / (gamma * (1 - 0.2)) for e, Np in enumerate(N_ps_to_test)],
#          color='k', linestyle='--', label='Prediction')
# sns.swarmplot(data=std_data_lyt, x='N_p', y='all_total_pops',
#               color=dark_colors[3], alpha=1, size=25 / max(N_ps_to_test) / np.log10(num_trials))
# plt.xlabel(r'$N_c$')
# plt.ylabel(r'$\langle \sum_{c,f} P_{cf} \rangle_t$')
# plt.legend(fontsize=15)#, loc='lower left')#center right
# plt.xticks(range(len(N_ps_to_test))[:: int(len(N_ps_to_test)/4)], N_ps_to_test[:: int(len(N_ps_to_test)/4)])
# if save_figs:
#     save_svg_and_png('total_pop_Np_Nr_' + str(N_r))
# plt.show()


plt.figure()
plt.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i]['all_total_log_pops']) 
          for i in N_ps_to_test], 
         color=colors[0], linestyle=':', 
         linewidth=3.5, label='Total population')#r'$r=0.0$')  # marker='o', 
sns.swarmplot(data=std_data_lyt, x='N_p', y='all_total_log_pops',
              color=dark_colors[0], alpha=0.5, size=20 / max(N_ps_to_test) / np.log10(num_trials))
if N_r > 1:
    plt.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i]['avg_log_pops_lyt']) 
              for i in N_ps_to_test], 
             color=colors[3], linestyle=linestyles[0.], 
             linewidth=3.5, label='Obligate lytic')#r'$r=0.0$')  # marker='o', 
    sns.swarmplot(data=std_data_lyt, x='N_p', y='avg_log_pops_lyt',
                  color=dark_colors[3], alpha=0.5, size=20 / max(N_ps_to_test) / np.log10(num_trials))
    plt.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i]['avg_log_pops_lyt_lys']) 
              for i in N_ps_to_test], 
             color=colors[0], linestyle=linestyles[0.2], 
             linewidth=3.5, label='Temperate cousin')#r'$r=0.0$')  # marker='o', 
    sns.swarmplot(data=std_data_lyt, x='N_p', y='avg_log_pops_lyt_lys',
                  color=dark_colors[0], alpha=0.5, size=20 / max(N_ps_to_test) / np.log10(num_trials))
plt.plot([np.mean(std_data_lys[std_data_lys['N_p'] == i]['avg_log_pops_lys']) 
          for i in N_ps_to_test], 
         color=colors[4], linestyle=linestyles[0.2], 
         linewidth=3.5, label='Temperate')#r'$r=0.0$')  # marker='o', 
sns.swarmplot(data=std_data_lys, x='N_p', y='avg_log_pops_lys',
              color=dark_colors[4], alpha=0.5, size=20 / max(N_ps_to_test) / np.log10(num_trials))
plt.xlabel(r'$N_c$')
plt.ylabel(r'$\langle \log \sum_{c,f} P_{cf} \rangle_t$')
plt.xticks(range(len(N_ps_to_test))[:: int(len(N_ps_to_test)/4)], N_ps_to_test[:: int(len(N_ps_to_test)/4)])
plt.legend(fontsize=15)#, loc='lower left')#center right
if save_figs:
    save_svg_and_png('total_log_pop_Np_Nr_' + str(N_r), trials=True)
plt.show()


for ylog in [True, False]:
    plt.figure()
    # plt.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i]['all_total_pops']) for i in N_ps_to_test], 
    #          color=colors[3], linestyle=linestyles[0.], 
    #          linewidth=3.5, label=temp_lyt[0.])#r'$r=0.0$')  # marker='o', 
    if N_r > 1:
        plt.plot([e for e, Np in enumerate(N_ps_to_test)], 
                 [(1 / (Np - 1)) * (alpha_l - lam) / (gamma * (1 - 0.)) for e, Np in enumerate(N_ps_to_test)],
                 color=colors[0], linestyle='--', label=r'$P_{1, 0.0}$' + ' steady state')
        plt.plot([e for e, Np in enumerate(N_ps_to_test)], 
                 [beta * lam * ((beta + 1) * (1 + (Np - 2) * 0.2) * alpha_l + (-1 - Np * beta + (2 - Np + beta) * 0.2) * lam) / 
                  ((1 - 0.2) * gamma * ( (beta + 1) * (1 + (Np - 1) * (beta + 1) * 0.2) * alpha_l - 
                                        (1 + Np * beta + (Np - 1) * (beta + 1)**2 * 0.2) * lam ))
                     for e, Np in enumerate(N_ps_to_test)],
                 color=dark_colors[0], linestyle='--', label=r'$P_{1, 0.2}$' + ' steady state')
        sns.stripplot(data=std_data_lyt, x='N_p', y='avg_pops_lyt',
                      color=colors[0], alpha=1, size=18 / max(N_ps_to_test) / np.log10(num_trials))
        sns.stripplot(data=std_data_lyt, x='N_p', y='avg_pops_lyt_lys',
                      color=dark_colors[0], alpha=1, size=18 / max(N_ps_to_test) / np.log10(num_trials))
        # sns.stripplot(data=std_data_lyt, x='N_p', y='avg_pops_lyt_plus_lyt_lys',
        #               color=dark_colors[2], alpha=0.5, size=18 / max(N_ps_to_test) / np.log10(num_trials))
    plt.plot([e for e, Np in enumerate(N_ps_to_test)], 
             [(1 / (Np - 1)) * (alpha_l - lam) / (gamma * (1 - 0.2)) for e, Np in enumerate(N_ps_to_test)],
             color=colors[4], linestyle='-', label=r'$P_{c \neq 1, 0.2}$' + ' steady state')
    sns.stripplot(data=std_data_lys, x='N_p', y='avg_pops_lys',
                  color=dark_colors[4], alpha=1, size=18 / max(N_ps_to_test) / np.log10(num_trials))
    plt.xlabel(r'$N_c$')
    plt.ylabel(r'$\langle \sum_{f} P_{cf} \rangle_t$')
    plt.legend(fontsize=15)#, loc='lower left')#center right
    #plt.yticks([0, 0.5, 1.])
    if ylog:
        plt.yscale('log')
    plt.xticks(range(len(N_ps_to_test))[:: int(len(N_ps_to_test)/4)], N_ps_to_test[:: int(len(N_ps_to_test)/4)])
    if save_figs:
        save_svg_and_png('avg_pop_Np_Nr_' + str(N_r), trials=True)
    plt.show()


plt.figure()
plt.plot([2] * 2, [-19] * 2, color=colors[0], linestyle='-', linewidth=8, alpha=0.7, label=r'$P_{1, 0.0}$')
plt.plot([2] * 2, [-19] * 2, color=dark_colors[0], linestyle='-', linewidth=8, alpha=0.7, label=r'$P_{1, 0.2}$')
if random_r1s:
    plt.plot([2] * 2, [-19] * 2, color=dark_colors[1], linestyle='-', linewidth=8, alpha=0.7, label=r'$P_{c\neq 1, T}$')
else:
    plt.plot([2] * 2, [-19] * 2, color=dark_colors[1], linestyle='-', linewidth=8, alpha=0.7, label=r'$P_{c\neq 1, 0.2}$')
plt.plot([2] * 2, [-19] * 2, color=colors[2], linestyle='-', linewidth=8, alpha=0.7, label=r'$\sum_{c,f}\ P_{c,f}$')

# plt.plot([np.mean(std_data_lys[std_data_lys['N_p'] == i]['avg_cov_lys']) for i in N_ps_to_test], 
#           color=colors[4], linestyle=linestyles[0.2], 
#           linewidth=3.5, label='Temperate')#r'$r=0.0$')  # marker='o', 
# sns.swarmplot(data=std_data_lys, x='N_p', y='avg_cov_lys',
#               color=dark_colors[4], alpha=1, size=25 / max(N_ps_to_test) / np.log10(num_trials))
ax_viol = sns.violinplot(data=std_data_lys, x='N_p', y='avg_cov_lys',
                         color=dark_colors[1], inner=None, linewidth=0, width=0.8, scale='width')
sns.stripplot(data=std_data_lys, x='N_p', y='avg_cov_lys',  
              color=dark_colors[1], alpha=0.5, size=25 / max(N_ps_to_test) / np.log10(num_trials))

if N_r > 1:
    # plt.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i]['avg_cov_lyt_lys']) for i in N_ps_to_test], 
    #           color=colors[0], linestyle=linestyles[0.2], 
    #           linewidth=3.5, label='Temperate cousin')#r'$r=0.0$')  # marker='o', 
    # sns.swarmplot(data=std_data_lyt, x='N_p', y='avg_cov_lyt_lys',
    #               color=dark_colors[0], alpha=1, size=25 / max(N_ps_to_test) / np.log10(num_trials))
    ax_viol = sns.violinplot(data=std_data_lyt, x='N_p', y='avg_cov_lyt_lys',
                             color=dark_colors[0], inner=None, linewidth=0, width=0.8, scale='width')
    sns.stripplot(data=std_data_lyt, x='N_p', y='avg_cov_lyt_lys',  
                  color=dark_colors[0], alpha=0.5, size=25 / max(N_ps_to_test) / np.log10(num_trials))

    # plt.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i]['avg_cov_lyt']) for i in N_ps_to_test], 
    #           color=colors[3], linestyle=linestyles[0.], 
    #           linewidth=3.5, label='Obligate lytic')#r'$r=0.0$')  # marker='o', 
    # sns.swarmplot(data=std_data_lyt, x='N_p', y='avg_cov_lyt',
    #               color=dark_colors[3], alpha=1, size=25 / max(N_ps_to_test) / np.log10(num_trials))
    ax_viol = sns.violinplot(data=std_data_lyt, x='N_p', y='avg_cov_lyt',
                             color=colors[0], inner=None, linewidth=0, width=0.8, scale='width')
    sns.stripplot(data=std_data_lyt, x='N_p', y='avg_cov_lyt',  
                  color=colors[0], alpha=0.5, size=25 / max(N_ps_to_test) / np.log10(num_trials))    

# plt.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i]['all_cov']) for i in N_ps_to_test], 
#           color=colors[3], linestyle=':', 
#           linewidth=3.5, label='Total population')#r'$r=0.0$')  # marker='o', 
# sns.swarmplot(data=std_data_lyt, x='N_p', y='all_cov',
#               color=colors[3], alpha=1, size=25 / max(N_ps_to_test) / np.log10(num_trials))
ax_viol = sns.violinplot(data=std_data_lyt, x='N_p', y='all_cov',
                         color=colors[2], inner=None, linewidth=0, width=0.8, scale='width')
sns.stripplot(data=std_data_lyt, x='N_p', y='all_cov',  
              color=colors[2], alpha=0.5, size=25 / max(N_ps_to_test) / np.log10(num_trials))    

for violin in ax_viol.collections:  # * 2[::2] # If not inner=None  
    violin.set_alpha(0.7)  

plt.plot(np.arange(1 + len(N_ps_to_test)) - 0.5, [1] * (1 + len(N_ps_to_test)), '--k')
plt.xlabel(r'$N_c$')
plt.ylabel('COV') #r'$\langle \sum_{c,f} P_{cf} \rangle_t$')
plt.legend(fontsize=15)#, loc='lower left')#center right
plt.xticks(range(len(N_ps_to_test))[:: int(len(N_ps_to_test)/4)], N_ps_to_test[:: int(len(N_ps_to_test)/4)])
plt.ylim([-1.23, 11.38 - (2 not in N_ps_to_test)])
if save_figs:
    save_svg_and_png('cov_Np_Nr_' + str(N_r), trials=True)
plt.show()


plt.figure()
if N_r > 1:
    plt.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i]['avg_cov_log_lyt']) for i in N_ps_to_test], 
              color=colors[3], linestyle=linestyles[0.], 
              linewidth=3.5, label='Obligate lytic')#r'$r=0.0$')  # marker='o', 
    sns.swarmplot(data=std_data_lyt, x='N_p', y='avg_cov_log_lyt',
                  color=dark_colors[3], alpha=1, size=25 / max(N_ps_to_test) / np.log10(num_trials))
    plt.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i]['avg_cov_log_lyt_lys']) for i in N_ps_to_test], 
              color=colors[0], linestyle=linestyles[0.2], 
              linewidth=3.5, label='Temperate cousin')#r'$r=0.0$')  # marker='o', 
    sns.swarmplot(data=std_data_lyt, x='N_p', y='avg_cov_log_lyt_lys',
                  color=dark_colors[0], alpha=1, size=25 / max(N_ps_to_test) / np.log10(num_trials))
plt.plot([np.mean(std_data_lys[std_data_lys['N_p'] == i]['avg_cov_log_lys']) for i in N_ps_to_test], 
          color=colors[4], linestyle=linestyles[0.2], 
          linewidth=3.5, label='Temperate')#r'$r=0.0$')  # marker='o', 
sns.swarmplot(data=std_data_lys, x='N_p', y='avg_cov_log_lys',
              color=dark_colors[4], alpha=1, size=25 / max(N_ps_to_test) / np.log10(num_trials))
# plt.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i]['all_cov']) for i in N_ps_to_test], 
#           color=colors[0], linestyle=':', 
#           linewidth=3.5, label='Total population')#r'$r=0.0$')  # marker='o', 
# sns.swarmplot(data=std_data_lyt, x='N_p', y='all_cov',
#               color=dark_colors[0], alpha=1, size=25 / max(N_ps_to_test) / np.log10(num_trials))
plt.xlabel(r'$N_c$')
plt.ylabel('COV of log populations') #r'$\langle \sum_{c,f} P_{cf} \rangle_t$')
plt.legend(fontsize=15)#, loc='lower left')#center right
plt.xticks(range(len(N_ps_to_test))[:: int(len(N_ps_to_test)/4)], N_ps_to_test[:: int(len(N_ps_to_test)/4)])
if save_figs:
    save_svg_and_png('cov_log_Np_Nr_' + str(N_r), trials=True)
plt.show()


plt.figure()
if N_r > 1:
    plt.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i]['avg_cov_lyt']) for i in N_ps_to_test], 
              color=colors[3], linestyle=linestyles[0.], 
              linewidth=3.5, label='Obligate lytic')#r'$r=0.0$')  # marker='o', 
    sns.swarmplot(data=std_data_lyt, x='N_p', y='avg_cov_lyt',
                  color=dark_colors[3], alpha=1, size=25 / max(N_ps_to_test) / np.log10(num_trials))
    plt.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i]['avg_cov_lyt_lys']) for i in N_ps_to_test], 
              color=colors[0], linestyle=linestyles[0.2], 
              linewidth=3.5, label='Temperate cousin')#r'$r=0.0$')  # marker='o', 
    sns.swarmplot(data=std_data_lyt, x='N_p', y='avg_cov_lyt_lys',
                  color=dark_colors[0], alpha=1, size=25 / max(N_ps_to_test) / np.log10(num_trials))
plt.plot([np.mean(std_data_lys[std_data_lys['N_p'] == i]['avg_cov_lys']) for i in N_ps_to_test], 
          color=colors[4], linestyle=linestyles[0.2], 
          linewidth=3.5, label='Temperate')#r'$r=0.0$')  # marker='o', 
sns.swarmplot(data=std_data_lys, x='N_p', y='avg_cov_lys',
              color=dark_colors[4], alpha=1, size=25 / max(N_ps_to_test) / np.log10(num_trials))
plt.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i]['all_cov']) for i in N_ps_to_test], 
          color=colors[0], linestyle=':', 
          linewidth=3.5, label='Total population')#r'$r=0.0$')  # marker='o', 
sns.swarmplot(data=std_data_lyt, x='N_p', y='all_cov',
              color=dark_colors[0], alpha=1, size=25 / max(N_ps_to_test) / np.log10(num_trials))
plt.xlabel(r'$N_c$')
plt.ylabel('COV') #r'$\langle \sum_{c,f} P_{cf} \rangle_t$')
plt.legend(fontsize=15)#, loc='lower left')#center right
plt.xticks(range(len(N_ps_to_test))[:: int(len(N_ps_to_test)/4)], N_ps_to_test[:: int(len(N_ps_to_test)/4)])
if save_figs:
    save_svg_and_png('cov_Np_Nr_' + str(N_r), trials=True)
plt.show()


sigma_labels = [1, 0, -1]  # y-axis ticks in log10 space. Need to be specified if range doesn't span enough OOM

log_sigma = True
if log_sigma:
    std_name = 'log_std_'
else:
    std_name = 'std_'# 'std_log_' #'std_'

plt.figure()
if N_r > 1:
    plt.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i][std_name + 'lyt']) for i in N_ps_to_test], 
             color=colors[3], linestyle=linestyles[0.], 
             linewidth=3.5, label=temp_lyt[0.])#r'$r=0.0$')  # marker='o', 
    sns.swarmplot(data=std_data_lyt, x='N_p', y=std_name + 'lyt', 
                  color=dark_colors[3], alpha=0.5, size=20 / max(N_ps_to_test) / np.log10(num_trials))
    plt.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i][std_name + 'lyt_lys']) for i in N_ps_to_test], 
             color=colors[0], linestyle=linestyles[0.2], 
             linewidth=3.5, label='Temperate cousin')#r'$r=0.0$')  # marker='o', 
    sns.swarmplot(data=std_data_lyt, x='N_p', y=std_name + 'lyt_lys', 
                  color=dark_colors[0], alpha=0.5, size=20 / max(N_ps_to_test) / np.log10(num_trials))
plt.plot([np.mean(std_data_lys[std_data_lys['N_p'] == i][std_name + 'lys']) for i in N_ps_to_test], 
         color=colors[4], linestyle=linestyles[0.2], 
         linewidth=3.5, label=temp_lyt[0.2])#r'$r=0.2$')  # marker='o',
sns.swarmplot(data=std_data_lys, x='N_p', y=std_name + 'lys', 
              color=dark_colors[4], alpha=0.5, size=20 / max(N_ps_to_test) / np.log10(num_trials))  #8C6902  # for dark yellow
# plt.plot([e for e, Np in enumerate(N_ps_to_test)], 
#          [(Np / (Np - 1)) * (alpha_l - lam) / (gamma * (1 - 0.2)) for e, Np in enumerate(N_ps_to_test)],
#          color='k', linestyle='--', label='Prediction')
plt.xlabel(r'$N_c$')
plt.ylabel(r'$\sigma_{P}$')
plt.xticks(range(len(N_ps_to_test))[:: int(len(N_ps_to_test)/4)], N_ps_to_test[:: int(len(N_ps_to_test)/4)])
if log_sigma:
    plt.yticks([np.log(10**i) for i in sigma_labels],
               [r'$10^{{{0}}}$'.format(i) for i in sigma_labels])
else:
    pass
if log_sigma:  #sigma_labels == [1, 0, -1]:
    plt.ylim([std_offset, 2.5])
plt.legend(fontsize=15)#, loc='lower left')#center right
if save_figs:
    save_svg_and_png('std_vs_Np_lyslyt_Nr_' + str(N_r), trials=True)
plt.show()



plt.figure()
if N_r > 1:
    plt.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i].log_min_pop_lyt) for i in N_ps_to_test], 
             color=colors[3], linestyle=linestyles[0.], 
             linewidth=4, label=temp_lyt[0.])#r'$r=0.0$')  # marker='o', 
    sns.swarmplot(data=std_data_lyt, x='N_p', y='log_min_pop_lyt', 
                  color=dark_colors[3], alpha=1, size=20 / max(N_ps_to_test) / np.log10(num_trials))
    plt.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i].log_min_pop_lyt_lys) for i in N_ps_to_test], 
             color=colors[0], linestyle=linestyles[0.2], 
             linewidth=4, label='Temperate cousin')#r'$r=0.0$')  # marker='o', 
    sns.swarmplot(data=std_data_lyt, x='N_p', y='log_min_pop_lyt_lys', 
                  color=dark_colors[0], alpha=1, size=20 / max(N_ps_to_test) / np.log10(num_trials))
plt.plot([np.mean(std_data_lys[std_data_lys['N_p'] == i].log_min_pop_lys) for i in N_ps_to_test], 
         color=colors[4], linestyle=linestyles[0.2], 
         linewidth=4, label=temp_lyt[0.2])#r'$r=0.2$')  # marker='o',
sns.swarmplot(data=std_data_lys, x='N_p', y='log_min_pop_lys', 
              color=dark_colors[4], alpha=1, size=20 / max(N_ps_to_test) / np.log10(num_trials))  #8C6902  # for dark yellow
plt.xlabel(r'$N_p$')
plt.ylabel(r'$min_{P}$')
plt.xticks(range(len(N_ps_to_test))[:: int(len(N_ps_to_test)/4)], N_ps_to_test[:: int(len(N_ps_to_test)/4)])
plt.yticks([np.log(10**-9), np.log(10**-6), np.log(10**-3)], 
           [r'$10^{-9}$', r'$10^{-6}$', r'$10^{-3}$'])
plt.legend(fontsize=15)#, loc='lower left')#title='Strain')
if save_figs:
    save_svg_and_png('minpop_vs_Np_lyslyt_Nr_' + str(N_r), trials=True)
plt.show()


plt.figure()
if N_r > 1:
    plt.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i].log_min_pop_lyt) for i in N_ps_to_test], 
             color=colors[3], linestyle=linestyles[0.], alpha=0.3,
             linewidth=2, label=temp_lyt[0.])#r'$r=0.0$')  # marker='o', 
    sns.violinplot(data=std_data_lyt, x='N_p', y='log_min_pop_lyt', 
                  color=dark_colors[3], inner=None)
    plt.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i].log_min_pop_lyt_lys) for i in N_ps_to_test], 
             color=colors[0], linestyle=linestyles[0.2], 
             linewidth=2, label='Temperate cousin')#r'$r=0.0$')  # marker='o', 
    sns.violinplot(data=std_data_lyt, x='N_p', y='log_min_pop_lyt_lys', 
                  color=dark_colors[0], inner=None)
plt.plot([np.mean(std_data_lys[std_data_lys['N_p'] == i].log_min_pop_lys) for i in N_ps_to_test], 
         color=colors[4], linestyle=linestyles[0.2], 
         linewidth=2, label=temp_lyt[0.2])#r'$r=0.2$')  # marker='o',
ax = sns.violinplot(data=std_data_lys, x='N_p', y='log_min_pop_lys', 
                    color=dark_colors[4], inner=None)  #8C6902  # for dark yellow
# Make the last violin plot transparent
for violin in ax.collections[-len(N_ps_to_test):]:  # * 2[::2] # If not inner=None  
    violin.set_alpha(0.3)  
plt.xlabel(r'$N_p$')
plt.ylabel(r'$min_{P}$')
plt.xticks(range(len(N_ps_to_test))[:: int(len(N_ps_to_test)/4)], N_ps_to_test[:: int(len(N_ps_to_test)/4)])
plt.yticks([np.log(10**-9), np.log(10**-6), np.log(10**-3)], 
           [r'$10^{-9}$', r'$10^{-6}$', r'$10^{-3}$'])
plt.legend(fontsize=15)#, loc='lower left')#title='Strain')
if save_figs:
    save_svg_and_png('minpop_vs_Np_lyslyt_Nr_' + str(N_r), trials=True)
plt.show()



fig, ax = plt.subplots()
plt.plot([2] * 2, [-19] * 2, color=colors[0], linestyle='-', linewidth=8, alpha=0.7, label=r'$P_{1, 0.0}$')
plt.plot([2] * 2, [-19] * 2, color=dark_colors[0], linestyle='-', linewidth=8, alpha=0.7, label=r'$P_{1, 0.2}$')
if random_r1s:
    plt.plot([2] * 2, [-19] * 2, color=dark_colors[4], linestyle='-', linewidth=8, alpha=0.7, label=r'$P_{c\neq 1, T}$')
else:
    plt.plot([2] * 2, [-19] * 2, color=dark_colors[4], linestyle='-', linewidth=8, alpha=0.7, label=r'$P_{c\neq 1, 0.2}$')
if N_r > 1:
    plt.plot([e for e, Np in enumerate(N_ps_to_test)], 
             [(1 / (Np - 1)) * (alpha_l - lam) / (gamma * (1 - 0.)) 
              # (alpha_l * (beta + 1) - lam * (1 + Np * beta)) * (alpha_l * (1 + (Np - 1) * (beta + 1) * 0.2) - lam * (1 + (Np - 1) * (beta + 0.2))) / 
              # ((Np - 1) * (beta + 1) * (1 + (Np - 1) * (beta + 1) * 0.2) * gamma * alpha_l - 
              #  (Np - 1) * gamma * lam * (1 + Np * beta + (Np - 1) * (beta + 1)**2 * 0.2))
              for e, Np in enumerate(N_ps_to_test)],
             color=colors[0], linestyle='--', label=r'$f=0.0$' + ' steady state')  #r'$P_{1, 0.0}$' + ' prediction')
    sns.stripplot(data=std_data_lyt, x='N_p', y='avg_pops_lyt',
                  color=colors[0], alpha=1, size=40 / max(N_ps_to_test) / np.log10(num_trials))
    sns.stripplot(data=std_data_lyt, x='N_p', y='avg_pops_lyt_lys',
                  color=dark_colors[0], alpha=1, size=40 / max(N_ps_to_test) / np.log10(num_trials))
    # sns.violinplot(data=std_data_lyt, x='N_p', y='avg_pops_lyt',
    #               color=colors[0], inner=None, linewidth=0, width=0.8, scale='width')
    # sns.violinplot(data=std_data_lyt, x='N_p', y='avg_pops_lyt_lys',
    #               color=dark_colors[0], inner=None, linewidth=0, width=0.8, scale='width')
plt.plot([e for e, Np in enumerate(N_ps_to_test)],
         [(1 / (Np - 1)) * (alpha_l - lam) / (gamma * (1 - 0.2)) for e, Np in enumerate(N_ps_to_test)],
         color=colors[4], linestyle='--', label=r'$f=0.2$' + ' steady state')
sns.stripplot(data=std_data_lys, x='N_p', y='avg_pops_lys',
              color=dark_colors[4], alpha=1, size=40 / max(N_ps_to_test) / np.log10(num_trials))
# sns.violinplot(data=std_data_lys, x='N_p', y='avg_pops_lys',
#               color=dark_colors[4], inner=None, linewidth=0, width=0.8, scale='width')
plt.xlabel(r'$N_c$')
plt.ylabel(r'$\langle P_{cf} \rangle$')
plt.xticks(range(len(N_ps_to_test))[:: int(len(N_ps_to_test)/4)], N_ps_to_test[:: int(len(N_ps_to_test)/4)])
plt.ylim([-0.07, 1.35])
plt.legend(fontsize=15, bbox_to_anchor=(1.63, 1))#title='Strain')

axins = ax.inset_axes([0.5, 0.58, 0.47, 0.39]) #[0.67, 0.4, 0.3, 0.3])
# axins.plot([np.mean(std_data_lys[std_data_lys['N_p'] == i].log_min_pop_lys) for i in N_ps_to_test], 
#           color=dark_colors[4], linestyle=linestyles[0.2], 
#           linewidth=2)#r'$r=0.2$')  # marker='o',
ax_viol = sns.violinplot(data=std_data_lys, x='N_p', y='log_min_pop_lys', 
                    color=dark_colors[4], inner=None, ax=axins, linewidth=0, width=0.8, scale='width') 
# sns.swarmplot(data=std_data_lys, x='N_p', y='log_min_pop_lys', 
#               color=colors[4], ax=axins, alpha=0.5, size=18 / max(N_ps_to_test) / np.log10(num_trials)) 
for violin in ax_viol.collections:  # * 2[::2] # If not inner=None  
    violin.set_alpha(0.7)  
if N_r > 1:
    # axins.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i].log_min_pop_lyt) for i in N_ps_to_test], 
    #             color=colors[0], linestyle=linestyles[0.2], alpha=0.3,
    #             linewidth=2)#r'$r=0.0$')  # marker='o', 
    sns.violinplot(data=std_data_lyt, x='N_p', y='log_min_pop_lyt', 
                  color=colors[0], inner=None, ax=axins, linewidth=0, width=0.8, scale='width')
    # sns.swarmplot(data=std_data_lyt, x='N_p', y='log_min_pop_lyt', 
    #               color=colors[3], ax=axins, alpha=0.5, size=18 / max(N_ps_to_test) / np.log10(num_trials))
    # axins.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i].log_min_pop_lyt_lys) for i in N_ps_to_test], 
    #             color=dark_colors[0], linestyle=linestyles[0.2], 
    #             linewidth=2)#r'$r=0.0$')  # marker='o', 
    sns.violinplot(data=std_data_lyt, x='N_p', y='log_min_pop_lyt_lys', 
                  color=dark_colors[0], inner=None, ax=axins, linewidth=0, width=0.8, scale='width',
                  label='$P_{1, 0.2}$')
    # sns.swarmplot(data=std_data_lyt, x='N_p', y='log_min_pop_lyt_lys', 
    #               color=colors[0], ax=axins, alpha=0.5, size=18 / max(N_ps_to_test) / np.log10(num_trials))
for violin in ax_viol.collections:  # * 2[::2] # If not inner=None  
    violin.set_alpha(0.7)  
axins.set_ylabel(r'$\min(P)$', fontsize=14)  #  / \langle P \rangle
# axins.set_yticks([np.log(10**-12), np.log(10**-6), np.log(10**0)], 
#             [r'$10^{-12}$', r'$10^{-6}$', r'$10^{0}$'], fontsize=10)
axins.set_yticks([np.log(10**-9), np.log(10**-6), np.log(10**-3)], 
            [r'$10^{-9}$', r'$10^{-6}$', r'$10^{-3}$'], fontsize=10)
# axins.set_yscale('log')
# axins.set_ylim([min_pops_offset, -3])
axins.set_xlabel(r'$N_c$', fontsize=14, labelpad=-4*(max(N_ps_to_test) > 6)) # default for labelpad=4
axins.set_xticks(range(len(N_ps_to_test))[::int(len(N_ps_to_test)/3)], 
                 [i + min(N_ps_to_test) for i in range(len(N_ps_to_test))[::int(len(N_ps_to_test)/3)]], 
                 fontsize=10)
if save_figs:
    save_svg_and_png('avgpop_minpop_vs_Np_lyslyt_2_Nr_' + str(N_r), trials=False)
    save_svg_and_png('avgpop_minpop_vs_Np_lyslyt_2_Nr_' + str(N_r), trials=True)
plt.show()



fig, ax = plt.subplots()
if N_r > 1:
    plt.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i][std_name + 'lyt']) for i in N_ps_to_test], 
             color=colors[3], linestyle=linestyles[0.], 
             linewidth=3.5, label='Obligate lytic'#temp_lyt[0.]
             )#r'$r=0.0$')  # marker='o', 
    sns.swarmplot(data=std_data_lyt, x='N_p', y=std_name + 'lyt', 
                  color=dark_colors[3], alpha=0.5, size=20 / max(N_ps_to_test) / np.log10(num_trials))
plt.plot([np.mean(std_data_lys[std_data_lys['N_p'] == i][std_name + 'lys']) for i in N_ps_to_test], 
         color=colors[4], linestyle=linestyles[0.2], 
         linewidth=3.5, label='Temperate'#temp_lyt[0.2]
         )#r'$r=0.2$')  # marker='o',
sns.swarmplot(data=std_data_lys, x='N_p', y=std_name + 'lys', 
              color=dark_colors[4], alpha=0.5, size=20 / max(N_ps_to_test) / np.log10(num_trials))  #8C6902  # for dark yellow
plt.xlabel(r'$N_c$')
plt.ylabel(r'$\sigma_{P}$')
plt.xticks(range(len(N_ps_to_test))[:: int(len(N_ps_to_test)/4)], N_ps_to_test[:: int(len(N_ps_to_test)/4)])
if log_sigma:
    plt.yticks([np.log(10**i) for i in sigma_labels],
               [r'$10^{{{0}}}$'.format(i) for i in sigma_labels])
else:
    plt.yticks([0, 2, 4, 6])
if log_sigma:  #sigma_labels == [1, 0, -1]:
    plt.ylim([std_offset, 2.5])
else:
    plt.ylim([-0.6, 6.6])
plt.legend(fontsize=15, loc='lower left')#title='Strain')

axins = ax.inset_axes([0.5, 0.58, 0.47, 0.39]) #[0.67, 0.4, 0.3, 0.3])
if N_r > 1:
    axins.plot([np.mean(std_data_lyt[std_data_lyt['N_p'] == i].log_min_pop_lyt) for i in N_ps_to_test], 
             color=colors[3], linestyle=linestyles[0.], 
             linewidth=2)
    sns.swarmplot(data=std_data_lyt, x='N_p', y='log_min_pop_lyt', 
                  color=dark_colors[3], alpha=1, size=20 / max(N_ps_to_test) / np.log10(num_trials), ax=axins)
axins.plot([np.mean(std_data_lys[std_data_lys['N_p'] == i].log_min_pop_lys) for i in N_ps_to_test], 
         color=colors[4], linestyle=linestyles[0.2], 
         linewidth=2)
sns.swarmplot(data=std_data_lys, x='N_p', y='log_min_pop_lys', 
              color=dark_colors[4], alpha=1, size=1 / np.log10(num_trials), ax=axins)  #8C6902  # for dark yellow
axins.set_ylabel(r'$\min(P)$', fontsize=14)
axins.set_yticks([np.log(10**-9), np.log(10**-6), np.log(10**-3)], 
           [r'$10^{-9}$', r'$10^{-6}$', r'$10^{-3}$'], fontsize=10)
# axins.set_ylim([min_pops_offset, -3])
axins.set_xlabel(r'$N_c$', fontsize=14, labelpad=-4) # default for labelpad=4
axins.set_xticks(range(len(N_ps_to_test))[::int(len(N_ps_to_test)/4)], 
                 [i + min(N_ps_to_test) for i in range(len(N_ps_to_test))[::int(len(N_ps_to_test)/4)]], 
                 fontsize=10)
if save_figs:
    save_svg_and_png('std_minpop_vs_Np_lyslyt_2_Nr_' + str(N_r), trials=True)
plt.show()



pop_floor = 1e-20  # if Population decreases below this number, set it to 0
Kinf_pop_floor = [Kinf, pop_floor]

#%%
# =============================================================================
# =============================================================================
# =============================================================================
# # #           Begin supplement & presentation figures
# =============================================================================
# =============================================================================
# =============================================================================
stop  # End of figures we use in paper for now

#%%
# =============================================================================
# Show that lysogenic phage can dominate
# =============================================================================

t_max = int(30 * doubling_time_lys) + 1
euler_method = False

pop_floor = 1e-100  # if Population decreases below this number, set it to 0
Kinf = False
K = 1
Kinf_pop_floor = [Kinf, pop_floor]

N_p = 1  # number of initial phage species
N_r = 2  # number of branching ratios for each phage species
p0_rand = False

r1_jk = np.round(np.ones((N_p, N_r)) *  # branching ratio for single lysogen
    #np.expand_dims(np.linspace(0., 0.2, N_r), 0)  # all phages have same r's
    #np.linspace(0., 0.7, N_p * N_r).reshape((N_p, N_r))  # all different r's
    np.array([0.2, 0.]).reshape((N_p, N_r))  # something we specify
    , 6)

double_lysogen_mask, phage_mask, lysogen_mask, r2_jk, r3_jk, lam_ijk, gamma_ijk, b0, p0, l0, l20 = other_vars_from_params(
    N_b, N_p, N_r, r1_jk, lam, gamma, c1, alpha_l2, p0_rand=p0_rand)
dt, save_every, t_eval, rtol, atol = dt_tol_params(euler_method, t_max, use_t_eval=True)
if log_space:
    dt_tols = [dt, 1e-3, 1e-6] #rtol, atol]
else:
    dt_tols = [dt, 1e-13, 1e-100]

params = [alpha_b, delta_b, beta, delta_p, alpha_l, delta_l, alpha_l2, delta_l2, 
          lam_ijk, c1, c2, K, r1_jk, r2_jk, r3_jk, gamma_ijk, phage_mask, lysogen_mask, double_lysogen_mask,
          phage_die_when_infecting_lysogen, triple_lysogens_displace_double,
          only_phage_die_with_multiple_lys, log_space]

title = make_title(params, Kinf_pop_floor)

t_vec, b_t, p_t, l_t, l2_t = phage_lysogen_simulation(
    b0, p0, l0, l20, 0, t_max, Kinf_pop_floor, euler_method, dt_tols, params, save_every, t_eval)


t_start = 0
    
plt.figure()
plt.plot(t_vec[t_start:], p_t[0, 1, t_start:], linewidth=4,
         linestyle=linestyles[r1_jk[0, 1]], color=colors[0],
         label=#'Obligate lytic strain' 
         #p_label(1, 2, r1_jk[0, 1], include_T_OL=True, include_r=False)
         p_label(1, r1_jk[0, 1], r1_jk[0, 1], include_T_OL=False, include_r=False)
         )
plt.plot(t_vec[t_start:], p_t[0, 0, t_start:], linewidth=4,
         linestyle=linestyles[r1_jk[0, 0]], color=colors[0],
         label=#'Temperate strain'
         #p_label(1, 1, r1_jk[0, 0], include_T_OL=True, include_r=False)
         p_label(1, r1_jk[0, 0], r1_jk[0, 0], include_T_OL=False, include_r=False)
         )
plt.plot(t_vec[t_start:], np.sum(l_t[0, 0, :, t_start:], 0), linewidth=1,
         color=colors[0], 
         label=#'Lysogens'
         r'$L_{0}$'.format(1)
         , alpha=0.5)
plt.yscale('log')
plt.xlabel('Time (generations)')
plt.ylabel('Population')
leg = plt.legend(fontsize=15, loc='best')
for legobj in leg.legendHandles:
    if legobj.get_linestyle() != '-':
        if legobj.get_linestyle() == '--':
            legobj.set_linewidth(2.5)  # 2.
        else:
            legobj.set_linewidth(2.5)

plt.yticks([1e0, 1e-5, 1e-10, 1e-15])
plt.xticks([i * doubling_time_lys for i in [0, 10, 20, 30]], [0, 10, 20, 30])
plt.ylim([1e-16, K * 100])   
if save_figs:
    save_svg_and_png('lysogen_wins')
plt.show()
        
Kinf = True
K = 1000
pop_floor = 1e-20  # if Population decreases below this number, set it to 0
Kinf_pop_floor = [Kinf, pop_floor]

#%%
# =============================================================================
# Show that lysogenic phage can dominate with sensitives (1)
# =============================================================================

c1 = 1e-5
pop_floor = 1e-100  # if Population decreases below this number, set it to 0
Kinf = False
K = 1
Kinf_pop_floor = [Kinf, pop_floor]


t_max = int(doubling_time_lys * (50 - 10 * np.log10(K))) + 1
euler_method = False

N_p = 1  # number of initial phage species
N_r = 2  # number of branching ratios for each phage species
p0_rand = False

r1_jk = np.round(np.ones((N_p, N_r)) *  # branching ratio for single lysogen
    #np.expand_dims(np.linspace(0., 0.2, N_r), 0)  # all phages have same r's
    #np.linspace(0., 0.7, N_p * N_r).reshape((N_p, N_r))  # all different r's
    np.array([0.2, 0.]).reshape((N_p, N_r))  # something we specify
    , 6)

double_lysogen_mask, phage_mask, lysogen_mask, r2_jk, r3_jk, lam_ijk, gamma_ijk, b0, p0, l0, l20 = other_vars_from_params(
    N_b, N_p, N_r, r1_jk, lam, gamma, c1, alpha_l2, p0_rand=p0_rand)
dt, save_every, t_eval, rtol, atol = dt_tol_params(euler_method, t_max, use_t_eval=True)
dt_tols = [dt, 1e-13, 1e-100]

params = [alpha_b, delta_b, beta, delta_p, alpha_l, delta_l, alpha_l2, delta_l2, 
          lam_ijk, c1, c2, K, r1_jk, r2_jk, r3_jk, gamma_ijk, phage_mask, lysogen_mask, double_lysogen_mask,
          phage_die_when_infecting_lysogen, triple_lysogens_displace_double,
          only_phage_die_with_multiple_lys, log_space]

title = make_title(params, Kinf_pop_floor)

t_vec, b_t, p_t, l_t, l2_t = phage_lysogen_simulation(
    b0, p0, l0, l20, 0, t_max, Kinf_pop_floor, euler_method, dt_tols, params, save_every, t_eval)


t_start = 0
    
plt.figure()
plt.plot(t_vec[t_start:], p_t[0, 1, t_start:], linewidth=4,
         linestyle=linestyles[r1_jk[0, 1]], color=colors[0],
         label=p_label(1, r1_jk[0, 1], r1_jk[0, 1], include_T_OL=False, include_r=False)) #'Obligate lytic strain')
plt.plot(t_vec[t_start:], p_t[0, 0, t_start:], linewidth=4,
         linestyle=linestyles[r1_jk[0, 0]], color=colors[0],
         label=p_label(1, r1_jk[0, 0], r1_jk[0, 0], include_T_OL=False, include_r=False)) #'Temperate strain')
plt.plot(t_vec[t_start:], np.sum(l_t[0, 0, :, t_start:], 0), linewidth=1,
         color=colors[0], label=r'$L_1$', alpha=0.5)
plt.plot(t_vec[t_start:], b_t[0, t_start:], ':k', linewidth=2, 
         label=r'$S$')
plt.yscale('log')
plt.xlabel('Time (generations)')
plt.ylabel('Population')
leg = plt.legend(fontsize=15, loc='lower left')#, bbox_to_anchor=(1.05, 1))
for legobj in leg.legendHandles:
    if legobj.get_linestyle() != '-':
        if legobj.get_linestyle() == '--':
            legobj.set_linewidth(2.5)  # 2.
        else:
            legobj.set_linewidth(2.5)

plt.xticks([i * doubling_time_lys for i in np.arange(
    0, int(t_max / doubling_time_lys) + 1, int(t_max / doubling_time_lys / 5))], 
            [int(i) for i in np.arange(
    0, int(t_max / doubling_time_lys) + 1, int(t_max / doubling_time_lys / 5))])
# plt.yticks([1e0, 1e-3, 1e-6])
# plt.ylim([1e-7, 1e1])  
plt.yticks([1e0, 1e-5, 1e-10, 1e-15])
plt.ylim([1e-16, K * 100])   
 
if save_figs:
    save_svg_and_png('temperate_wins_with_sensitive')
plt.show()
        
c1 = 0
Kinf = True
K = 1000
pop_floor = 1e-20  # if Population decreases below this number, set it to 0
Kinf_pop_floor = [Kinf, pop_floor]


#%%
# =============================================================================
# Show that lytic phage can dominate (2)
# =============================================================================

t_max = int(doubling_time_lys * 300) + 1
euler_method = False

pop_floor = 1e-100  # if Population decreases below this number, set it to 0
Kinf = False
K = 1
Kinf_pop_floor = [Kinf, pop_floor]

N_p = 2  # number of initial phage species
N_r = 1  # number of branching ratios for each phage species
p0_rand = False

r1_jk = np.round(np.ones((N_p, N_r)) *  # branching ratio for single lysogen
    #np.expand_dims(np.linspace(0., 0.2, N_r), 0)  # all phages have same r's
    #np.linspace(0., 0.7, N_p * N_r).reshape((N_p, N_r))  # all different r's
    np.array([0., 0.2]).reshape((N_p, N_r))  # something we specify
    , 6)

double_lysogen_mask, phage_mask, lysogen_mask, r2_jk, r3_jk, lam_ijk, gamma_ijk, b0, p0, l0, l20 = other_vars_from_params(
    N_b, N_p, N_r, r1_jk, lam, gamma, c1, alpha_l2, p0_rand=p0_rand)
dt, save_every, t_eval, rtol, atol = dt_tol_params(euler_method, t_max, use_t_eval=True)
if log_space:
    dt_tols = [dt, 1e-3, 1e-6] #rtol, atol]
else:
    dt_tols = [dt, 1e-13, 1e-100]

params = [alpha_b, delta_b, beta, delta_p, alpha_l, delta_l, alpha_l2, delta_l2, 
          lam_ijk, c1, c2, K, r1_jk, r2_jk, r3_jk, gamma_ijk, phage_mask, lysogen_mask, double_lysogen_mask,
          phage_die_when_infecting_lysogen, triple_lysogens_displace_double,
          only_phage_die_with_multiple_lys, log_space]

title = make_title(params, Kinf_pop_floor)

t_vec, b_t, p_t, l_t, l2_t = phage_lysogen_simulation(
    b0, p0, l0, l20, 0, t_max, Kinf_pop_floor, euler_method, dt_tols, params, save_every, t_eval)


t_start = 0
    
plt.figure()
plt.plot(t_vec[t_start:], p_t[0, 0, t_start:], linewidth=4,
         linestyle=linestyles[r1_jk[0, 0]], color=colors[0],
         label=#r'$P_{1,1}$' + '; ' + temp_lyt[r1_jk[0, 0]]
         p_label(1, r1_jk[0, 0], r1_jk[0, 0], include_T_OL=False, include_r=False)
         )
plt.plot(t_vec[t_start:], p_t[1, 0, t_start:], linewidth=4,
         linestyle=linestyles[r1_jk[1, 0]], color=colors[1],
         label=#r'$P_{2,1}$' + '; ' + temp_lyt[r1_jk[1, 0]]
         p_label(2, r1_jk[1, 0], r1_jk[1, 0], include_T_OL=False, include_r=False)
         )
plt.plot(t_vec[t_start:], np.sum(l_t[0, 1, :, t_start:], 0), linewidth=1,
         color=colors[1], label=r'$L_{0}$'.format(2), alpha=0.5)
plt.yscale('log')
plt.xlabel('Time (generations)')
plt.ylabel('Population')
leg = plt.legend(fontsize=15, loc='lower right')
for legobj in leg.legendHandles:
    if legobj.get_linestyle() != '-':
        if legobj.get_linestyle() == '--':
            legobj.set_linewidth(2.5)  # 2.
        else:
            legobj.set_linewidth(2.5)

plt.yticks([1e0, 1e-3, 1e-6])
plt.xticks([i * doubling_time_lys for i in np.arange(0, t_max / doubling_time_lys, int(t_max / doubling_time_lys / 5))], 
           [int(i) for i in np.arange(0, t_max / doubling_time_lys, int(t_max / doubling_time_lys / 5))])
plt.ylim([1e-7, 1e1])   
if save_figs:
    save_svg_and_png('lytic_wins')
plt.show()
        
Kinf = True
K = 1000
pop_floor = 1e-20  # if Population decreases below this number, set it to 0
Kinf_pop_floor = [Kinf, pop_floor]

#%%
# =============================================================================
# Show that lytic phage can dominate with sensitives (2)
# =============================================================================

c1 = 1e-5
pop_floor = 1e-100  # if Population decreases below this number, set it to 0
Kinf = False
K = 1
Kinf_pop_floor = [Kinf, pop_floor]
alpha_b_real = copy.copy(alpha_b)
#alpha_b = alpha_l


t_max = int(doubling_time_lys * 300) + 1
euler_method = False

N_p = 2  # number of initial phage species
N_r = 1  # number of branching ratios for each phage species
p0_rand = 1

r1_jk = np.round(np.ones((N_p, N_r)) *  # branching ratio for single lysogen
    #np.expand_dims(np.linspace(0., 0.2, N_r), 0)  # all phages have same r's
    #np.linspace(0., 0.7, N_p * N_r).reshape((N_p, N_r))  # all different r's
    np.array([0.2, 0.] + [0.2] * (N_p - 2)).reshape((N_p, N_r))  # something we specify
    , 6)

double_lysogen_mask, phage_mask, lysogen_mask, r2_jk, r3_jk, lam_ijk, gamma_ijk, b0, p0, l0, l20 = other_vars_from_params(
    N_b, N_p, N_r, r1_jk, lam, gamma, c1, alpha_l2, p0_rand=p0_rand)#, b0_init=1e-3, l0_init=1e-3)
dt, save_every, t_eval, rtol, atol = dt_tol_params(euler_method, t_max, use_t_eval=True)
if N_p < 3:
    dt_tols = [dt, 1e-10, 1e-30]
else:
    dt_tols = [dt, rtol, atol]

params = [alpha_b, delta_b, beta, delta_p, alpha_l, delta_l, alpha_l2, delta_l2, 
          lam_ijk, c1, c2, K, r1_jk, r2_jk, r3_jk, gamma_ijk, phage_mask, lysogen_mask, double_lysogen_mask,
          phage_die_when_infecting_lysogen, triple_lysogens_displace_double,
          only_phage_die_with_multiple_lys, log_space]

title = make_title(params, Kinf_pop_floor)

t_vec, b_t, p_t, l_t, l2_t = phage_lysogen_simulation(
    b0, p0, l0, l20, 0, t_max, Kinf_pop_floor, euler_method, dt_tols, params, save_every, t_eval)


t_start = 0
    
plt.figure()
plt.plot(t_vec[t_start:], p_t[1, 0, t_start:], linewidth=4,
         linestyle=linestyles[r1_jk[1, 0]], color=colors[0],
         label=p_label(1, r1_jk[1, 0], r1_jk[1, 0], include_T_OL=False, include_r=False))# 'Obligate lytic phage')
plt.plot(t_vec[t_start:], p_t[0, 0, t_start:], linewidth=4,
         linestyle=linestyles[r1_jk[0, 0]], color=colors[1],
         label=p_label(2, r1_jk[0, 0], r1_jk[0, 0], include_T_OL=False, include_r=False))# 'Temperate phage')
plt.plot(t_vec[t_start:], np.sum(l_t[0, 0, :, t_start:], 0), linewidth=1,
         color=colors[1], label=r'$L_{0}$'.format(2), alpha=0.5)
for j in range(2, N_p):
    plt.plot(t_vec[t_start:], p_t[j, 0, t_start:], linewidth=4,
             linestyle=linestyles[r1_jk[j, 0]], color=colors[j],
             label=p_label(j + 1, r1_jk[j, 0], r1_jk[j, 0], include_T_OL=False, include_r=False))# 'Temperate phage')
    plt.plot(t_vec[t_start:], np.sum(l_t[0, j, :, t_start:], 0), linewidth=1,
             color=colors[j], label=r'$L_{0}$'.format(j + 1), alpha=0.5)
plt.plot(t_vec[t_start:], b_t[0, t_start:], ':k', linewidth=2, 
         label=r'$S$')
plt.yscale('log')
plt.xlabel('Time (generations)')
plt.ylabel('Population')
leg = plt.legend(fontsize=15, loc='lower right')#bbox_to_anchor=(1.05, 1))
for legobj in leg.legendHandles:
    if legobj.get_linestyle() != '-':
        if legobj.get_linestyle() == '--':
            legobj.set_linewidth(2.5)  # 2.
        else:
            legobj.set_linewidth(2.5)

plt.yticks([1e0, 1e-3, 1e-6])
plt.xticks([i * doubling_time_lys for i in np.arange(
    0, int(t_max / doubling_time_lys) + 1, int(t_max / doubling_time_lys / 5))], 
           [int(i) for i in np.arange(
    0, int(t_max / doubling_time_lys) + 1, int(t_max / doubling_time_lys / 5))])
plt.ylim([1e-7, 1e1])   
if save_figs:
    save_svg_and_png('lytic_wins_with_sensitive')
plt.show()

alpha_b = alpha_b_real   
c1 = 0
Kinf = True
K = 1000
pop_floor = 1e-20  # if Population decreases below this number, set it to 0
Kinf_pop_floor = [Kinf, pop_floor]


#%% 

# Show that sensitive bacteria are negligible
c1 = 1e-5  # 1e-5  # rate of prophage -> cryptic prophage for single lysogen

t_max = 1000
euler_method = False

N_p = 4  # number of initial phage species
N_r = 1  # number of branching ratios for each phage species
p0_rand = 1

r1_jk = np.round(np.ones((N_p, N_r)) *  # branching ratio for single lysogen
    np.expand_dims(np.linspace(0.2, 0.2, N_r), 0)  # all phages have same r's
    #np.linspace(0., 0.7, N_p * N_r).reshape((N_p, N_r))  # all different r's
    #np.array([0, 0.2]).reshape((N_p, N_r))  # something we specify
    , 6)

double_lysogen_mask, phage_mask, lysogen_mask, r2_jk, r3_jk, lam_ijk, gamma_ijk, b0, p0, l0, l20 = other_vars_from_params(
    N_b, N_p, N_r, r1_jk, lam, gamma, c1, alpha_l2, p0_rand=p0_rand, p0_init=1e-3, l0_init=1e-2, b0_init=1e-2)
dt, save_every, t_eval, rtol, atol = dt_tol_params(euler_method, t_max, use_t_eval=True)
dt_tols = [dt, rtol, atol]

params = [alpha_b, delta_b, beta, delta_p, alpha_l, delta_l, alpha_l2, delta_l2, 
          lam_ijk, c1, c2, K, r1_jk, r2_jk, r3_jk, gamma_ijk, phage_mask, lysogen_mask, double_lysogen_mask,
          phage_die_when_infecting_lysogen, triple_lysogens_displace_double,
          only_phage_die_with_multiple_lys, log_space]

title = make_title(params, Kinf_pop_floor)

t_vec, b_t, p_t, l_t, l2_t = phage_lysogen_simulation(
    b0, p0, l0, l20, 0, t_max, Kinf_pop_floor, euler_method, dt_tols, params, save_every, t_eval)


t_start = 7500
t_end = 7551

plt.figure()
for j in range(N_p):
    for k in range(N_r):
        if phage_mask[j, k]:
            plt.plot(t_vec[t_start:t_end], p_t[j, k, t_start:t_end], linewidth=4,
                     linestyle=linestyles[r1_jk[j, k]], color=colors[j],
                     label=p_label(j + 1, r1_jk[j, k], r1_jk[j, k], include_T_OL=False, include_r=False))
    plt.plot(t_vec[t_start:t_end], np.sum(l_t[0, j, :, t_start:t_end], 0), linewidth=1,
             color=colors[j], label=r'$L_{0}$'.format(j+1), alpha=0.5)
plt.plot(t_vec[t_start:t_end], b_t[0, t_start:t_end], color='k', linestyle=':', linewidth=2,
         label='$S$')
plt.yscale('log')
plt.xlabel('Time (generations)')
plt.ylabel('Population')
plt.yticks([1e0, 1e-3, 1e-6])
# plt.ylim([5e-6, 2e1])
plt.ylim([1e-7, 1e1])

plt.xticks(np.arange(t_start/10, (t_end + 1)/10, 1 * doubling_time_lys), [])
ax = plt.gca()
scalebar = AnchoredSizeBar(ax.transData,
                           1 * doubling_time_lys, '1', loc='lower right', 
                           pad=0.5,
                           color='black',
                           frameon=False,
                           label_top=True
                           )
ax.add_artist(scalebar)
         
leg = plt.legend(fontsize=13, bbox_to_anchor=(1.05, 1))  # loc='lower right')
for legobj in leg.legendHandles:
    if legobj.get_linestyle() != '-':
        if legobj.get_linestyle() == '--':
            legobj.set_linewidth(2.5)  # 2.
        else:
            legobj.set_linewidth(2.5)
if save_figs:
    save_svg_and_png('sensitives_are_negligible')
plt.show()


c1 = 0#1e-5  # 1e-5  # rate of prophage -> cryptic prophage for single lysogen

#%% 
# Show oscillations --> chaos with sensitive bacteria

euler_method = False
K = 1000  # carrying capacity; also affects initial conditions for bacteria
Kinf = False  # should we set K to infinity? (if it's true, we disregard the value of K set above)
pop_floor = 1e-20  # if Population decreases below this number, set it to 0
Kinf_pop_floor = [Kinf, pop_floor]

N_r = 2  # number of branching ratios for each phage species
p0_rand = 1

for c1 in [0, 1e-5]:
    for N_p in range(1, 5):
        if N_p == 1:    
            t_max = int(30 * doubling_time_lys) + 1
            dt_tols = [0.1, 1e-13, 1e-100]
            r1_jk = np.round(np.ones((N_p, N_r)) *  # branching ratio for single lysogen
                np.expand_dims(np.linspace(0., 0.2, N_r), 0)  # all phages have same r's
                #np.linspace(0., 0.7, N_p * N_r).reshape((N_p, N_r))  # all different r's
                #np.array([0, 0.2]).reshape((N_p, N_r))  # something we specify
                , 6)
        else:            
            t_max = 500 #351
    
            r1_jk = np.round(np.ones((N_p, N_r)) *  # branching ratio for single lysogen
                # np.expand_dims(np.linspace(0.2, 0.2, N_r), 0)  # all phages have same r's
                # np.linspace(0.2, 0.4, N_p * N_r).reshape((N_p, N_r))  # all different r's
                np.array([0, 0.2] + [0.2] * 2 * (N_p - 1)).reshape((N_p, N_r))  # something we specify
                , 6)
    
        
        double_lysogen_mask, phage_mask, lysogen_mask, r2_jk, r3_jk, lam_ijk, gamma_ijk, b0, p0, l0, l20 = other_vars_from_params(
            N_b, N_p, N_r, r1_jk, lam, gamma, c1, alpha_l2, p0_rand=p0_rand, p0_init=1e-3, l0_init=1e-2, b0_init=1e-2)
        dt, save_every, t_eval, rtol, atol = dt_tol_params(euler_method, t_max, use_t_eval=True)
        if N_p > 1:
            dt_tols = [dt, rtol, atol] #[dt, 1e-10, 1e-20]
        
        params = [alpha_b, delta_b, beta, delta_p, alpha_l, delta_l, alpha_l2, delta_l2, 
                  lam_ijk, c1, c2, K, r1_jk, r2_jk, r3_jk, gamma_ijk, phage_mask, lysogen_mask, double_lysogen_mask,
                  phage_die_when_infecting_lysogen, triple_lysogens_displace_double,
                  only_phage_die_with_multiple_lys, log_space]
        
        title = make_title(params, Kinf_pop_floor)
        
        t_vec, b_t, p_t, l_t, l2_t = phage_lysogen_simulation(
            b0, p0, l0, l20, 0, t_max, Kinf_pop_floor, euler_method, dt_tols, params, save_every, t_eval)
        
        
        if N_p == 1:
            t_start = 0
            t_end = len(t_vec)
        else:
            t_start = len(t_vec) - int(1000 * doubling_time_lys + 1)  #2900
            t_end = len(t_vec) #3411
        
        plt.figure()
        for j in range(N_p):
            for k in range(N_r):
                if phage_mask[j, k]:
                    plt.plot(t_vec[t_start:t_end], p_t[j, k, t_start:t_end], linewidth=4,
                             linestyle=linestyles[r1_jk[j, k]], color=colors[j],
                             label=p_label(j + 1, r1_jk[j, k], r1_jk[j, k], include_T_OL=False, include_r=False))
            plt.plot(t_vec[t_start:t_end], np.sum(l_t[0, j, :, t_start:t_end], 0), linewidth=1,
                     color=colors[j], label=r'$L_{0}$'.format(j+1), alpha=0.5)
        if c1 > 0:
            plt.plot(t_vec[t_start:t_end], b_t[0, t_start:t_end], color='k', linestyle=':', linewidth=2,
                     label='$S$')
        
        # Make j=0 phage pop
        j = 0
        for k in range(N_r):
            if phage_mask[j, k]:
                plt.plot(t_vec[t_start:t_end], p_t[j, k, t_start:t_end], linewidth=4,
                         linestyle=linestyles[r1_jk[j, k]], color=colors[j])

        plt.yscale('log')
        plt.xlabel('Time (generations)')
        plt.ylabel('Population')
        # plt.yticks([1e0, 1e-3, 1e-6])
        # plt.ylim([5e-6, 2e1])
        # if N_p == 1:
        #     plt.ylim([1e-7, 5e1])    
        # else:
        #     plt.ylim([4e-7, 5e1])    
        
        #if N_p == 1:
        plt.yticks([1e2, 1e-5, 1e-12]) #[1e0, 1e-4, 1e-8, 1e-12])
        plt.ylim([5e-14, 5e3])    
        # else: # if N_p == 2 or N_p == 3:
        #     plt.yticks([1e0, 1e-4, 1e-8])
        #     plt.ylim([1e-10, 5e3])
        
        if len(t_vec) < 600:
            plt.xticks(np.arange(t_start/10, (t_end + 1)/10, 10 * doubling_time_lys), #[]
                       np.arange(t_start/10 / doubling_time_lys, np.round((t_end + 9)/10 / doubling_time_lys), 10, dtype=int)
                       )
        else:
            plt.xticks(np.arange(t_start/10, (t_end + 1)/10, 10 * doubling_time_lys), [])
            ax = plt.gca()
            scalebar = AnchoredSizeBar(ax.transData,
                                        10 * doubling_time_lys, '10', loc='lower right', 
                                        pad=0.5,
                                        color='black',
                                        frameon=False,
                                        label_top=True
                                        )
            ax.add_artist(scalebar)
                 
        leg = plt.legend(fontsize=13, bbox_to_anchor=(1.05, 1))  # loc='lower right')
        for legobj in leg.legendHandles:
            if legobj.get_linestyle() != '-':
                if legobj.get_linestyle() == '--':
                    legobj.set_linewidth(2.5)  # 2.
                else:
                    legobj.set_linewidth(2.5)
        if save_figs:
            save_svg_and_png('Np_' + str(N_p) + '_including_sensitives' * (c1 > 0))
        plt.show()
        
        
        
c1 = 0#1e-5  # 1e-5  # rate of prophage -> cryptic prophage for single lysogen
K = 1000  # carrying capacity; also affects initial conditions for bacteria
Kinf = True  # should we set K to infinity? (if it's true, we disregard the value of K set above)

pop_floor = 1e-20  # if Population decreases below this number, set it to 0
Kinf_pop_floor = [Kinf, pop_floor]


#%% Include double lysogens

euler_method = False
p0_rand = 1

N_r = 2  # number of branching ratios for each phage species

c1 = 1e-5  # 1e-5  # rate of prophage -> cryptic prophage for single lysogen
c2 = 1e-5
alpha_l2 = 0.95 * alpha_l
K = 100  # carrying capacity; also affects initial conditions for bacteria
Kinf = False  # should we set K to infinity? (if it's true, we disregard the value of K set above)

for N_p in range(1, 5):
    if N_p < 3:
        pop_floor = 1e-40  # if Population decreases below this number, set it to 0
        Kinf_pop_floor = [Kinf, pop_floor]
        
        t_max = int(30 * doubling_time_lys) + 1
        dt_tols = [0.1, 1e-13, 1e-100]
        r1_jk = np.round(np.ones((N_p, N_r)) *  # branching ratio for single lysogen
            np.expand_dims(np.linspace(0., 0.2, N_r), 0)  # all phages have same r's
            #np.linspace(0., 0.7, N_p * N_r).reshape((N_p, N_r))  # all different r's
            #np.array([0, 0.2]).reshape((N_p, N_r))  # something we specify
            , 6)
    else:
        # c1 = 0
        # Kinf = True
        pop_floor = 1e-20
        Kinf_pop_floor = [Kinf, pop_floor]
        
        t_max = 351

        r1_jk = np.round(np.ones((N_p, N_r)) *  # branching ratio for single lysogen
            # np.expand_dims(np.linspace(0.2, 0.2, N_r), 0)  # all phages have same r's
            # np.linspace(0.2, 0.4, N_p * N_r).reshape((N_p, N_r))  # all different r's
            np.array([0, 0.2] + [0.2] * 2 * (N_p - 1)).reshape((N_p, N_r))  # something we specify
            , 6)

    
    double_lysogen_mask, phage_mask, lysogen_mask, r2_jk, r3_jk, lam_ijk, gamma_ijk, b0, p0, l0, l20 = other_vars_from_params(
        N_b, N_p, N_r, r1_jk, lam, gamma, c1, alpha_l2, p0_rand=p0_rand, p0_init=1e-3, l0_init=1e-2, b0_init=1e-2)
    dt, save_every, t_eval, rtol, atol = dt_tol_params(euler_method, t_max, use_t_eval=True)
    if N_p > 1:
        dt_tols = [dt, rtol, atol] #[dt, 1e-10, 1e-20]
    
    params = [alpha_b, delta_b, beta, delta_p, alpha_l, delta_l, alpha_l2, delta_l2, 
              lam_ijk, c1, c2, K, r1_jk, r2_jk, r3_jk, gamma_ijk, phage_mask, lysogen_mask, double_lysogen_mask,
              phage_die_when_infecting_lysogen, triple_lysogens_displace_double,
              only_phage_die_with_multiple_lys, log_space]
    
    title = make_title(params, Kinf_pop_floor)
    
    t_vec, b_t, p_t, l_t, l2_t = phage_lysogen_simulation(
        b0, p0, l0, l20, 0, t_max, Kinf_pop_floor, euler_method, dt_tols, params, save_every, t_eval)
    
    
    if N_p < 3:
        t_start = 0
        t_end = len(t_vec)
    else:
        # t_start = 2900
        # t_end = 3411
        t_start = len(t_vec) - int(1000 * doubling_time_lys + 1)  #2900
        t_end = len(t_vec) #3411
        
    plt.figure()
    for j in range(N_p):
        for k in range(N_r):
            if phage_mask[j, k]:
                plt.plot(t_vec[t_start:t_end], p_t[j, k, t_start:t_end], linewidth=4,
                         linestyle=linestyles[r1_jk[j, k]], color=colors[j],
                         label=p_label(j + 1, r1_jk[j, k], r1_jk[j, k], include_T_OL=False, include_r=False))
        plt.plot(t_vec[t_start:t_end], np.sum(l_t[0, j, :, t_start:t_end], 0), linewidth=1,
                 color=colors[j], label=r'$L_{0}$'.format(j+1), alpha=0.5)
        
        if alpha_l2 > 0: 
            for j2 in range(N_p):
                if np.any(double_lysogen_mask[0, j, :, j2, :]):
                    plt.plot(t_vec[t_start:t_end], np.sum(l2_t[0, j, :, j2, :, t_start:t_end], (0,1)), 
                             linewidth=2, linestyle=':', color=double_lysogen_colors[j][j2], 
                             label=r'$D_{{{0}, {1}}}$'.format(j2 + 1, j + 1))
    # if alpha_l2 > 0 and N_p == 2:
    #     plt.plot(t_vec[t_start:t_end], np.sum(l2_t[0, j, :, :, :, t_start:t_end], (0,1,2)), linewidth=2, linestyle=':',
    #              color='green', label=r'$D_{1,2}$')
    if c1 > 0:
        plt.plot(t_vec[t_start:t_end], b_t[0, t_start:t_end], color='k', linestyle=':', linewidth=2,
                 label='$S$')
        
    # Make j=0 phage pop
    j = 0
    for k in range(N_r):
        if phage_mask[j, k]:
            plt.plot(t_vec[t_start:t_end], p_t[j, k, t_start:t_end], linewidth=4,
                     linestyle=linestyles[r1_jk[j, k]], color=colors[j])
    
    plt.yscale('log')
    plt.xlabel('Time (generations)')
    plt.ylabel('Population')
    plt.yticks([1e2, 1e-5, 1e-12]) #[1e0, 1e-4, 1e-8, 1e-12])
    # plt.ylim([5e-6, 2e1])
    if N_p <= 2:
        plt.ylim([5e-14, 5e3])    
    else:
        plt.ylim([5e-14, 5e3])    

    if len(t_vec) < 600:
        plt.xticks(np.arange(t_start/10, (t_end + 1)/10, 10 * doubling_time_lys), #[]
                   np.arange(t_start/10 / doubling_time_lys, np.round((t_end + 9)/10 / doubling_time_lys), 10, dtype=int)
                   )
    else:
        plt.xticks(np.arange(t_start/10, (t_end + 1)/10, 10 * doubling_time_lys), [])
        ax = plt.gca()
        scalebar = AnchoredSizeBar(ax.transData,
                                    10 * doubling_time_lys, '10', loc='lower right', 
                                    pad=0.5,
                                    color='black',
                                    frameon=False,
                                    label_top=True
                                    )
        ax.add_artist(scalebar)
             
    leg = plt.legend(fontsize=13, ncol=1 + (N_p>=3), bbox_to_anchor=(1.05, 1))  # loc='lower right')
    for legobj in leg.legendHandles:
        if legobj.get_linestyle() != '-':
            if legobj.get_linestyle() == '--':
                legobj.set_linewidth(2.5)  # 2.
            else:
                legobj.set_linewidth(2.5)
    if save_figs:
        save_svg_and_png('double_lysogens_Np_' + str(N_p))
    plt.show()
    
    
c1 = 0
c2 = 0
alpha_l2 = 0
K = 1000  # carrying capacity; also affects initial conditions for bacteria
Kinf = True  # should we set K to infinity? (if it's true, we disregard the value of K set above)
pop_floor = 1e-20  # if Population decreases below this number, set it to 0
Kinf_pop_floor = [Kinf, pop_floor]



#%%
# =============================================================================
# Show that trajectories are chaotic by measuring distance between nearby trajectories
# =============================================================================

t_max_init = 200  # time to initialize vars
t_max = int(1000 * doubling_time_lys) + 1
euler_method = False

N_p = 3  # number of initial phage species
p0_rand = 3
for N_r in [1]:  # [1, 2]
    if N_r == 1:
        r1_jk = np.round(np.ones((N_p, N_r)) *  # branching ratio for single lysogen
            np.expand_dims(np.linspace(0.2, 0.2, N_r), 0)  # all phages have same r's
            #np.linspace(0., 0.7, N_p * N_r).reshape((N_p, N_r))  # all different r's
            #np.array([0, 0.2]).reshape((N_p, N_r))  # something we specify
            , 6)
    else:
        r1_jk = np.round(np.ones((N_p, N_r)) *  # branching ratio for single lysogen
            #np.expand_dims(np.linspace(0., 0.2, N_r), 0)  # all phages have same r's
            #np.linspace(0., 0.7, N_p * N_r).reshape((N_p, N_r))  # all different r's
            np.array([[0.2, 0.], [0.2] * 2, [0.2] * 2]).reshape((N_p, N_r))  # something we specify
            , 6)

    double_lysogen_mask, phage_mask, lysogen_mask, r2_jk, r3_jk, lam_ijk, gamma_ijk, b0, p0, l0, l20 = other_vars_from_params(
        N_b, N_p, N_r, r1_jk, lam, gamma, c1, alpha_l2, p0_rand=p0_rand, p0_init=1e-3, l0_init=1e-3)
    
    params = [alpha_b, delta_b, beta, delta_p, alpha_l, delta_l, alpha_l2, delta_l2, 
              lam_ijk, c1, c2, K, r1_jk, r2_jk, r3_jk, gamma_ijk, phage_mask, lysogen_mask, double_lysogen_mask,
              phage_die_when_infecting_lysogen, triple_lysogens_displace_double,
              only_phage_die_with_multiple_lys, log_space]
    
    title = make_title(params, Kinf_pop_floor)
    
    # Initialize variables to start at a natural place (no burn-in time)
    dt, save_every, t_eval, rtol, atol = dt_tol_params(euler_method, t_max_init, use_t_eval=True)
    dt_tols = [dt, rtol, atol]
    t_vec, b_t, p_t, l_t, l2_t = phage_lysogen_simulation(
        b0, p0, l0, l20, 0, t_max_init, Kinf_pop_floor, euler_method, dt_tols, params, save_every, t_eval)
    
    b0 = b_t[:, -1]; p0 = p_t[:, :, -1]; l0 = l_t[:, :, :, -1]; l20 = l2_t[:, :, :, :, :, -1]
    
    # Run the simulation once
    dt, save_every, t_eval, rtol, atol = dt_tol_params(euler_method, t_max, use_t_eval=True)
    dt_tols = [dt, 3e-14, 1e-10]  # make rtol very small since that's what we care about measuring here
    t_vec, b_t, p_t_1, l_t, l2_t = phage_lysogen_simulation(
        b0, p0, l0, l20, 0, t_max, Kinf_pop_floor, euler_method, dt_tols, params, save_every, t_eval)
    
    
    # Run it again with slightly modified p0
    p0_2 = p0 + 1e-13
    t_vec, b_t, p_t_2, l_t, l2_t = phage_lysogen_simulation(
        b0, p0_2, l0, l20, 0, t_max, Kinf_pop_floor, euler_method, dt_tols, params, save_every, t_eval)
    
    
    t_start = 0
    # if only_phage_die_with_multiple_lys:
    #     t_end = 2900
    # else:
    #     t_end = 3500
    if N_r == 1:
        t_end = 3500
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        t_vec[t_start:t_end], np.log(np.sqrt((p_t_2[0, 0, t_start:t_end] - p_t_1[0, 0, t_start:t_end])**2)))
    print('Slope of best fit line to chaos function is ' + str(slope))
    
    plt.figure()
    plt.plot(t_vec, np.sqrt((p_t_2 - p_t_1)**2)[0, 0, :], '-', linewidth=1, color=colors[0])
    plt.plot(t_vec[t_start:t_end], np.exp(intercept) * np.exp(t_vec[t_start:t_end] * slope), 
             color=colors[1], linestyle='--', linewidth=4)
    plt.xticks([i * doubling_time_lys for i in [0, 250, 500, 750, 1000]], 
               [0, 250, 500, 750, 1000])
    plt.yscale('log')
    plt.xlabel('Time (generations)')
    plt.ylabel('Distance between\ntrajectories')
    plt.yticks([1e-15, 1e-10, 1e-5, 1e0])
    if save_figs:
        save_svg_and_png('chaos_Nr_' + str(N_r))
    plt.show()



#%%
# =============================================================================
# Show that chaos persists with heterogeneities
# =============================================================================

t_max = int(1e3 * doubling_time_lys) + 1
euler_method = False

N_p = 3  # number of initial phage species
p0_rand = 1

for N_r in [1, 2]:  #[1, 2]:
    if N_r == 1:
        r1_jk = np.round(np.ones((N_p, N_r)) *  # branching ratio for single lysogen
            # np.expand_dims(np.linspace(0.2, 0.2, N_r), 0)  # all phages have same r's
            np.linspace(0.1, 0.3, N_p * N_r).reshape((N_p, N_r))  # all different r's
            #np.array([0, 0.2]).reshape((N_p, N_r))  # something we specify
            , 6)
    else:
        r1_jk = np.round(np.ones((N_p, N_r)) *  # branching ratio for single lysogen
            #np.expand_dims(np.linspace(0., 0.2, N_r), 0)  # all phages have same r's
            #np.linspace(0., 0.7, N_p * N_r).reshape((N_p, N_r))  # all different r's
            np.array([[0., 0.2], [0.1] * 2, [0.3] * 2]).reshape((N_p, N_r))  # something we specify
            , 6)
    
    double_lysogen_mask, phage_mask, lysogen_mask, r2_jk, r3_jk, lam_ijk, gamma_ijk, b0, p0, l0, l20 = other_vars_from_params(
        N_b, N_p, N_r, r1_jk, lam, gamma, c1, alpha_l2, p0_rand=p0_rand, p0_init=1e-7, l0_init=1e-2)
    dt, save_every, t_eval, rtol, atol = dt_tol_params(euler_method, t_max, use_t_eval=True)
    dt_tols = [dt, 1e-3, 1e-5]
    
    params = [alpha_b, delta_b, beta, delta_p, alpha_l, delta_l, alpha_l2, delta_l2, 
              lam_ijk, c1, c2, K, r1_jk, r2_jk, r3_jk, gamma_ijk, phage_mask, lysogen_mask, double_lysogen_mask,
              phage_die_when_infecting_lysogen, triple_lysogens_displace_double,
              only_phage_die_with_multiple_lys, log_space]
    
    title = make_title(params, Kinf_pop_floor)
    
    t_vec, b_t, p_t, l_t, l2_t = phage_lysogen_simulation(
        b0, p0, l0, l20, 0, t_max, Kinf_pop_floor, euler_method, dt_tols, params, save_every, t_eval)
    
    t_start = len(t_vec) - int(1000 * doubling_time_lys)
    t_end = len(t_vec)
        
    plt.figure()
    for j in range(N_p):
        for k in range(N_r):
            if phage_mask[j, k]:
                lsty = '-'
                if r1_jk[j, k] ==0:
                    lsty = '--'
                plt.plot(t_vec[t_start:t_end], p_t[j, k, t_start:t_end], linewidth=4,
                         linestyle=lsty, #linestyles[r1_jk[j, k]], 
                         color=colors[j],
                         label=#p_label(j+1, k+1, r1_jk[j, k], include_T_OL=True, include_r=False)
                         p_label(j+1, r1_jk[j, k], r1_jk[j, k], include_T_OL=False, include_r=False)
                         )
        plt.plot(t_vec[t_start:t_end], np.sum(l_t[0, j, :, t_start:t_end], 0), linewidth=1,
                 color=colors[j], label=r'$L_{0}$'.format(j+1), alpha=0.5)
    # Really make blue pop
    j = 0
    for k in range(N_r):
        lsty = '-'
        if r1_jk[j, k] ==0:
            lsty = '--'
        plt.plot(t_vec[t_start:t_end], p_t[j, k, t_start:t_end], linewidth=4,
                 linestyle=lsty, color=colors[j],
                 )
    
    plt.yscale('log')
    plt.xlabel('Time (generations)')
    plt.ylabel('Population')
    plt.yticks([1e0, 1e-2, 1e-4])
    plt.ylim([5e-6, 2e1])
    #plt.ylim([1e-6, 1e2])    
    
    plt.xticks(np.arange(t_start/10, (t_end + 1)/10, 10 * doubling_time_lys), [])
    ax = plt.gca()
    scalebar = AnchoredSizeBar(ax.transData,
                               10 * doubling_time_lys, '10', loc='lower right', 
                               pad=0.5,
                               color='black',
                               frameon=False,
                               label_top=True
                               )
    ax.add_artist(scalebar)
             
    leg = plt.legend(fontsize=13, bbox_to_anchor=(1.05, 1))  # loc='lower right')
    for legobj in leg.legendHandles:
        if legobj.get_linestyle() != '-':
            if legobj.get_linestyle() == '--':
                legobj.set_linewidth(2.5)  # 2.
            else:
                legobj.set_linewidth(2.5)
    if save_figs:
        save_svg_and_png('heterogeneous_Nr_' + str(N_r))
    plt.show()


#%%
# =============================================================================
# Show bunching effect for full model
# =============================================================================
c1 = 1e-5  # 1e-5  # rate of prophage -> cryptic prophage for single lysogen
c2 = 1e-5
alpha_l2 = 0.95 * alpha_l
K = 100  # carrying capacity; also affects initial conditions for bacteria
Kinf = False  # should we set K to infinity? (if it's true, we disregard the value of K set above)

t_max = 1000
euler_method = False

N_p = 4  # number of initial phage species
N_r = 4  # number of branching ratios for each phage species
p0_rand = 2

for diff_l0s in [True]:
    fig_title_append = diff_l0s * '_diff_l0s'

    # r1s = np.array([0., 0.1, 0.2, 0.3])
    r1s = np.array([0., 0.2, 0.3, 0.4])
    
    r1_jk = np.round(np.ones((N_p, N_r)) *  # branching ratio for single lysogen
        np.expand_dims(r1s, 0)  # all phages have same r's
        #np.linspace(0., 0.7, N_p * N_r).reshape((N_p, N_r))  # all different r's
        #np.array([0, 0.1, 0.2]).reshape((N_p, N_r))  # something we specify
        , 6)
    
    double_lysogen_mask, phage_mask, lysogen_mask, r2_jk, r3_jk, lam_ijk, gamma_ijk, b0, p0, l0, l20 = other_vars_from_params(
        N_b, N_p, N_r, r1_jk, lam, gamma, c1, alpha_l2, p0_rand=p0_rand, p0_init=1e-3, l0_init=1e-2)
    
    if diff_l0s:
        np.random.seed(1)
        l0 *= np.exp(7 * np.random.random(l0.shape)) / 100
        l20 *= np.exp(7 * np.random.random(l20.shape)) / 100
        
    dt, save_every, t_eval, rtol, atol = dt_tol_params(euler_method, t_max, use_t_eval=True)
    dt_tols = [dt, rtol, atol]
    
    params = [alpha_b, delta_b, beta, delta_p, alpha_l, delta_l, alpha_l2, delta_l2, 
              lam_ijk, c1, c2, K, r1_jk, r2_jk, r3_jk, gamma_ijk, phage_mask, lysogen_mask, double_lysogen_mask,
              phage_die_when_infecting_lysogen, triple_lysogens_displace_double,
              only_phage_die_with_multiple_lys, log_space]
    
    title = make_title(params, Kinf_pop_floor)
    
    t_vec, b_t, p_t, l_t, l2_t = phage_lysogen_simulation(
        b0, p0, l0, l20, 0, t_max, Kinf_pop_floor, euler_method, dt_tols, params, save_every, t_eval)
    
    
    # THIS FIRST FIGURE NEEDS A BUNCH MORE WORK
    # t_start = 2400
    # t_end = t_start + int(700 * doubling_time_lys) #8811
        
    # plt.figure()# figsize=(8, 6))
    # for j in [1, 2, 3, 0]:
    #     k_list = [1, 2, 3, 0]  # k_list = range(N_r)
    #     for k in k_list:
    #         if phage_mask[j, k]:
    #             if k_list == [1, 2, 3, 0]:
    #                 k_mod = [4, 1, 2, 3]
    #             else:
    #                 k_mod = [1, 2, 3, 4]
    
    #             label = ''
    
    #             plt.plot(t_vec[t_start:t_end], p_t[j, k, t_start:t_end], linewidth=4 - 2 * (j > 0),
    #                      linestyle=linestyles[r1_jk[j, k]], color=colors[j],
    #                      label=label, 
    #                      alpha=1 * 0.55**(j))
                
    #             if diff_l0s and r1_jk[j, k] > 0:
    #                 plt.plot(t_vec[t_start:t_end], l_t[0, j, k, t_start:t_end], linewidth=1 - 0.5 * (j > 0),
    #                          linestyle=linestyles[r1_jk[j, k]], color=colors[j], #label=r'$L_{0}$'.format(j+1), 
    #                          alpha=0.5 * (j==0) + 0.25*(j > 0))
    #                 if alpha_l2 > 0: 
    #                     for j2 in range(N_p):
    #                         for k2 in range(N_r):
    #                             if np.any(double_lysogen_mask[0, j, :, j2, :]):
    #                                 plt.plot(t_vec[t_start:t_end], l2_t[0, j, k, j2, k2, t_start:t_end], 
    #                                          linewidth=1 - 0.5 * (j > 0), linestyle=linestyles[r1_jk[j, k]],  
    #                                          color=double_lysogen_colors[j][j2], 
    #                                          alpha=0.5 * (j==0) + 0.25*(j > 0),
    #                                          #label=r'$D_{{{0}, {1}}}$'.format(j2 + 1, j + 1)
    #                                          )
    #     if not diff_l0s:
    #         plt.plot(t_vec[t_start:t_end], np.sum(l_t[0, j, :, t_start:t_end], 0), linewidth=1 - 0.5 * (j > 0),
    #                  color=colors[j], #label=r'$L_{0}$'.format(j+1), 
    #                  alpha=0.5 * (j==0) + 0.25*(j > 0))
    #         if alpha_l2 > 0: 
    #             for j2 in range(N_p):
    #                 if np.any(double_lysogen_mask[0, j, :, j2, :]):
    #                     plt.plot(t_vec[t_start:t_end], np.sum(l2_t[0, j, :, j2, :, t_start:t_end], (0,1)), 
    #                              linewidth=1 - 0.5 * (j > 0), linestyle=':', 
    #                              color=double_lysogen_colors[j][j2], 
    #                              alpha=0.5 * (j==0) + 0.25*(j > 0),
    #                              #label=r'$D_{{{0}, {1}}}$'.format(j2 + 1, j + 1)
    #                              )
    # if c1 > 0:
    #     plt.plot(t_vec[t_start:t_end], b_t[0, t_start:t_end], color='k', linestyle=':', linewidth=2,
    #              label='$S$')
    
    # plt.yscale('log')
    # plt.xlabel('Time (generations)')
    # plt.ylabel('Population')
    # plt.yticks([1e0, 1e-4, 1e-8])
    # plt.ylim([2e-10, 5e1])
    
    # plt.xticks(np.arange(t_start/10, (t_end + 1)/10, 10 * doubling_time_lys), [])
    # ax = plt.gca()
    # scalebar = AnchoredSizeBar(ax.transData,
    #                            10 * doubling_time_lys, '10', loc='lower right', 
    #                            pad=0.5,
    #                            color='black',
    #                            frameon=False,
    #                            label_top=True
    #                            )
    # ax.add_artist(scalebar)
    
    # # Make legend labels; TODO: Still need to make legend labels for double lysogens & sensitives
    # j = 0
    # k_list = range(N_r) #[1, 2, 3, 0]
    # for k in k_list:
    #     if phage_mask[j, k]:
    #         if k_list == [1, 2, 3, 0]:
    #             k_mod = [4, 1, 2, 3]
    #         else:
    #             k_mod = [1, 2, 3, 4]
    #         label = p_label('c', r1_jk[j, k], r1_jk[j, k], include_T_OL=False, include_r=False)
            
    #         plt.plot([t_vec[t_end]] * 2, [1e-8] * 2, color='k', 
    #                  linestyle=linestyles[r1_jk[j, k]], 
    #                  label=label, linewidth=0
    #                  )
    #         if diff_l0s and r1_jk[j, k] > 0:
    #             plt.plot([t_vec[t_end]] * 2, [1e-8] * 2, linewidth=0,
    #                      linestyle=linestyles[r1_jk[j, k]], color='k',
    #                      label=l_label('c', r1_jk[j, k], r1_jk[j, k], include_T_OL=False, include_r=False))
    #             # if alpha_l2 > 0: 
    #             #     for j2 in range(N_p):
    #             #         for k2 in range(N_r):
    #             #             if np.any(double_lysogen_mask[0, j, :, j2, :]):
    #             #                 plt.plot([t_vec[t_end]] * 2, [1e-8] * 2, linewidth=0,
    #             #                          linewidth=0, linestyle=linestyles[r1_jk[j, k]],  
    #             #                          color=double_lysogen_colors[j][j2], 
    #             #                          alpha=0.5 * (j==0) + 0.25*(j > 0),
    #             #                          #label=r'$D_{{{0}, {1}}}$'.format(j2 + 1, j + 1)
    #             #                          )
    # if not diff_l0s:
    #     plt.plot([t_vec[t_end]] * 2, [1e-8] * 2, color='k', 
    #              linestyle='-', 
    #              label=r'$L_{c}$', linewidth=0
    #              )
    #     # if alpha_l2 > 0: 
    #     #     for j2 in range(N_p):
    #     #         if np.any(double_lysogen_mask[0, j, :, j2, :]):
    #     #             plt.plot(t_vec[t_start:t_end], np.sum(l2_t[0, j, :, j2, :, t_start:t_end], (0,1)), 
    #     #                      linewidth=1 - 0.5 * (j > 0), linestyle=':', 
    #     #                      color=double_lysogen_colors[j][j2], 
    #     #                      alpha=0.5 * (j==0) + 0.25*(j > 0),
    #     #                      #label=r'$D_{{{0}, {1}}}$'.format(j2 + 1, j + 1)
    #     #                      )

    # for j in range(N_p):
    #     plt.plot([t_vec[t_end]] * 2, [1e-8] * 2, color=colors[j],
    #              linestyle='-', linewidth=0, label=r'$c={0}$'.format(j+1))
    
    # leg = plt.legend(fontsize=13, bbox_to_anchor=(1.05,1), ncol=1+diff_l0s)#, handlelength=2) #loc='best') #
    # for legobj in leg.legendHandles:
    #     if legobj.get_linestyle() != '-':
    #         if legobj.get_linestyle() == '--':
    #             legobj.set_linewidth(2.5)  # 2.
    #         elif legobj.get_label() in [r'$L_{c}$'] + [
    #                 l_label('c', r1, r1, include_T_OL=False, include_r=False) for r1 in r1s]:
    #             legobj.set_linewidth(1)
    #         else:
    #             legobj.set_linewidth(2.5)
    #     elif legobj.get_label() in [r'$L_{c}$'] + [
    #             l_label('c', r1, r1, include_T_OL=False, include_r=False) for r1 in r1s]:
    #         legobj.set_linewidth(1)
    #     elif legobj.get_label() == r'$c=2$' or legobj.get_label() == r'$c=3$':
    #         legobj.set_linewidth(1.5)
    #     else:
    #         legobj.set_linewidth(4)
            
    # if save_figs:
    #     save_svg_and_png('bunching_withlyt_Np3_Nr4' + fig_title_append + '_full_model')
    # plt.show()
    
    
    
    # Make figure of ratio of phage to lysogen populations
    t_start = 4000    
    t_end = t_start + int(900 * doubling_time_lys)
    
    plt.figure()# figsize=(8, 6))
    for j in [1, 2, 3, 0]:
        k_list = [1, 2, 3]  # k_list = range(N_r)
        for k in k_list:
            if phage_mask[j, k]:
                if k_list == [1, 2, 3]:
                    k_mod = [1, 2, 3]
                else:
                    k_mod = [1, 2, 3, 4]
    
                label = ''
                plt.plot(t_vec[t_start:t_end], 
                         p_t[j, k, t_start:t_end] / (l_t[0, j, k, t_start:t_end] + 
                                                     np.sum(l2_t[0, j, k, :, :, t_start:t_end], (0, 1)) + 
                                                     np.sum(l2_t[0, :, :, j, k, t_start:t_end], (0, 1))), 
                         linewidth=4 - 2 * (j > 0), 
                         linestyle=linestyles[r1_jk[j, k]], 
                         color=colors[j],
                         label=label, 
                         alpha=1 * 0.55**(j))
    plt.yscale('log')
    plt.xlabel('Time (generations)')
    plt.ylabel('Population ratio')
    plt.yticks([1e-3, 1e0, 1e3])
    plt.ylim([2e-5, 5e3])
    plt.xticks(np.arange(t_start/10, (t_end + 1)/10, 10 * doubling_time_lys), [])
    ax = plt.gca()
    scalebar = AnchoredSizeBar(ax.transData,
                               10 * doubling_time_lys, '10', loc='lower right', 
                               pad=0.5,
                               color='black',
                               frameon=False,
                               label_top=True
                               )
    ax.add_artist(scalebar)
    
    # Make legend labels
    j = 0
    k_list = [1, 2, 3]  # k_list = range(N_r)
    for k in k_list:
        if phage_mask[j, k]:
            if k_list == [1, 2, 3, 0]:
                k_mod = [4, 1, 2, 3]
            else:
                k_mod = [1, 2, 3, 4]
            # label = (r'$P_{{c,{0}}} / L_{{c, {0}}}$'.format(k_mod[k]))
            # if r1_jk[j, k] > 0.:
            #     label += '; ' + r'$f=$' + "{:.1f}".format(r1_jk[j, k])            
            # label = (r"$\frac{{P_{{c,{0}}}}}{{L_{{c,{0}}} + \sum_{{c',f'}} D_{{c,{0},c',f'}} }}$".format(r1_jk[j, k]))
            label = (r"$P_{{c,{0}}}\ / \left(L_{{c,{0}}} + \sum_{{c',f'}} D_{{c,{0},c',f'}} \right)$".format(r1_jk[j, k]))
            
            plt.plot([t_vec[t_end]] * 2, [1e-8] * 2, color='k', 
                     linestyle=linestyles[r1_jk[j, k]], 
                     label=label, linewidth=0
                     )
    for j in range(N_p):
        plt.plot([t_vec[t_end]] * 2, [1e-8] * 2, color=colors[j],
                 linestyle='-', linewidth=0, label=r'$c={0}$'.format(j+1))
    
    leg = plt.legend(fontsize=13, bbox_to_anchor=(1.05,1))#, handlelength=2) #loc='best') #
    for legobj in leg.legendHandles:
        if legobj.get_linestyle() != '-':
            if legobj.get_linestyle() == '--':
                legobj.set_linewidth(2.5)  # 2.
            else:
                legobj.set_linewidth(2.5)
        elif legobj.get_label() == r'$L_{c}$':
            legobj.set_linewidth(1)
        elif legobj.get_label() == r'$c=2$' or legobj.get_label() == r'$c=3$':
            legobj.set_linewidth(1.5)
        else:
            legobj.set_linewidth(4)
            
    for with_pred_line in [False, True]:
        if with_pred_line:
            plt.plot(t_vec[t_start:t_end],
                     beta * lam / (delta_p #+ gamma * np.sum(l_t[0, 0, :, t_start:t_end], 0)
                                   * np.array([1 for _ in t_vec[t_start:t_end]])
                                   ), 
                     '--k')#, label=r'$b \gamma / \delta$')
    
        if save_figs:
            save_svg_and_png('bunching_frac_Np3_Nr4_pred' + fig_title_append + '_full_model')
    plt.show()

c1 = 0
c2 = 0
alpha_l2 = 0
K = 1000  # carrying capacity; also affects initial conditions for bacteria
Kinf = True  # should we set K to infinity? (if it's true, we disregard the value of K set above)
pop_floor = 1e-20  # if Population decreases below this number, set it to 0
Kinf_pop_floor = [Kinf, pop_floor]


#%%
# =============================================================================
# Show that chaos is really long-lasting
# =============================================================================

t_max = int(1e6 * doubling_time_lys) + 1
euler_method = False
include_het = False

N_p = 3  # number of initial phage species
p0_rand = 1
c1 = 1e-5  # rate of prophage -> cryptic prophage for single lysogen

for N_r in [1]:  #[1, 2]:
    if not include_het:
        if N_r == 1:
            r1_jk = np.round(np.ones((N_p, N_r)) *  # branching ratio for single lysogen
                np.expand_dims(np.linspace(0.2, 0.2, N_r), 0)  # all phages have same r's
                , 6)
        else:
            r1_jk = np.round(np.ones((N_p, N_r)) *  # branching ratio for single lysogen
                np.array([[0., 0.2], [0.2] * 2, [0.2] * 2]).reshape((N_p, N_r))  # something we specify
                , 6)
    if include_het:
        if N_r == 1:
            r1_jk = np.round(np.ones((N_p, N_r)) *  # branching ratio for single lysogen
                np.linspace(0.1, 0.3, N_p * N_r).reshape((N_p, N_r))  # all different r's
                , 6)
        else:
            r1_jk = np.round(np.ones((N_p, N_r)) *  # branching ratio for single lysogen
                np.array([[0., 0.2], [0.1] * 2, [0.3] * 2]).reshape((N_p, N_r))  # something we specify
                , 6)
    
    double_lysogen_mask, phage_mask, lysogen_mask, r2_jk, r3_jk, lam_ijk, gamma_ijk, b0, p0, l0, l20 = other_vars_from_params(
        N_b, N_p, N_r, r1_jk, lam, gamma, c1, alpha_l2, p0_rand=p0_rand, p0_init=1e-7, l0_init=1e-2)
    dt, save_every, t_eval, rtol, atol = dt_tol_params(euler_method, t_max, use_t_eval=True)  # changing this to False doesn't speed it up
    dt_tols = [dt, rtol, atol]
    
    params = [alpha_b, delta_b, beta, delta_p, alpha_l, delta_l, alpha_l2, delta_l2, 
              lam_ijk, c1, c2, K, r1_jk, r2_jk, r3_jk, gamma_ijk, phage_mask, lysogen_mask, double_lysogen_mask,
              phage_die_when_infecting_lysogen, triple_lysogens_displace_double,
              only_phage_die_with_multiple_lys, log_space]
    
    title = make_title(params, Kinf_pop_floor)
    
    t_vec, b_t, p_t, l_t, l2_t = phage_lysogen_simulation(
        b0, p0, l0, l20, 0, t_max, Kinf_pop_floor, euler_method, dt_tols, params, save_every, t_eval)
    
    
    
    t_start = len(t_vec) - np.argmin(np.abs(t_vec - 200 * doubling_time_lys))
    tick_space = 100
    t_end = len(t_vec)
        
    plt.figure()
    for j in range(N_p):
        for k in range(N_r):
            if phage_mask[j, k]:
                plt.plot(t_vec[t_start:t_end], p_t[j, k, t_start:t_end], linewidth=4,
                         linestyle=all_solid_temperate_linestyles[r1_jk[j, k]], color=colors[j],
                         label=#p_label(j+1, k+1, r1_jk[j, k], include_T_OL=True, include_r=False)
                         p_label(j+1, r1_jk[j, k], r1_jk[j, k], include_T_OL=False, include_r=False)
                         )
        plt.plot(t_vec[t_start:t_end], np.sum(l_t[0, j, :, t_start:t_end], 0), linewidth=1,
                 color=colors[j], label=r'$L_{0}$'.format(j+1), alpha=0.5)
    if c1 > 0:
        plt.plot(t_vec[t_start:t_end], b_t[0, t_start:t_end], color='k', linestyle=':', linewidth=2,
                 label='$S$')
    # Really make blue pop
    j = 0
    for k in range(N_r):
        plt.plot(t_vec[t_start:t_end], p_t[j, k, t_start:t_end], linewidth=4,
                 linestyle=all_solid_temperate_linestyles[r1_jk[j, k]], color=colors[j],
                 )
    
    plt.yscale('log')
    plt.xlabel('Time (generations)')
    plt.ylabel('Population')
    if N_r == 1 and N_p == 3 and not include_het:
        if c1 == 0:
            plt.yticks([1e0, 1e-2, 1e-4, 1e-6])
            plt.ylim([1e-7, 1e2])
        else:
            plt.yticks([1e0, 1e-2, 1e-4, 1e-6])
            plt.ylim([2e-8, 5e1])
    else:
        plt.yticks([1e0, 1e-2, 1e-4])
        plt.ylim([5e-6, 2e1])
    #plt.ylim([1e-6, 1e2])    
    
    plt.xticks(np.arange(t_vec[t_start], t_vec[t_end - 1] + 1, tick_space * doubling_time_lys), 
               [int(i / doubling_time_lys) - 1 for i in np.arange(t_vec[t_start], t_vec[t_end - 1] + 1, tick_space * doubling_time_lys)])
             
    leg = plt.legend(fontsize=13, bbox_to_anchor=(1.05, 1))  # loc='lower right')
    for legobj in leg.legendHandles:
        if legobj.get_linestyle() != '-':
            if legobj.get_linestyle() == '--':
                legobj.set_linewidth(2.5)  # 2.
            else:
                legobj.set_linewidth(2.5)
    if save_figs:
        save_svg_and_png('long_time_coexistence_Nr_' + str(N_r) + '_Np_' + str(N_p) + 
                         '_het' * include_het + '_sensitives' * (c1 > 0))
    plt.show()


c1 = 0

#%% Check the basin of attraction of the oscillatory steady state found for N_r=1, N_p=3

t_max = int(1e3 * doubling_time_lys) + 1
euler_method = False

N_p = 3  # number of initial phage species
p0_rand = 1

N_r = 1
r1_jk = np.round(np.ones((N_p, N_r)) *  # branching ratio for single lysogen
    np.expand_dims(np.linspace(0.2, 0.2, N_r), 0)  # all phages have same r's
    #np.linspace(0., 0.7, N_p * N_r).reshape((N_p, N_r))  # all different r's
    #np.array([0, 0.2]).reshape((N_p, N_r))  # something we specify
    , 6)

double_lysogen_mask, phage_mask, lysogen_mask, r2_jk, r3_jk, lam_ijk, gamma_ijk, b0, p0, l0, l20 = other_vars_from_params(
    N_b, N_p, N_r, r1_jk, lam, gamma, c1, alpha_l2, p0_rand=p0_rand, p0_init=1e-7, l0_init=1e-2)
dt, save_every, t_eval, rtol, atol = dt_tol_params(euler_method, t_max, use_t_eval=False)
dt_tols = [dt, rtol, atol]

# Initialize at oscillatory solution:
p0_osc = np.array([[1.03610085e-05], [4.03031436e-03], [1.27465228e-01]])
l0_osc = np.array([[[1.07457259e-04], [2.66964764e-06], [1.60443746e-02]]])

np.random.seed(p0_rand)
l0_p0_var = 1.5  # multiply l0 and p0 by random numbers ranging uniformly in log-space from 1/l0_p0_var to l0_p0_var
l0 = l0_osc * 10**(np.log10(l0_p0_var) * (-1 + 2 * np.ones(l0.shape)))
p0 = p0_osc * 10**(np.log10(l0_p0_var) * (-1 + 2 * np.ones(p0.shape)))
# l0 = l0_osc * (1 + 0.5 * np.random.random(l0.shape))
# p0 = p0_osc * (1 + 0.5 * np.random.random(p0.shape))

params = [alpha_b, delta_b, beta, delta_p, alpha_l, delta_l, alpha_l2, delta_l2, 
          lam_ijk, c1, c2, K, r1_jk, r2_jk, r3_jk, gamma_ijk, phage_mask, lysogen_mask, double_lysogen_mask,
          phage_die_when_infecting_lysogen, triple_lysogens_displace_double,
          only_phage_die_with_multiple_lys, log_space]

title = make_title(params, Kinf_pop_floor)

t_vec, b_t, p_t, l_t, l2_t = phage_lysogen_simulation(
    b0, p0, l0, l20, 0, t_max, Kinf_pop_floor, euler_method, dt_tols, params, save_every, t_eval)

t_start = len(t_vec) - np.argmin(np.abs(t_vec - 400 * doubling_time_lys))
tick_space = 200
t_end = len(t_vec)
    
plt.figure()
for j in range(N_p):
    for k in range(N_r):
        if phage_mask[j, k]:
            plt.plot(t_vec[t_start:t_end], p_t[j, k, t_start:t_end], linewidth=4,
                     linestyle=linestyles[r1_jk[j, k]], color=colors[j],
                     label=#p_label(j+1, k+1, r1_jk[j, k], include_T_OL=True, include_r=False)
                     p_label(j+1, r1_jk[j, k], r1_jk[j, k], include_T_OL=False, include_r=False)
                     )
    plt.plot(t_vec[t_start:t_end], np.sum(l_t[0, j, :, t_start:t_end], 0), linewidth=1,
             color=colors[j], label=r'$L_{0}$'.format(j+1), alpha=0.5)
# Really make blue pop
j = 0
for k in range(N_r):
    plt.plot(t_vec[t_start:t_end], p_t[j, k, t_start:t_end], linewidth=4,
             linestyle=linestyles[r1_jk[j, k]], color=colors[j],
             )

plt.yscale('log')
plt.xlabel('Time (generations)')
plt.ylabel('Population')
if N_r == 1:
    plt.yticks([1e0, 1e-2, 1e-4, 1e-6])
    plt.ylim([1e-7, 1e2])
else:
    plt.yticks([1e0, 1e-2, 1e-4])
    plt.ylim([5e-6, 2e1])
#plt.ylim([1e-6, 1e2])    

plt.xticks(np.arange(t_vec[t_start], t_vec[t_end - 1] + 1, tick_space * doubling_time_lys), 
           [int(i / doubling_time_lys) - 1 for i in np.arange(t_vec[t_start], t_vec[t_end - 1] + 1, tick_space * doubling_time_lys)])
         
leg = plt.legend(fontsize=13, bbox_to_anchor=(1.05, 1))  # loc='lower right')
for legobj in leg.legendHandles:
    if legobj.get_linestyle() != '-':
        if legobj.get_linestyle() == '--':
            legobj.set_linewidth(2.5)  # 2.
        else:
            legobj.set_linewidth(2.5)
if save_figs:
    save_svg_and_png('long_time_coexistence_osc_Nr_' + str(N_r))
plt.show()
