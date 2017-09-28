# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 17:26:15 2017

@author: ulamaca
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

N = []
filename = 'error_summary_' + str(N) + '.pickle'
filename = 'error_summary_.pickle'
with open(filename,'rb') as f:    
    [error, variance, performance] = pickle.load(f) # for those results with both error and var information

error = np.asarray(error)
variance = np.asarray(variance)    
num_iters = error[:,1]    
N_iters = num_iters[-1]
error = error[:,0]
variance = variance[:,0]

tds = np.zeros_like(num_iters)
v_tds = np.zeros_like(num_iters)
rpes = np.zeros_like(num_iters)
v_rpes = np.zeros_like(num_iters)
spes = np.zeros_like(num_iters)
v_spes = np.zeros_like(num_iters)

for i in range(len(error)):
    tds[i] = error[i][0]
    v_tds[i] = variance[i][0]
    rpes[i] = error[i][1]
    v_rpes[i] = variance[i][1]
    spes[i] = error[i][2]
    v_spes[i] = variance[i][2]

# Performance-related processing
performance = np.asarray(performance)
num_iters_per = performance[:,1]
temp = performance[:,0]
rews = np.zeros_like(num_iters_per)
v_rews = np.zeros_like(num_iters_per)
for i in range(len(num_iters_per)):
    rews[i] = temp[i][0]
    v_rews[i] = temp[i][1]    
    
f, axarr = plt.subplots(4, sharex=True)
#axarr[0].plot(num_iters, tds, 'r')
axarr[0].errorbar(num_iters, tds, yerr=v_tds, fmt='k', ecolor='r')
axarr[0].set_title('td error' + ', num_iters = ' + str(N_iters))
#axarr[1].plot(num_iters, rpes, 'k*')
axarr[1].errorbar(num_iters, rpes, yerr=v_rpes, fmt='k', ecolor='r') # plot with error_bar
axarr[1].set_title('rpe' + ', num_iters = ' + str(N_iters))
#axarr[2].plot(num_iters, spes, 'b*')
axarr[2].set_title('spe' + ', num_iters = ' + str(N_iters))
axarr[2].errorbar(num_iters, spes, yerr=v_spes, fmt='k', ecolor='r') # plot with error_bar
axarr[3].set_title('Average Episode Reward' + ', num_iters = ' + str(N_iters))
axarr[3].errorbar(num_iters_per, rews, yerr=v_rews, fmt='k', ecolor='r')