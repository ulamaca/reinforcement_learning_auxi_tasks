# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 17:26:15 2017

@author: ulamaca
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

filename = 'error_summary.pickle'
with open(filename,'rb') as f:    
    [errors, stds_errors, performance, num_iters] = pickle.load(f) # for those results with both error and var information

errors = np.vstack(errors)
stds_errors = np.vstack(stds_errors)
performance = np.vstack(performance)
num_iters = np.vstack(num_iters)

tds, s_tds = errors[:,0], stds_errors[:,0]
rpes, s_rpes = errors[:,1], stds_errors[:,1]
spes, s_spes = errors[:,2], stds_errors[:,2]
rews, s_rews = performance[:,0], performance[:,1]

f, axarr = plt.subplots(4, sharex=True)
axarr[0].set_title('td error')
axarr[0].errorbar(num_iters, tds, yerr=s_tds, fmt='k', ecolor='r')
axarr[1].set_title('rpe')
axarr[1].errorbar(num_iters, rpes, yerr=s_rpes, fmt='k', ecolor='r') # plot with error_bar
axarr[2].set_title('spe')
axarr[2].errorbar(num_iters, spes, yerr=s_spes, fmt='k', ecolor='r') # plot with error_bar
axarr[3].set_title('Average Episode Reward')
axarr[3].errorbar(num_iters, rews, yerr=s_rews, fmt='k', ecolor='r')
