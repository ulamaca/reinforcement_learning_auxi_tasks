#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:19:10 2017

@author: klee
"""

import numpy as np
from sklearn.decomposition import PCA
import os

def features_pca(data_path, N_pc = 10):
    """ loading data from frames in atari games and aiming at learning representation for the frames
    """
    # 1 Data loading
    data = np.load(data_path)
    data = np.squeeze(data)
    
    print(" data shape: " + str(data.shape) )    
    
    N_frames = data.shape[0] # number of frames collected
    print("Total number of frames for training PCA: %i " % N_frames)
    
    # 2 PCA and Visualization of feature    
    pca = PCA(n_components=N_pc)
    pca.fit(data)
    print(pca.explained_variance_ratio_, " variation explained by all PCs: %f " % np.sum(pca.explained_variance_ratio_))
    params = pca.get_params()
    data_pca = pca.fit_transform(data)
        # No visualization yet because we need far more than 2 PCs to explain >50% variablity
            
    return params # data_pca


if __name__ == '__main__':
    N_pc = 350
    dir_path = '/is/sg/klee/repo1/atari-state-representation-learning/pong_expPCA/model-atari-pong-1/exp_in_total_010_episodes/'
    data_name = 'finConvAct.npy'
    data_path = os.path.join(dir_path, data_name)
    params_feature_pca = features_pca(data_path, N_pc ) #, can check for data_pca.shape
    np.save( os.path.join(dir_path, 'pca_params.npy' ), params_feature_pca)
                                 
