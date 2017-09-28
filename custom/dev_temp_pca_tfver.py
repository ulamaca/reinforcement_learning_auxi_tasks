#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:19:35 2017

@author: klee
    This code is not functional but just a demonstration of some of my testing
"""

import numpy as np
from sklearn.decomposition import PCA
import os
import tensorflow as tf

N_pc = 100
dir_path = '/is/sg/klee/repo1/atari-state-representation-learning/pong_expPCA/model-atari-pong-1/exp_in_total_010_episodes/'
data_name = 'fstFullAct.npy'
data_path = os.path.join(dir_path, data_name)
data = np.load(data_path)
data = np.squeeze(data)
feature_dim = data.shape[1]

data_new = np.random.normal(size=[100,512])
test_pca = PCA(n_components=10)
test_pca.fit(data)
data_new_dr = test_pca.transform(data_new)
print(inspect.getsource(PCA.transform))
print(data_new_dr.shape)

N_data_new = 100
U = test_pca.components_
sess = tf.Session()
matrix = tf.constant(U.T)
x = tf.ones([1, feature_dim], dtype='float64')
x_pca_tf = sess.run(tf.matmul(x, matrix) )
x_pca_np = sess.run(tf.matmul(x-test_pca.mean_, U.T)) # I can just put tf object to multiply with np-array.
                  

# Study1: get sourcecode of PCA                   
print(inspect.getsource(PCA)) # getting source code
     
# Study2: (Wrong example) Testing if PCA.transform works for loaded parameters
params = np.load( os.path.join(dir_path, 'pca_params.npy' )).item()
test_pca_2 = PCA()
test_pca_2.set_params(**params) # not able to load the .components_ property from this simple 
test_pca_2.transform(data_new) # therefore, unable to DR new data

# Playing with tensor-flow
# play1: (Wrong example) doing matrix multiplication before tf.sess
test_tf = fast_dot(x, U.T)
print(type(test_tf))
print(sess.run(test_tf)) # so, using fast_dot before running session fails