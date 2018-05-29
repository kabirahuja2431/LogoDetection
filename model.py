
# coding: utf-8

# In[24]:


import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
import skimage.io
from misc import *
from scipy.misc import imresize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from global_vars import *



def forward_pass(x,y):
    #Xavier Initializer to get better initialization of weights of the network
    initializer = tf.contrib.layers.xavier_initializer()
    #Convolutional Layers
    a1 = tf.layers.conv2d(x,32,3,padding = 'SAME',activation=tf.nn.relu,kernel_initializer=initializer)
    p1 = tf.nn.max_pool(a1,[1,2,2,1],[1,2,2,1],padding='SAME')
    a2 = tf.layers.conv2d(p1,64,3,padding='SAME',activation=tf.nn.relu,kernel_initializer=initializer)
    p2 = tf.nn.max_pool(a2,[1,2,2,1],[1,2,2,1],padding = 'SAME')
    a3 = tf.layers.conv2d(p2,128,3,padding='SAME',activation=tf.nn.relu,kernel_initializer=initializer)
    p3 = tf.nn.max_pool(a3,[1,2,2,1],[1,2,2,1],padding = 'SAME')
    #Fully Connected Layers
    flat = tf.contrib.layers.flatten(p3)
    d1 = tf.layers.dense(flat,2048,activation = tf.nn.relu)
    d2 = tf.layers.dense(d1,NUM_CLASSES)
    preds = tf.nn.softmax(d2)
    #Loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=d2,labels=y))
    
    return preds,loss




def get_solver(lr):
    return tf.train.AdamOptimizer(learning_rate=lr)



