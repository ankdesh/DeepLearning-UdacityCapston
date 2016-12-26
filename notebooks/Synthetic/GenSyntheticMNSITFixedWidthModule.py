
# coding: utf-8

# In[1]:

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt
import random
import numpy as np


# In[2]:

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=False)
X = X.reshape([-1, 28, 28])
testX = testX.reshape([-1, 28, 28])


# In[3]:

# Function to genterate synthetic benchmark. Takes num of digits to concat as parameter and number of data points to generate
# Returns a tuple of (list of images and target value 

def getDataSet(num_digits, num_samples):
    X_seq = np.empty(shape=(num_samples, 28, 28 * num_digits),dtype='float32')
    Y_seq = np.empty(shape=(num_samples, num_digits),dtype='uint8')
    for i in range(num_samples): # For each sample to generate
        indices = np.random.randint(0,len(Y) - 1, size=num_digits) # generate indices for creating this wide image
        X_seq[i] = np.concatenate(X[indices], axis=1)
        Y_seq[i] = Y[indices]
    return (X_seq, Y_seq)


# In[4]:

def testDataSetGen():
    get_ipython().magic(u'matplotlib inline')
    gen_X, gen_Y = getDataSet(3, 100)
    print (gen_Y[99])
    plt.imshow(gen_X[99], cmap='Greys')


# In[ ]:



