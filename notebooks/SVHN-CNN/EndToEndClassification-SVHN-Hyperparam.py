
# coding: utf-8

# In[1]:

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np
import matplotlib.pyplot as plt
import sklearn.utils
import tensorflow as tf
import h5py
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation


# In[2]:

IMG_WIDTH = 32 # Side for each transformed Image
IMG_HEIGHT = 32
IMG_DEPTH = 1 # RGB files


# In[3]:

MAX_DIGITS = 5


# In[4]:

imgsAll = np.empty(shape = (0,IMG_HEIGHT, IMG_WIDTH), dtype=float)
labelsAll = np.empty(shape = (0,MAX_DIGITS), dtype=float)
numDigitsAll = np.empty(shape = (0), dtype=float)


# In[5]:

for numDigits in range(1,MAX_DIGITS + 1):
    h5FileName = 'svhn_' + str(numDigits) + '.h5'
    data = h5py.File(h5FileName)
    imgs = np.array(data['images']).astype(float)
    labels = np.array(data['digits'])
    # Buff up labels to MAX_DIGITS width ( use 10 for undefined value)
    valsToFill = np.full(shape = (labels.shape[0], MAX_DIGITS - numDigits ), fill_value= 10.0, dtype = float)
    labels = np.concatenate ((labels, valsToFill), axis = 1)
    # Concat to full Dataset
    imgsAll = np.concatenate((imgsAll, imgs), axis = 0)
    labelsAll = np.concatenate((labelsAll, labels), axis = 0)
    numDigitsAll = np.concatenate((numDigitsAll, np.full(labels.shape[0], numDigits, dtype= float))) # Add num of digits for this set of images


# In[6]:

print (imgsAll.shape)
print (labelsAll.shape)
print (numDigitsAll.shape)


# In[7]:

print (labelsAll[100000])
plt.imshow(imgsAll[100000], cmap='gray')
print (numDigitsAll[100000])


# In[8]:

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    index_update = [int(x) for x in index_offset + labels_dense]
    labels_one_hot.flat[index_update] = 1
    return labels_one_hot


# In[9]:

# Get the dataset
X = imgsAll.reshape([-1, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
Y = labelsAll


# In[10]:

#X, Y = sklearn.utils.shuffle(X, Y, random_state=0)


# In[11]:

# Generate validation set
ratio = 0.9 # Train/Test set
randIdx = np.random.random(imgsAll.shape[0]) <= ratio
#print (sum(map(lambda x: int(x), randIdx)))
X_train = X[randIdx]
Y_train = Y[randIdx]
X_test = X[randIdx == False]
Y_test = Y[randIdx == False]
Y_train = [dense_to_one_hot(Y_train[:,idx], num_classes= 11) for idx in range(Y_train.shape[1])] 
Y_test = [dense_to_one_hot(Y_test[:,idx], num_classes= 11) for idx in range(Y_test.shape[1])] 
#del X, Y # release some space


# In[12]:

print (X_train.shape)
print (Y_train[0].shape)


# In[15]:

# Building convolutional network

# Building convolutional network
for dropOutProb in [0.5, 0.7, 0.8, 0.9]: 
    for optimizer in ['SGD', 'RMSProp', 'Adam']:
        for learning_rate in [0.01, 0.001, 0.0001]: 
            with tf.Graph().as_default():

                # Real-time data preprocessing
                img_prep = ImagePreprocessing()
                img_prep.add_featurewise_zero_center()
                img_prep.add_featurewise_stdnorm()

                # Real-time data augmentation
                img_aug = ImageAugmentation()
                #img_aug.add_random_flip_leftright()
                img_aug.add_random_rotation(max_angle=25.)
                input = input_data(shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH], name='input',
                                                                            data_preprocessing=img_prep,
                                                                            data_augmentation=img_aug)

                # Building convolutional network
                x = tflearn.conv_2d(input, 32, 3, activation='relu', name='conv1_1')
                x = tflearn.conv_2d(x, 32, 3, activation='relu', name='conv1_2')
                x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')
                x = dropout(x, dropOutProb)


                x = tflearn.conv_2d(x, 64, 3, activation='relu', name='conv2_1')
                x = tflearn.conv_2d(x, 64, 3, activation='relu', name='conv2_2')
                x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')
                x = dropout(x, dropOutProb)


                x = tflearn.conv_2d(x, 256, 3, activation='relu', name='conv3_1')
                x = tflearn.conv_2d(x, 256, 3, activation='relu', name='conv3_2')
                x = tflearn.conv_2d(x, 256, 3, activation='relu', name='conv3_3')
                x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')
                x = dropout(x, dropOutProb)

                # Training heads
                allHeads = []
                for idx in range(MAX_DIGITS):
                    fc = fully_connected(x, 1024, activation='tanh')
                    fc = dropout(fc, dropOutProb)
                    softmax = fully_connected(fc, 11, activation='softmax')
                    networkOut = regression(softmax, optimizer=optimizer, learning_rate=learning_rate,
                                 loss='categorical_crossentropy', name='target' + str(idx))
                    allHeads.append(networkOut)

                network = tflearn.merge(allHeads, mode='elemwise_sum')

                model = tflearn.DNN(network, tensorboard_verbose=0)
                feedTrainDict = {'target'+ str(i): Y_train[i] for i in range(MAX_DIGITS)}
                #feedTrainDict = {'target0': Y_train[0]}
                feedTestList =  [Y_test[i] for i in range(MAX_DIGITS)]
                #feedTestList =  Y_test[0]
                logDirName = 'EndToEndHyper/svhn_' + str(dropOutProb) + '_' + optimizer + '_' + str(learning_rate)
                model.fit({'input': X_train}, feedTrainDict, shuffle = True,
                          validation_set= (X_test, feedTestList), n_epoch=10, show_metric=True, snapshot_step=1000,
                          run_id=logDirName)


# In[ ]:



