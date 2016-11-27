
# coding: utf-8

# In[1]:

import tensorflow as tf
import os
import random
import pylab
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image


# In[3]:

DATA_FOLDER = '/home/ankdesh/explore/DeepLearning-UdacityCapston/data/train_sample/'


# In[4]:

IMG_SIZE = 100 # Side for each transformed Image
IMG_DEPTH = 3 # RGB files


# In[5]:

''' Code from Hang_Yao at https://discussions.udacity.com/t/how-to-deal-with-mat-files/160657/5'''
import h5py

# The DigitStructFile is just a wrapper around the h5py data.  It basically references 
#    inf:              The input h5 matlab file
#    digitStructName   The h5 ref to all the file names
#    digitStructBbox   The h5 ref to all struc data
class DigitStructFile:
    def __init__(self, inf):
        self.inf = h5py.File(inf, 'r')
        self.digitStructName = self.inf['digitStruct']['name']
        self.digitStructBbox = self.inf['digitStruct']['bbox']
        self.fileMap = {} # Map from File name to BBox Info

# getName returns the 'name' string for for the n(th) digitStruct. 
    def getName(self,n):
        return ''.join([chr(c[0]) for c in self.inf[self.digitStructName[n][0]].value])

# bboxHelper handles the coding difference when there is exactly one bbox or an array of bbox. 
    def bboxHelper(self,attr):
        if (len(attr) > 1):
            attr = [self.inf[attr.value[j].item()].value[0][0] for j in range(len(attr))]
        else:
            attr = [attr.value[0][0]]
        return attr

# getBbox returns a dict of data for the n(th) bbox. 
    def getBbox(self,n):
        bbox = {}
        bb = self.digitStructBbox[n].item()
        bbox['height'] = self.bboxHelper(self.inf[bb]["height"])
        bbox['label'] = self.bboxHelper(self.inf[bb]["label"])
        bbox['left'] = self.bboxHelper(self.inf[bb]["left"])
        bbox['top'] = self.bboxHelper(self.inf[bb]["top"])
        bbox['width'] = self.bboxHelper(self.inf[bb]["width"])
        return bbox
    
    def getDigitStructure(self,n):
        s = self.getBbox(n)
        s['name']=self.getName(n)
        return s
# getAllDigitStructure returns all the digitStruct from the input file.     
    def getAllDigitStructure(self):
        return [self.getDigitStructure(i) for i in range(len(self.digitStructName))]

# Return a restructured version of the dataset (one structure by boxed digit).
#
#   Return a list of such dicts :
#      'filename' : filename of the samples
#      'bbox' : list of such dicts (one by digit) :
#          'label' : 1 to 9 corresponding digits. 10 for digit '0' in image.
#          'left', 'top' : position of bounding box
#          'width', 'height' : dimension of bounding box
#
# Note: We may turn this to a generator, if memory issues arise.
    def getAllDigitStructure_ByDigit(self):
        pictDat = self.getAllDigitStructure()
        for i in range(len(pictDat)):
            figures = []
            for j in range(len(pictDat[i]['height'])):
               figure = {}
               figure['height'] = pictDat[i]['height'][j]
               figure['label']  = pictDat[i]['label'][j]
               figure['left']   = pictDat[i]['left'][j]
               figure['top']    = pictDat[i]['top'][j]
               figure['width']  = pictDat[i]['width'][j]
               if (figure['label'] == 10): # 0 in figure is represented as 10 in dataset files
                  figure['label'] = 0
               figures.append(figure)
            self.fileMap[pictDat[i]["name"]] = figures
        return self.fileMap


# In[6]:

matFile = os.path.join(DATA_FOLDER, 'digitStruct.mat')
dsf = DigitStructFile(matFile)
train_data = dsf.getAllDigitStructure_ByDigit()


# In[8]:

def getNumPngFiles(dirName):
    return len(glob.glob(dirName + '/*.png'))
def getNextImage(dirName):
    allFileNames = glob.glob(dirName + '/*.png')
    random.shuffle(allFileNames)
    for imgFile in allFileNames:
        img = Image.open(imgFile)
        labels = [int(x['label']) for x in train_data[os.path.split(imgFile)[1]]]
        img = img.resize((IMG_SIZE,IMG_SIZE), resample = (Image.BILINEAR))
        #my_img = tf.image.decode_png(imgFile)
        yield (np.asarray(img),labels)


# In[ ]:



