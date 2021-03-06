
# coding: utf-8

# In[1]:

import os
import random
import pylab
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
from tflearn.data_utils import pad_sequences 

DATA_FOLDER = '/home/ankdesh/explore/DeepLearning-UdacityCapston/data/extra'

IMG_WIDTH = 32 # Side for each transformed Image
IMG_HEIGHT = 32
IMG_DEPTH = 3 # RGB files

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
               figure['label']  = int(pictDat[i]['label'][j])
               figure['height'] = int(pictDat[i]['height'][j])
               figure['left']   = int(pictDat[i]['left'][j])
               figure['top']    = int(pictDat[i]['top'][j])
               figure['width']  = int(pictDat[i]['width'][j])
               if (figure['label'] == 10): # 0 in figure is represented as 10 in dataset files
                  figure['label'] = 0
               figures.append(figure)
            self.fileMap[pictDat[i]["name"]] = figures
        return self.fileMap

matFile = os.path.join(DATA_FOLDER, 'digitStruct.mat')
dsf = DigitStructFile(matFile)
train_data = dsf.getAllDigitStructure_ByDigit()
allFileNames = glob.glob(DATA_FOLDER + '/*.png')
random.shuffle(allFileNames)

def getNumPngFiles():
    return len(glob.glob(DATA_FOLDER + '/*.png'))
def getNextImage():
    # Images
    for imgFile in allFileNames:
        img = Image.open(imgFile)
        labels = [int(x['label']) for x in train_data[os.path.split(imgFile)[1]]]
        bboxTop = min([int(x['top']) for x in train_data[os.path.split(imgFile)[1]]])
        bboxLeft = min([int(x['left']) for x in train_data[os.path.split(imgFile)[1]]])
        bboxBottom = max([int(x['top'] + x['height']) for x in train_data[os.path.split(imgFile)[1]]])
        bboxRight = max([int(x['left'] + x['width']) for x in train_data[os.path.split(imgFile)[1]]])
        img = img.crop ((bboxLeft, bboxTop, bboxRight, bboxBottom))
        img = img.resize((IMG_WIDTH,IMG_HEIGHT), resample = (Image.BILINEAR)).convert('L') #, resample = (Image.BILINEAR))
        #my_img = tf.image.decode_png(imgFile)
        yield (np.asarray(img),labels)

# Returns tuple of images and a list of 
def getDataSet(numDataPoints, maxDigits):   
    images = np.empty(shape=(numDataPoints, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))
    digits = np.empty(shape=(numDataPoints, maxDigits))

    genImage = getNextImage()
    
    for idx in range(numDataPoints):
        sample_point = genImage.next()
        images[idx] = sample_point[0] 
        # Use value 10.0 for blank
        digits[idx] = pad_sequences([sample_point[1]], maxlen = maxDigits ,value = 10.0)  

    return (images, digits)

# Returns tuple of nums of datasetpoint, images and labels with given num of digits   
def getFixedNumDigistsDataSet(numDataPoints, numDigits):
    images = np.empty(shape=(numDataPoints, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))
    digits = np.empty(shape=(numDataPoints, numDigits),dtype = np.int8)

    genImage = getNextImage()
    
    idx = 0
    while idx < numDataPoints:
        sample_point = next(genImage, None)
        if sample_point == None:
            break
        if len(sample_point[1]) == numDigits:
            images[idx] = sample_point[0]
            digits[idx] = sample_point[1]
            idx += 1
         
    #open('/tmp/asd2.log','w').write(str(list(digits[0:idx])))
    images.resize((idx,images.shape[1],images.shape[2], IMG_DEPTH))
    digits.resize((idx,digits.shape[1]))
    return (idx, images, digits)

def createHDF5FixedNumDigits(numDigits):
    #images = np.empty(shape=(numDataPoints, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))
    #digits = np.empty(shape=(numDataPoints, numDigits),dtype = np.int8)

    genImage = getNextImage()

    images = []
    digits = []
    for sample_point in genImage:
        if len(sample_point[1]) == numDigits:
            images.append(sample_point[0])
            digits.append(sample_point[1])

    import h5py
    fileName = 'svhn_' + str(numDigits) + '.h5'
    h5f = h5py.File(fileName, 'w')
    h5f.create_dataset('images', data=np.asarray(images))
    h5f.create_dataset('digits', data=np.asarray(digits))
    h5f.close()

    #open('/tmp/asd2.log','w').write(str(list(digits[0:idx])))
