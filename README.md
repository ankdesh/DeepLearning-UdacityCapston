# DeepLearning-UdacityCapston

### Project Description 
This project aims at developing the skills for solving a Computer vision problem using Deep Learning approach. One of the problems in computer vision is to identify the digits and numbers from the images. This problem have a lot of practical applications from automated assistance for driving(reading speed limits etc.), vehicle localization based on surrounding images, extracting
telephone number from visiting cards etc. The particular problem being worked on in this project is to identify the house numbers from the
images taken from a vehicle on road. The dataset we are using is called “The Street View House Numbers (SVHN) Dataset” available at http://ufldl.stanford.edu/housenumbers/. This dataset has been created from images captured for Google street view.

The detailed problem statement is available at https://docs.google.com/document/d/1L11EjK0uObqjaBhHNcVPxeyIripGHSUaoEWGypuuVtk/pub

### Directory Structure
Directory structure along with important files for the code looks is  
```
|-DataExploration (Code for data visualization)
  |- DatasetImagesStats.ipynb -> Generate image file statistics
  |- ExploreDataSet.ipynb -> Dataset visualization
|-SVHN-CNN ( Main Code directory)
  |- LoadDataset.py -> Utilities to read dataset images and target values
  |- GenDataSetHDF5.ipynb -> Creates intermediate HDF5 files for dataset after simple preprocessing 
  |- FixedWidth-VGG.ipynb -> VGG like model for 2 stage pipeline
  |- EndToEndClassification-SVHN.ipynb -> VGG like model using End to End learning
  |- EndToEndClassification-SVHN-Hyperparam.ipynb -> Code for hyperparameter search for fine tuning
|-SingleDigitExplore ( Code for working through on simpler dataset with single digit images )
  |- MNISTVisualize.ipynb -> Visualize MNIST dataset
  |- SingleDigitDatasetClassifier.ipynb -> Classifier for single image dataset from SVHN
|-Synthetic/FixedWidthMNIST ( Fixed width sythetic image dataset generated using concatenated MNIST digits)
  |- CNNSyntheticMNISTFixedWidth.ipynb -> CNN to train and predict on synthetic dataset 
  |- GenSyntheticMNSITFixedWidth.ipynb -> Generation of sythetic dataset
|-TFRecord (Abandoned code for creating TFRecord files - Now using HDF5 files instead)
```

