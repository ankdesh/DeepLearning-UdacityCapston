Directory Structure
--------------------
Directory structure along with important files for the code looks is

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

Software/Libraries used

- IPython notebook
- Python 2.7
- Tensorflow 0.11
- TFLearn
- Scikit Learn
- Matplotlib
- Numpy
- H5

