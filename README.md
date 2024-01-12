# Brain_tumor_segmentation
BraTS has always been focusing on the evaluation of state-of-the-art methods for the segmentation of brain tumors in multimodal magnetic resonance imaging (MRI) scans. BraTS 2019 utilizes multi-institutional pre-operative MRI scans and focuses on the segmentation of intrinsically heterogeneous (in appearance, shape, and histology) brain tumors, namely gliomas.

*** CODE is based on Naomi Fridman's submission to the BraTS '19 Segmentation challenge. We borrowed her U-net and data processing methods, and we made changes where necessary to run on our systems as well as to help observe changes as we adjust parameters. ***

https://medium.com/@naomi.fridman/multi-class-image-segmentation-a5cc671e647a


The source code for our paper "Attention-Guided Version of 2D UNet for Automatic Brain Tumor Segmentation"

Our paper can be found at [this link](https://ieeexplore.ieee.org/document/8964956).

## Overview
- [Dataset/Pre-processing](#Dataset/Pre-processing)
- [Architecture](#Architecture)
- [Training Process](#Training-Process)
- [Results](#Results)
- [Usage](#Usage)

### Dependencies
- [numpy 1.19.5](https://numpy.org/)
- [nibabel 3.0.1](https://nipy.org/nibabel/)
- [scipy 1.3.2](https://www.scipy.org/)
- [Tensorflow 2.4.0](https://www.tensorflow.org/)
- [Keras 2.4.3](https://keras.io/)
- [pandas 1.2.4]
- [matplotlib 3.4.1]
- [(for NVIDIA GPU) Nvidia GPU Computing Toolkit v11.3 w/ CUDNN 11.3]
- [scikit-learn and scikit-image]
- [jupyter notebook and ipython]

### Dataset/Pre-processing
Make sure to set the paths in the paths.py to there corresponding files once you have downloaded the data. our directory looks like the following
When down loading the whole data set. Other wise the numpy_images and reduced files will be empty until preprocessing files are ran.


├── MICCAI_BraTS_2019_Data_Training

│   ├── HGG

│   ├── LGG

│   ├── numpy_images

│   ├── reducedHGG

│   ├── reducedLGG

├── predictions

│   └── pred1

│       ├── test

│       │   └── numpy_images

│       └── validation

│           ├── nii

│           └── numpy_images


└── reducedVal
      
      
      
      
      
      
   │           └── numpy_images

**
1. run "pip install -r requirements.txt"

2. download BraTS '19 Dataset from https://www.med.upenn.edu/cbica/brats2019/data.html, by creating an account and requesting the training and validation data.
    - 

3. In "Preprocessing_N4BiasFieldCorrectionImageFilter.ipynb", set the directories to the location of your data and run the notebook. You should receive a reduced subset of the existing data for easier computing. This script also performs bias field correction on the data before it is turned into .npy images in the next script

4. Next, run the cells of "Brats_segmentation_preprocess_data.ipynb." If the numpy_images folders were created properly they should be populated with .npy images.

5. Finally to run the model, either executing the cells of "Unet_Brats_segmentation_train_and_predict.ipynb" or run "ipython Unet_Brats_segmentation_train_and_predict.py." Choose to run with or without bounding boxes within the .py files as noted by the comments. We have provided the sample data subset that we used to run this model, as preprocessing is time-consuming.


Data to run our repo quickly that is preprocessed can be found at: 
https://1sfu-my.sharepoint.com/:f:/g/personal/nsimms_sfu_ca/EnB_gVulIclEsGagl1AtKloBwgMmGAghaQ6OPdHRNc9VgA?e=VrJkXq

The repoData.zip contain the preprocessed data generated from Preprocessing_N4BiasFieldCorrectionImageFilter and Brats_segmentation_preprocess_data to run just the model the files in
preprocessed Data.zip can be used. It contains the numpy images needed for the Unet_Brats_segmentation_train_and_predict.ipynb. The repoData.zip is about 29GB the 
preprocessed Data.zip is about 19GB and is all that is needed to run the Unet_Brats_segmentation_train_and_predict this is the suggested download.
### Architecture


### Training Process

### Results



### Usage
