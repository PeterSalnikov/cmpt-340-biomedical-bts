#import sys
#import os
#entries = os.listdir(".")

#def set_names():
#    path_names = []
#    for e in entries:
#        if e == 'reducedHGG' or e == 'reducedLGG':
#            temp = os.path.abspath("." + "/" + e)
#            path_names.append(temp)
#    return path_names


#e = set_names()
#print(e)
#lis = set_names()

#tem = os.listdir(lis[0])
#print(tem)
import os

#import os 
def set_names():
    path_names = []
    VALIDATION_DATA = 'reducedVal/'
    REDUCED_VALIDATION_DATA = 'reducedVal/'
    DATA_HGG = 'reducedHGG/'
    DATA_LGG = 'reducedLGG/'
    REDUCED_DATA_HGG = 'reducedHGG/'
    REDUCED_DATA_LGG = 'reducedLGG/'
    REDUCED_NUMPY_DIR = 'reducedTrain/numpy_images/'
    REDUCED_VALIDATION_NUMPY_DIR = 'reducedVal/numpy_images/'
    TEST_PRED_NUMPY_DIR = 'predictions/pred1/test/numpy_images/'
    VALIDATION_PRED_NUMPY_DIR = 'predictions/pred1/validation/numpy_images/'
    VALIDATION_PRED_NII_DIR = 'predictions/pred1/validation/nii/'
    TRAIN_ROOT = 'reducedTrain/'

    path_names.append(VALIDATION_DATA) #0
    path_names.append(REDUCED_VALIDATION_DATA) #1
    path_names.append(DATA_HGG) #2
    path_names.append(DATA_LGG) #3
    path_names.append(REDUCED_DATA_HGG) #4
    path_names.append(REDUCED_DATA_LGG) #5
    path_names.append(REDUCED_NUMPY_DIR) #6
    path_names.append(REDUCED_VALIDATION_NUMPY_DIR) #7
    path_names.append(TEST_PRED_NUMPY_DIR) #8
    path_names.append(VALIDATION_PRED_NUMPY_DIR) #9
    path_names.append(VALIDATION_PRED_NII_DIR) #10
    path_names.append(TRAIN_ROOT) #11
    return path_names


# = set_names()
#print(lis)