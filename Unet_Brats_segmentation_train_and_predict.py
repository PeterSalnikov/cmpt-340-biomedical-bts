#!/usr/bin/env python
# coding: utf-8

# # Unet for multi-class segmentation
# Train and predict

# ## Chalange description

# The data is 3D MRI images with 4 channels:
# * Flair
# * T1
# * T1ce
# * T2
# 
# There are 259 photos in train data, each has segmentation labeling file, with the following lables:
# 
# * Label 0: background
# * Label 1: necrotic and non-enhancing tumor
# * Label 2: edema 
# * Label 4: enhancing tumor
# 
# In the data preprocessing stage, I converted and merged the nii.gz files, to anumpy file of the format:
# > (155, 240, 240, 4)
# > For simplicity, I change label 4 to 3. So we need to change it back, for submitting results to Brats challange.

# In[1]:


# import model_unet
label_type_shrt = ['background', 'necrotic',
             'edema', 'enhancing']
label_type = ['background', 'necrotic and non-enhancing tumor', 'edema', 'enhancing tumor']


# In[2]:

import paths

path_names = paths.set_names()

DATA= path_names[11]
VALIDATION_DATA = path_names[1]
DATA_HGG = DATA + path_names[4]
DATA_LGG = DATA + path_names[5]

NUMPY_DIR = path_names[6]
VALIDATION_NUMPY_DIR = path_names[7]
FLAIR = 'flair'
T1 = 't1'
T2 = 't2'
T1CE = 't1ce'


# In[3]:


img_type=['FLAIR', 'T1','T1CE', 'T2']


# In[4]:



import os, sys, glob
import numpy as np
import SimpleITK as sitk
import sys
import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.utils import shuffle
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


import pandas
import numpy


# In[6]:


from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank
from scipy.ndimage import gaussian_filter
from skimage import data
from skimage import img_as_float
from skimage.morphology import reconstruction
from scipy import ndimage
import random
from random import randrange
from keras.utils import np_utils


# ## Visualization utilities

from importlib import reload  # Python 3.4+ only.
import visualization_utils as vu
from visualization_utils import show_lable_on_image4
reload(vu)



red_multiplier = [1, 0.2, 0.2]
green_multiplier = [0.35,0.75,0.25]
blue_multiplier = [0,0.5,1.] #[0,0.25,0.9]
yellow_multiplier = [1,1,0.25]
brown_miltiplier = [40./255, 26./255, 13./255]
my_colors=[blue_multiplier, yellow_multiplier, brown_miltiplier]




def show_img_lable(img, lbl, modality = 0):
    
    if (len(lbl.shape)> 2):
        lbl[0,0,3]=1 # for uniqe colors in plot
        lbl = lbl_from_cat(lbl)
    vu.show_n_images([img[:,:,modality],lbl, show_lable_on_image4(img[:,:,modality],lbl)],
                    titles = [img_type[modality], 'Label', 'Label on '+ img_type[modality]]);
    



def show_lable(lbl):
    
    
    vu.show_n_images([lbl[:,:,k] for k in range(4)]+[lbl_from_cat(lbl)],
                 titles = label_type_shrt + ['Label']) 



def show_pred_im_lable(im, lb, pred):
    
    vu.show_n_images([im[:,:,1], lb[:,:], 
                   show_lable_on_image4(im[:,:,1], lb[:,:]),
                  show_lable_on_image4(im[:,:,1], pred[:,:])],
                 titles=['Flair', 'Label', 'Label on T1', 'Prediction on Flair'])



def show_pred_im(im, pred):
    
    vu.show_n_images([im[:,:,1], 
                   im[:,:,0],pred,
                  show_lable_on_image4(im[:,:,1], pred[:,:])],
                 titles=['Flair','T1', 'Pred',  'Prediction on Flair'])


# ## Read image description files
# In the preprocessing notebook, all files where saved as numpy.
# 
# Some statistics on the labels were collected, to assist the training phase.


df_train= pd.read_csv('df_train.csv')
df_train_full_img = pd.read_csv('df_train_full_img.csv') #for running w/o bounding box
df_test= pd.read_csv('df_test.csv')
df_test_full_img = pd.read_csv('df_test_full_img.csv') #for running w/o bounding box
df_val= pd.read_csv('df_val.csv')



df_train[['lab' +str(i) for i in range(1,4)]].sum().plot(kind='bar', color=my_colors,
                                                   title='Lable distribution in Train data')



df_test[['lab' +str(i) for i in range(1,4)]].sum().plot(kind='bar', color=my_colors,
                                                   title='Lable distribution in Test data')


# ## Build Train generator

def get_numpy_img_lbl(img_id = 'BraTS19_TCIA10_632_1', np_dir=NUMPY_DIR):
    img=np.load(os.path.join(np_dir, img_id+'.npy'))
    lbl=np.load(os.path.join(np_dir, img_id+'_lbl.npy'))
    return img,lbl


def get_random_img(axis=0, df=df_train, np_dir=NUMPY_DIR):
    
    ind = randrange(len(df))
    img_id= df.iloc[ind].id
    img,lbl = get_numpy_img_lbl(img_id, np_dir=NUMPY_DIR)
        
    if (axis==0):
        x = randrange(df.iloc[ind].rmin, df.iloc[ind].rmax+1)
        return img[ x,:,:, :], lbl[x,:,:]

    im = np.zeros((240,240,4),dtype=np.float32)    
    lb = np.zeros((240,240),dtype=np.int)
        
    if (axis==1):
        y = randrange(df.iloc[ind].cmin, df.iloc[ind].cmax+1)
        im[40:40+155,:,:]=img[:, y,:, :]
        lb[40:40+155,:]=lbl[:, y,:]
        return im,lb
    
    if (axis == 2):
        z = randrange(df.iloc[ind].zmin, df.iloc[ind].zmax+1)
        im[40:40+155,:,:]=img[:,:, z, :]
        lb[40:40+155,:]=lbl[:,:,z]
        return im,lb
    return None    


# Function randomly selects a 2D image that includes the given label

def get_img_for_label(lab=2, axis=0, df=df_train,np_dir = NUMPY_DIR):
    
    img_id= random.choice(df[df['lab'+str(lab)] > 0].id.values)
    
    img,lbl = get_numpy_img_lbl(img_id, np_dir)
    ind = np.where(lbl==lab)
    k = random.randrange(len(ind[0]))
    
    if (axis==0):        
        return img[ind[0][k],:,:] , lbl[ind[0][k],:,:]
        
    lb = np.zeros((240,240),dtype=np.int)
    im = np.zeros((240,240,4),dtype=np.float32)
    
    if (axis==1):
        im[40:40+155,:,:]=img[:, ind[1][k],:,:]
        lb[40:40+155,:]=lbl[:, ind[1][k],:]
        return im,lb
    
    if (axis == 2):
        im[40:40+155,:,:]=img[:, :, ind[2][k],:]
        lb[40:40+155,:]=lbl[:,:,ind[2][k]]
        return im,lb
    return None


def lbl_from_cat(cat_lbl):
    
    lbl=0
    if (len(cat_lbl.shape)==3):
        for i in range(1,4):
            lbl = lbl + cat_lbl[:,:,i]*i
    elif (len(cat_lbl.shape)==4):
        for i in range(1,4):
            lbl = lbl + cat_lbl[:,:,:,i]*i
    else:
        print('Error in lbl_from_cat', cat_lbl.shape)
        return None
    return lbl


# For test we will create batch from few test images. only planes with lables >0 will be included.

def normalize_3D_image(img):
    for z in range(img.shape[0]):
        for k in range(4):
            if (img[z,:,:,k].max()>0):
                img[z,:,:,k] /= img[z,:,:,k].max()
    return img


def normalize_2D_image(img):

        for c in range(4):
            if (img[:,:,c].max()>0):
                img[:,:,c] = img[:,:,c]/img[:,:,c].max()
        return img


# Function returns all z-planes of the image that have non-zerolable

def get_img_batch(row, np_dir=NUMPY_DIR):
    
    im,lb = get_numpy_img_lbl(row['id'], np_dir)
    
    n_im = row['rmax']-row['rmin']
    rmin=row['rmin']
    rmax=row['rmax']
    
    return normalize_3D_image(im[rmin:rmax]), np_utils.to_categorical(lb[rmin:rmax],4)


def get_df_img_batch(df_batch, np_dir=NUMPY_DIR):
    
        n_images = (df_batch.rmax - df_batch.rmin).sum()
        b_images = np.zeros((n_images, 240, 240, 4), np.float32)
        b_label = np.zeros((n_images, 240, 240, 4), np.int8)    
        ind=0
        for index, row in df_batch.iterrows():
 
            b_im, b_lb = get_img_batch(row, np_dir)
            n_im = b_im.shape[0]
            b_images[ind:ind+n_im] = b_im
            b_label[ind:ind+n_im] = b_lb
            ind+=n_im
               
        return b_images, b_label


from keras.utils import np_utils
def generate_im_test_batch(n_images = 3, batch_size=60, df = df_test, np_dir=NUMPY_DIR):

    while 1:
         
        df_batch = df.sample(n_images)
        b_images, b_label = get_df_img_batch(df_batch, np_dir)                    
        b_images, b_label = shuffle(b_images, b_label)
        if (batch_size > 0):
            b_images = b_images[0:batch_size]
            b_label = b_label[0:batch_size]
            
        yield b_images, b_label


get_ipython().run_cell_magic('time', '', 'gen_test_im = generate_im_test_batch(1)\nimtest,lbtest = next(gen_test_im)\nimtest.shape, lbtest.shape')


#***
from keras.utils import np_utils
def generate_faste_train_batch(batch_size = 12, df = df_train ,np_dir=NUMPY_DIR):
    # *** COMMENT ABOVE LINE AND UNCOMMENT BELOW LINE TO RUN WITH NO BOUNDING BOX, AND VICE-VERSA ***
# def generate_faste_train_batch(batch_size = 12, df = df_train_full_img ,np_dir=NUMPY_DIR):

    
    batch_images = np.zeros((batch_size, 240, 240, 4), np.float32)
    batch_label = np.zeros((batch_size, 240, 240, 4), np.int8)    
    
    # lab1 22%
    # lab2 58%
    # lab3 18%

    while 1:
        
        df_batch = df.sample(3)
        b_images, b_label = get_df_img_batch(df_batch, np_dir)                    
        b_images, b_label = shuffle(b_images, b_label)
        batch_images[0:batch_size//2]=b_images[0:batch_size//2]
        batch_label[0:batch_size//2]=b_label[0:batch_size//2]
        
        i=batch_size//2
        # lab 1
        nim = batch_size//4
        for j in range(nim):
            im,lbl = get_img_for_label(lab=1, axis=random.choice([0,1,2]), df=df)
            batch_images[i] = normalize_2D_image(im)
            batch_label[i] = np_utils.to_categorical(lbl, 4)
            i+=1
                        
        # lab 3
        nim = batch_size//4
        for j in range(nim):
            im,lbl = get_img_for_label(lab=3, axis=random.choice([0,1,2]), df=df)
            batch_images[i] = normalize_2D_image(im)
            batch_label[i] = np_utils.to_categorical(lbl, 4)
            i+=1

        batch_images, batch_label = shuffle(batch_images, batch_label)
            
        yield batch_images, batch_label



get_ipython().run_cell_magic('time', '', 'gen_train_fast = generate_faste_train_batch(batch_size=16)\nbimg,blbl = next(gen_train_fast)\nbimg.shape, blbl.shape')



### define Base Unet Model

import tensorflow as tf
# tf.disable_v2_behavior()
from keras import backend as K

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers import Input, UpSampling2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from tensorflow.python.keras.backend import set_session

config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
config.log_device_placement = True # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
# TF_GPU_ALLOCATOR=cuda_malloc_async

IMG_HEIGHT = 240
IMG_WIDTH = 240
IMG_CHANNELS = 4



from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add

# Build U-Net model
dropout=0.2
hn = 'he_normal'
def unet(input_size = (IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)):
    
    inputs = Input(input_size)
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(conv4)
    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(conv5)
    drop5 = Dropout(dropout)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = hn)(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = hn)(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = hn)(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = hn)(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = hn)(conv9)
    #conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    conv10 = Conv2D(4, (1,1), activation = 'softmax')(conv9)
    
    model = Model(inputs = inputs, outputs = conv10)

    return model 



from keras.models import Model, load_model

model = unet(input_size = (IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS))

"""
OUR CUSTOM LOSS FUNCTION
"""

import tensorflow.keras.backend as K
from tensorflow.keras.losses import categorical_crossentropy
def generalized_dice(y_true, y_pred):

    """
    Generalized Dice Score
    https://arxiv.org/pdf/1707.03237

    """
    y_true = tf.cast(y_true, tf.float32)
    y_true    = K.reshape(y_true,shape=(-1,4))
    y_pred    = K.reshape(y_pred,shape=(-1,4))
    sum_p     = K.sum(y_pred, -2)
    sum_r     = K.sum(y_true, -2)
    sum_pr    = K.sum(y_true * y_pred, -2)
    weights   = K.pow(K.square(sum_r) + K.epsilon(), -1)
    generalized_dice = (2 * K.sum(weights * sum_pr)) / (K.sum(weights * (sum_r + sum_p)))

    return generalized_dice

def generalized_dice_loss(y_true, y_pred):
    return 1-generalized_dice(y_true, y_pred)



# SET LOSS TO 'categorical_crossentropy' for default and generalized_dice_loss for a different result
model.compile(optimizer = Adam(lr = 0.0001), loss = generalized_dice_loss, 
              metrics = ['accuracy'])



get_ipython().run_cell_magic('time', '', "from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau\n\nearlystopper = EarlyStopping(patience=8, verbose=1)\ncheckpointer = ModelCheckpoint(filepath = 'model_unet_4ch.hdf5',\n                               verbose=1,\n                               save_best_only=True, save_weights_only = True)\n\nreduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,\n                              patience=5, min_lr=0.000001, verbose=1,  cooldown=1)")

from keras.preprocessing.image import ImageDataGenerator
b_images, b_label = get_df_img_batch(df_train, NUMPY_DIR)

v_images, v_label = get_df_img_batch(df_test, NUMPY_DIR)


data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip=True,
                     vertical_flip=True,
                     zoom_range=0.2)

image_datagen = ImageDataGenerator(data_gen_args)
validation_datagen= ImageDataGenerator(data_gen_args)
# Provide the same seed and keyword arguments to the fit and flow methods

seed = 1
image_datagen.fit(b_images, augment=True, seed=seed)
validation_datagen.fit(v_images, augment=True, seed=seed)
image_generator = image_datagen.flow(
    b_images,
    y=b_label,
    batch_size=10,
)


validation_datagen = validation_datagen.flow(
    v_images,
    y=v_label,
    batch_size=10,
)


# combine generators into one which yields image and masks
# train_generator = zip(image_generator, mask_generator)

###ORIGINAL MODEL FIT; COMMENT ONE OR THE OTHER TO RUN
get_ipython().run_cell_magic('time', '', 'history = model.fit_generator(gen_train_fast,\n                                        validation_data = gen_test_im, validation_steps=1,\n                                              steps_per_epoch=5,\n                              epochs=30,\n                    callbacks=[earlystopper, checkpointer, reduce_lr])')
###NEW MODEL FIT
# get_ipython().run_cell_magic('time', '', 'history = model.fit(image_generator,\n                                         validation_data = validation_datagen, validation_steps=5,\n                                              steps_per_epoch=5,\n                              epochs=30,\n                    callbacks=[earlystopper, checkpointer, reduce_lr])')



reload(vu)

vu.drow_history(history)

vu.drow_accuracy(history)

with open('model_history.txt','w') as data:  
      data.write(str(history.history))

# In[ ]:


model.save_weights('model_unet_ce.hdf5')


# In[ ]:


model.load_weights('model_unet_4ch.hdf5')


# ## Predict with trained model

# Function converts probabilities to labels

# In[ ]:


def get_pred(img, threshold=0.5):
    out_img=img.copy()
    out_img=np.where(out_img>threshold, 1,0)
    return out_img


# In[ ]:


def prediction_from_probabily_3D(img):
    
    int_image = get_pred(img)
    return lbl_from_cat(int_image)


# In[ ]:


def get_prediction_for_batch(pred_batch, threshold=0.5):
    
    out_batch = np.zeros((pred_batch.shape[0], 240, 240),dtype=np.int)
    
    for j in range(pred_batch.shape[0]):
        pred = get_prediction(pred_batch[j])
        if (pred.sum()>0):
            print(j, np.unique(pred , return_counts=True))
        out_batch[j] = lbl_from_cat(get_prediction(pred_batch[j]))
    return out_batch  


# In[ ]:


def get_label_from_pred_batch(labels_batch):
    
    batch = np.zeros((labels_batch.shape[0], 240, 240), np.uint8)
     
    for j in range(labels_batch.shape[0]):
        batch[j]=get_pred(labels_batch[j,:,:,0])+                get_pred(labels_batch[j,:,:,1])*2+        get_pred(labels_batch[j,:,:,2])*4

    return batch


# In[ ]:


def predict_3D_img_prob(np_file):
    
    np_img = np.load(np_file)
    for_pred_img = np.zeros((155, 240, 240, 4), np.float32)

    # Normalize image
    for_pred_img = normalize_3D_image(np_img)

    mdl_pred_img =  model.predict(for_pred_img)

    #pred_label = prediction_from_probabily_3D(mdl_pred_img)

    return mdl_pred_img


# ## Predict on Test images

# ## Predict all test images to calculate IOU


TEST_PRED_NUMPY_DIR = path_names[8]
# TEST_PRED_NUMPY_DIR = 'predictions/pred1/test/numpy_images/'
VALIDATION_PRED_NUMPY_DIR = path_names[9]
# VALIDATION_PRED_NUMPY_DIR = 'predictions/pred1/validation/numpy_images/'
VALIDATION_PRED_NII_DIR = path_names[10]
# VALIDATION_PRED_NII_DIR = 'predictions/pred1/validation/nii/'


# Check that its all working :-)


# for index, row in df_test.iterrows(): 
# *** COMMENT ABOVE LINE AND UNCOMMENT BELOW LINE TO RUN WITH NO BOUNDING BOX, AND VICE-VERSA ***
for index, row in df_test_full_img.iterrows():

    img_id = row['id']
    im,lb = get_numpy_img_lbl(img_id = 'BraTS19_2013_18_1')

    nimg = os.path.join(NUMPY_DIR, img_id+'.npy')
    pred_stats = predict_3D_img_prob(nimg)

    pred = prediction_from_probabily_3D(pred_stats)

    out_img = os.path.join(TEST_PRED_NUMPY_DIR, img_id+'_pred.npy')

    np.save(out_img, pred)


# ## Predict all validation image


for index, row in df_val.iterrows():

    img_id = row['id']

    nimg = os.path.join(VALIDATION_NUMPY_DIR, img_id+'.npy')
    pred_stats = predict_3D_img_prob(nimg)

    pred = prediction_from_probabily_3D(pred_stats)

    out_img = os.path.join(VALIDATION_PRED_NUMPY_DIR, img_id+'_pred.npy')
    np.save(out_img, pred)
    
    pred = np.where(pred==3,4, pred)
    out_nii = os.path.join(VALIDATION_PRED_NII_DIR, img_id+'.nii.gz')

    sitk_img = sitk.GetImageFromArray(pred)
    sitk.WriteImage(sitk_img , out_nii)


