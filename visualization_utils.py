import sys
import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pylab
import numpy as np

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (6, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
#-----------------------------------------------------------
def show_n_images(imgs, titles = None, enlarge = 20, cmap='jet'):
    
    plt.set_cmap(cmap)
    n = len(imgs)
    gs1 = gridspec.GridSpec(1, n)   
    
    fig1 = plt.figure(); # create a figure with the default size 
    fig1.set_size_inches(enlarge, 2*enlarge);
            
    for i in range(n):

        ax1 = fig1.add_subplot(gs1[i]) 

        ax1.imshow(imgs[i], interpolation='none');
        if (titles is not None):
            ax1.set_title(titles[i])
        ax1.set_ylim(ax1.get_ylim()[::-1])

    plt.show();
#--------------------------------------------------------------
from skimage import io, color, img_as_float
from skimage.exposure import adjust_gamma
# Creates an image of original brain with segmentation overlay
def show_lable_on_image(test_img, test_lbl):

        modes = {'flair':0, 't1':1, 't1c':2, 't2':3}

        label_im = test_lbl
        
        ones = np.argwhere(label_im == 1)
        twos = np.argwhere(label_im == 2)
        threes = np.argwhere(label_im == 3)
        fours = np.argwhere(label_im == 4)

        gray_img = img_as_float(test_img/test_img.max())

        # adjust gamma of image
        image = adjust_gamma(color.gray2rgb(gray_img), 0.45)
        #sliced_image = image.copy()

        red_multiplier = [1, 0.2, 0.2]
        green_multiplier = [0.35,0.75,0.25]
        blue_multiplier = [0,0.5,1.]#[0,0.25,0.9]
        yellow_multiplier = [1,1,0.25]
        brown_miltiplier = [40./255, 26./255, 13./255]

        # change colors of segmented classes
        for i in range(len(ones)):
            image[ones[i][0]][ones[i][1]] = blue_multiplier#red_multiplier
        for i in range(len(twos)):
            image[twos[i][0]][twos[i][1]] = yellow_multiplier 
        for i in range(len(threes)):
            image[threes[i][0]][threes[i][1]] = brown_miltiplier#blue_multiplier
        for i in range(len(fours)):
            image[fours[i][0]][fours[i][1]] = green_multiplier#yellow_multiplier

        return image
#-------------------------------------------------------------------------------------
def show_lable_on_image4(test_img, label_im):
        
    alpha = 0.8

    img = img_as_float(test_img/test_img.max())
    rows, cols = img.shape

    # Construct a colour image to superimpose
    color_mask = np.zeros((rows, cols, 3))
    red_multiplier = [1, 0.2, 0.2]
    green_multiplier = [0.35,0.75,0.25]
    blue_multiplier = [0,0.25,0.9]
    yellow_multiplier = [1,1,0.25]
    brown_miltiplier = [40./255, 26./255, 13./255]
    
        
    color_mask[label_im==1] = blue_multiplier#[1, 0, 0]  # Red block
    color_mask[label_im==2] = yellow_multiplier#[0, 1, 0] # Green block
    color_mask[label_im==3] = brown_miltiplier#[0, 0, 1] # Blue block
    color_mask[label_im==4] = green_multiplier#[0, 1, 1] # Blue block

    # Construct RGB version of grey-level image
    img_color = np.dstack((img, img, img))

    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)

    return img_masked
#------------------------------------------------------------------------------
# DL visualizations

def drow_history(history):
    
    # list all data in history
    if (not type(history)==dict):
        history=history.history
    print(history.keys())
    plt.figure(figsize=(7,4))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_history.pdf')
    plt.show()


def drow_loss(history):
    
    # list all data in history
    if (not type(history)==dict):
        history=history.history

    print(history.keys())
    plt.figure(figsize=(6,5))
    plt.plot(history['lr'], history['loss'])
    plt.plot(history['lr'], history['val_loss'])
    plt.title('model loss by learning rate')
    plt.ylabel('loss')
    plt.xlabel('learning rate')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def append_history(history1, history2):
    
    if (type(history1)==dict):
        if (not type(history2)==dict):
            history2 = history2.history
            for key in history1:
                history1[key] = history1[key] + history2[key]
            return history1
    
    if (not type(history1)==dict):
        if (type(history2)==dict):
            
            for key in history1.history:
                history1.history[key] = history1.history[key] + history2[key]
            return history1
        
    for key in history1.history:
        history1.history[key] = history1.history[key] + history2.history[key]
    return history1


def drow_accuracy(history):
    # list all data in history
    if (not type(history) == dict):
        history = history.history
    print(history.keys())
    plt.figure(figsize=(7, 4))
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_accuracy.pdf')
    plt.show()
