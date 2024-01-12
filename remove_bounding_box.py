# code to help demonstrate the results of a trained model without the bounding box.
# remove_bounding_box does this by removing the last 4 entries of df_train.csv, which are co-ordinates of the box
# author: Peter Salnikov; 301362583; psalniko@sfu.ca

import pandas as pd

img_width = 240
img_height = 240
img_depth = 155 # #(slices)

df_train_full_img = pd.read_csv('df_train.csv')
df_test_full_img = pd.read_csv('df_test.csv')

df_train_full_img['rmin'] = 0
df_train_full_img['rmax'] = img_width
df_train_full_img['cmin'] = 0
df_train_full_img['cmax'] = img_height
df_train_full_img['zmin'] = 0
df_train_full_img['zmax'] = img_depth

df_test_full_img['rmin'] = 0
df_test_full_img['rmax'] = img_width
df_test_full_img['cmin'] = 0
df_test_full_img['cmax'] = img_height
df_test_full_img['zmin'] = 0
df_test_full_img['zmax'] = img_depth

df_train_full_img.to_csv('df_train_full_img.csv')
df_test_full_img.to_csv('df_test_full_img.csv')