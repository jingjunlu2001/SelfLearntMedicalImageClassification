import os, pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from skimage.io import imread
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import layers, optimizers
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.applications.vgg16 import VGG16
# make the necessary conversion
read_hu = lambda x: imread(x).astype(np.float32)-31750
read_hu_test = lambda x: imread(x).astype(np.float32)-32768
base_img_dir = 'minideeplesion/'


patient_df = pd.read_csv('DL_info_with_labels.csv')
patient_df['kaggle_path'] = patient_df.apply(lambda c_row: os.path.join(base_img_dir,
'{Patient_index:06d}_{Study_index:02d}_{Series_ID:02d}'.format(**c_row), '{Key_slice_index:03d}.png'.format(**c_row)), 1)
patient_df['kaggle_path_prev'] = patient_df.apply(lambda c_row: os.path.join(base_img_dir,
'{Patient_index:06d}_{Study_index:02d}_{Series_ID:02d}'.format(**c_row), '{Key_slice_index_prev:03d}.png'.format(**c_row)), 1)
patient_df['kaggle_path_next'] = patient_df.apply(lambda c_row: os.path.join(base_img_dir,
'{Patient_index:06d}_{Study_index:02d}_{Series_ID:02d}'.format(**c_row), '{Key_slice_index_next:03d}.png'.format(**c_row)), 1)
print('Loaded', patient_df.shape[0], 'cases')

patient_df['exists'] = patient_df['kaggle_path'].map(os.path.exists)
patient_df = patient_df[patient_df['exists']].drop('exists', 1)
# draw the bounding boxes
patient_df['bbox'] = patient_df['Bounding_boxes'].map(lambda x: np.reshape([float(y) for y in x.split(',')], (-1, 4)))
print('Found', patient_df.shape[0], 'patients with images')


def create_boxes(in_row):
    box_list = []
    for (start_x, start_y, end_x, end_y) in in_row['bbox']:
        box_list += [Rectangle((start_x, start_y),
                         np.abs(end_x-start_x),
                         np.abs(end_y-start_y)
                         )]
    return box_list


_, test_row = next(patient_df.sample(1).iterrows())
fig, ax1 = plt.subplots(1, 1, figsize = (10, 10))
c_img = read_hu_test(test_row['kaggle_path'])
ax1.imshow(c_img, vmin = -1200, vmax = 600, cmap = 'gray')
ax1.add_collection(PatchCollection(create_boxes(test_row), edgecolors=(0,0,1,1), facecolors=(0,1,0,0), linewidths=2))
ax1.set_title('{Patient_age}-{Patient_gender}'.format(**test_row))
plt.show()
