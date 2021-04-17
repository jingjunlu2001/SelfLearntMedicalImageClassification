from sklearn.utils import shuffle
import cv2
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
import keras
import nibabel as nib
from PIL import Image
from keras import metrics
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Maximum
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose, UpSampling2D
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D, MaxPooling3D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import os
import utils


# data preprocessing starts here
path = 'BRATS2017/Brats17TrainingData/HGG'
all_images = os.listdir(path)
# print(len(all_images))
all_images.sort()
data = np.zeros((240, 240, 155, 4))
x_to = []
y_to = []

for i in range(len(all_images)):
    print(i)

    x = all_images[i]
    print(x)
    folder_path = path + '/' + x;
    modalities = os.listdir(folder_path)
    modalities.sort()
    # data = []
    w = 0
    for j in range(len(modalities) - 1):
        # print(modalities[j])

        image_path = folder_path + '/' + modalities[j]
        if (image_path[-7:-1] + image_path[-1] == 'seg.nii'):
            img = nib.load(image_path);
            image_data2 = img.get_data()
            image_data2 = np.asarray(image_data2)
            print("Entered ground truth")
        else:
            img = nib.load(image_path);
            image_data = img.get_data()
            image_data = np.asarray(image_data)
            image_data = utils.standardize(image_data)
            data[:, :, :, w] = image_data
            print("Entered modality")
            w = w + 1

    print(data.shape)
    print(image_data2.shape)

    for slice_no in range(0, 155):
        a = slice_no
        X = data[:, :, slice_no, :]

        Y = image_data2[:, :, slice_no]
        # imgplot = plt.imshow(X[:,:,2])
        # plt.show(block=False)
        # plt.pause(0.3)
        # plt.close()

        # imgplot = plt.imshow(Y)
        # plt.show(block=False)
        # plt.pause(0.3)
        # plt.close()

        if (X.any() != 0 and Y.any() != 0 and len(np.unique(Y)) == 4):
            # print(slice_no)
            x_to.append(X)
            y_to.append(Y.reshape(240, 240, 1))

            # imgplot = plt.imshow(X[:, :, 0])
            # plt.show(block=False)
            # plt.pause(100)
            # plt.close()
            #
            # imgplot = plt.imshow(Y)
            # plt.show(block=False)
            # plt.pause(3)
            # plt.close()

            for l in range(4):
                img = Image.fromarray(X[:, :, l])
                img2 = img.rotate(45)
                rotated = np.asarray(img2)
                X[:, :, l] = rotated

            img = Image.fromarray(Y)
            img2 = img.rotate(45)
            rotated = np.asarray(img2)
            Y = rotated

            x_to.append(X)
            y_to.append(Y.reshape(240, 240, 1))

            # for l in range(4):
            #   img = Image.fromarray(X[:,:,l])
            #   img2 = img.rotate(45)
            #   rotated = np.asarray(img2)
            #   X[:,:,l] = rotated

            # img = Image.fromarray(Y)
            # img2 = img.rotate(45)
            # rotated = np.asarray(img2)
            # Y = rotated

            # x_to.append(X)
            # y_to.append(Y.reshape(240,240,1))

            # imgplot = plt.imshow(X[:,:,0])
            # plt.show(block=False)
            # plt.pause(3)
            # plt.close()

            # imgplot = plt.imshow(Y)
            # plt.show(block=False)
            # plt.pause(3)
            # plt.close()

    # hello = y_to.flatten()
    # print(hello[hello==3].shape)
    # print("Number of classes",np.unique(hello))
    # class_weights = class_weight.compute_class_weight('balanced',np.unique(hello),hello)
    #
    # class_weights.insert(3,0)
    # print("class_weights",class_weights)
x_to = np.asarray(x_to)
y_to = np.asarray(y_to)
print(x_to.shape)
print(y_to.shape)

y_to[y_to == 4] = 1  # since label 4 was missing in Brats dataset , changing all labels 4 to 3.
# y_to = one_hot_encode(y_to)
y_to[y_to == 2] = 1
y_to[y_to == 1] = 1
y_to[y_to == 0] = 0
print(y_to.shape)
# y_to = y_to.reshape(240,240,1)