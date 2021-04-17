import tensorflow as tf
import nibabel as nib
import os
import utils
import numpy as np
from PIL import Image
import keras
from keras.models import Model
from keras.models import load_model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


X_train_t1 = np.zeros((285, 240, 240, 155))
X_train_t1ce = np.zeros((285, 240, 240, 155))
X_train_flair = np.zeros((285, 240, 240, 155))
X_train_t2 = np.zeros((285, 240, 240, 155))
t1_small_ = np.zeros((285, 144, 144, 155))
t1ce_small_ = np.zeros((285, 144, 144, 155))
flair_small_ = np.zeros((285, 144, 144, 155))
t2_small_ = np.zeros((285, 144, 144, 155))
t1_small = np.zeros((285, 144, 144, 100))
t1ce_small = np.zeros((285, 144, 144, 100))
flair_small = np.zeros((285, 144, 144, 100))
t2_small = np.zeros((285, 144, 144, 100))

# data preprocessing starts here
path = 'BRATS2017/Brats17TrainingData/HGG'
all_images = os.listdir(path)
# print(len(all_images))
all_images.sort()
data = np.zeros((240, 240, 155, 4))

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
            # image_data = cv2.resize(image_data, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
            image_data = utils.standardize(image_data)
            data[:, :, :, w] = image_data
            print("Entered modality")
            w = w + 1

    print(data.shape)
    print(image_data2.shape)

    X_train_flair[i, :, :, :] = data[:, :, :, 0]
    X_train_t1[i, :, :, :] = data[:, :, :, 1]
    X_train_t1ce[i, :, :, :] = data[:, :, :, 2]
    X_train_t2[i, :, :, :] = data[:, :, :, 3]

    # X = data[:, :, :, 1]
    # print(X[:, :, 60])
    # imgplot = plt.imshow(X[:, :, 60])
    # plt.show(block=False)
    # plt.pause(0.3)
    # plt.close()

path = 'BRATS2017/Brats17TrainingData/LGG'
all_images = os.listdir(path)
# print(len(all_images))
all_images.sort()

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
            # image_data = cv2.resize(image_data, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
            image_data = utils.standardize(image_data)
            data[:, :, :, w] = image_data
            print("Entered modality")
            w = w + 1

    print(data.shape)
    print(image_data2.shape)

    X_train_flair[i+210, :, :, :] = data[:, :, :, 0]
    X_train_t1[i+210, :, :, :] = data[:, :, :, 1]
    X_train_t1ce[i+210, :, :, :] = data[:, :, :, 2]
    X_train_t2[i+210, :, :, :] = data[:, :, :, 3]


X_train_t1 = X_train_t1 / 255
X_train_t1ce = X_train_t1ce / 255
X_train_t2 = X_train_t2 / 255
X_train_flair = X_train_flair / 255

for i in range(285):
    for j in range(144):
        flair_small_[i, j, j, :] = X_train_flair[i, j+54, j+54, :]
for i in range(285):
    for j in range(100):
        flair_small[i, :, :, j] = flair_small_[i, :, :, j+36]

print('Shape of flair_small: ' + str(flair_small.shape))

model = load_model('flair_model')
# model.trainable = False
# model.summary()

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
tfconfig.allow_soft_placement = True
sess = tf.Session(config=tfconfig)
sess.run(tf.global_variables_initializer())
keras.backend.set_session(sess)

history = model.fit(x=flair_small, y=flair_small, batch_size=32, epochs=100)

encoder = Model(model.input, model.get_layer('activation_10').output)
encoder.summary()

bottleneck = encoder.predict(flair_small)
bottleneck = bottleneck.reshape((285, 20736))
print('Shape of bottleneck: ' + str(bottleneck.shape))

# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=3000, random_state=0)
#     kmeans.fit(bottleneck)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

kmeans = KMeans(n_clusters=2, random_state=0, max_iter=3000).fit(bottleneck)
print(kmeans.labels_)
