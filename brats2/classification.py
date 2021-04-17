import tensorflow as tf
import nibabel as nib
import os
import utils
import numpy as np
from PIL import Image
import keras
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
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

y = np.zeros((285, 1))
for i in range(210):
    y[i, 0] = 1

y = to_categorical(y, num_classes=2, dtype="uint8")
# print(y)

X_train, X_test, y_train, y_test = train_test_split(bottleneck, y, test_size=0.2)

optimizer = keras.optimizers.Adam(lr=0.00001)

classification_model = Sequential()
classification_model.add(Dense(1000, input_dim=20736, activation='relu'))
classification_model.add(Dense(2, activation='softmax'))
classification_model.summary()

classification_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = classification_model.fit(X_train, y_train, epochs=250, batch_size=32, validation_data=(X_test, y_test))

print(history.history.keys())

# Plot Loss
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()
plt.show()

# Plot Accuracy
plt.figure()
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()

