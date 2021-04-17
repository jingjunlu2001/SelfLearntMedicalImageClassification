import os, pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import layers
from keras.models import Model
from keras.layers import Input
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from models import Unet
# make the necessary conversion
read_hu = lambda x: imread(x).astype(np.float32)-31750
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

img = []
label = []
for index, row in patient_df.iterrows():
    c_img = []
    slice_curr = read_hu(row['kaggle_path'])
    slice_prev = read_hu(row['kaggle_path_prev'])
    slice_next = read_hu(row['kaggle_path_next'])
    c_img.append(slice_prev)
    c_img.append(slice_curr)
    c_img.append(slice_next)
    img.append(c_img)
    label.append(row['Coarse_lesion_type'])

x = np.asarray(img)
x = x.reshape(409, 512, 512, 3)
x = x / 2048
print('Shape of x: ', x.shape)
y = np.asarray(label)
y = y.reshape(409, 1)
y = y - 1
print('Shape of y: ', y.shape)
class_weights = {0: 5.386,
                 1: 0.396,
                 2: 0.518,
                 3: 0.838,
                 4: 0.5,
                 5: 2.236,
                 6: 1.362,
                 7: 1.016}
# y_ints = [y]
# class_weights = class_weight.compute_class_weight('balanced', 8, y_ints)
# class_weights = dict(enumerate(class_weights))

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
tfconfig.allow_soft_placement = True
sess = tf.Session(config=tfconfig)
sess.run(tf.compat.v1.global_variables_initializer())
keras.backend.set_session(sess)

input_img = Input((512, 512, 3))
model = Unet(input_img, 16, 0.1, True)
learning_rate = 0.001
epochs = 500
decay_rate = learning_rate / epochs
model.compile(optimizer=Adam(lr=learning_rate, decay=decay_rate), loss='mse')
model.summary()

history = model.fit(x=x, y=x, batch_size=32, epochs=20)

maxpool_model = keras.Sequential()
maxpool_model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))

flair_encoder = Model(model.input, model.get_layer('activation_10').output)
flair_encoder.summary()

bottleneck = flair_encoder.predict(x)
bottleneck = maxpool_model.predict(bottleneck)
print('Shape of bottleneck: ', bottleneck.shape)
bottleneck = bottleneck.reshape((409, 65536))

y = to_categorical(y, num_classes=8, dtype="uint8")

x_train, x_val, y_train, y_val = train_test_split(bottleneck, y, test_size=0.2)

optimizer = keras.optimizers.Adam(lr=0.00001)

classification_model = Sequential()
classification_model.add(Dense(10000, input_dim=65536, activation='relu'))
classification_model.add(Dropout(0.3))
classification_model.add(Dense(1000, activation='relu'))
classification_model.add(Dropout(0.3))
classification_model.add(Dense(100, activation='relu'))
classification_model.add(Dropout(0.3))
classification_model.add(Dense(8, activation='softmax'))
classification_model.summary()

classification_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = classification_model.fit(x_train, y_train, epochs=150, batch_size=32, class_weight=class_weights, validation_data=(x_val, y_val))

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
