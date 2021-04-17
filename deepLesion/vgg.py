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

x_raw = np.asarray(img)
x_raw = x_raw.reshape(409, 512, 512, 3)
x = np.zeros((409, 448, 448, 3))
x = x_raw[:, 32:480, 32:480, :]
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

x_tensor = tf.convert_to_tensor(x)
max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')
x_tensor = max_pool_2d(x_tensor)
x = x_tensor.eval(session=tf.compat.v1.Session())

vggmodel = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
for layers in (vggmodel.layers)[:19]:
    print(layers)
    layers.trainable = False

pool_5 = vggmodel.layers[-1].output
flatten = Flatten(name='flatten')(pool_5)
fc1 = Dense(4096, activation='relu', name='fc1')(flatten)
dropout1 = Dropout(0.3)(fc1)
fc2 = Dense(4096, activation='relu', name='fc2')(dropout1)
dropout2 = Dropout(0.3)(fc2)
predictions = Dense(8, activation='softmax')(dropout2)
model = Model(input=vggmodel.input, output=predictions)

# fc2 = vggmodel.layers[-2].output
# predictions = Dense(8, activation='softmax')(fc2)
# model = Model(input=vggmodel.input, output=predictions)

optimizer = optimizers.Adam(lr=0.00005)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

y = to_categorical(y, num_classes=8, dtype="uint8")
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15)

history = model.fit(x_train, y_train, epochs=35, batch_size=32, class_weight=class_weights, validation_data=(x_val, y_val))

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


# patient_df2 = pd.read_csv('DL_info_without_labels.csv')
# patient_df2['kaggle_path'] = patient_df2.apply(lambda c_row: os.path.join(base_img_dir,
# '{Patient_index:06d}_{Study_index:02d}_{Series_ID:02d}'.format(**c_row), '{Key_slice_index:03d}.png'.format(**c_row)), 1)
# patient_df2['kaggle_path_prev'] = patient_df2.apply(lambda c_row: os.path.join(base_img_dir,
# '{Patient_index:06d}_{Study_index:02d}_{Series_ID:02d}'.format(**c_row), '{Key_slice_index_prev:03d}.png'.format(**c_row)), 1)
# patient_df2['kaggle_path_next'] = patient_df2.apply(lambda c_row: os.path.join(base_img_dir,
# '{Patient_index:06d}_{Study_index:02d}_{Series_ID:02d}'.format(**c_row), '{Key_slice_index_next:03d}.png'.format(**c_row)), 1)
# print('Loaded', patient_df2.shape[0], 'cases')
#
# patient_df2['exists'] = patient_df2['kaggle_path'].map(os.path.exists)
# patient_df2 = patient_df2[patient_df2['exists']].drop('exists', 1)
# # draw the bounding boxes
# patient_df2['bbox'] = patient_df2['Bounding_boxes'].map(lambda x: np.reshape([float(y) for y in x.split(',')], (-1, 4)))
# print('Found', patient_df2.shape[0], 'patients with images')
#
# img2 = []
# for index, row in patient_df2.iterrows():
#     c_img = []
#     slice_curr = read_hu(row['kaggle_path'])
#     slice_prev = read_hu(row['kaggle_path_prev'])
#     slice_next = read_hu(row['kaggle_path_next'])
#     c_img.append(slice_prev)
#     c_img.append(slice_curr)
#     c_img.append(slice_next)
#     img2.append(c_img)
#
# x_test = np.asarray(img2)
# x_test = x_test.reshape(933, 512, 512, 3)
# x = x / 2048
# print('Shape of x_test: ', x_test.shape)
#
# lesion_type_dict = {
#     1: 'bone',
#     2: 'abdomen',
#     3: 'mediastinum',
#     4: 'liver',
#     5: 'lung',
#     6: 'kidney',
#     7: 'soft tissue',
#     8: 'pelvis'
# }
#
# lesion_predictions = model.predict(x_test)
# lesion_predictions = np.asarray(lesion_predictions)
# lesion_predictions = np.argmax(lesion_predictions, axis=1)
# lesion_predictions += 1
# lesion_predictions = lesion_predictions.tolist()
# patient_df2['pred'] = lesion_predictions
# print(patient_df2.head(10))
#
#
# def create_boxes(in_row):
#     box_list = []
#     for (start_x, start_y, end_x, end_y) in in_row['bbox']:
#         box_list += [Rectangle((start_x, start_y),
#                          np.abs(end_x-start_x),
#                          np.abs(end_y-start_y)
#                          )]
#     return box_list
#
#
# for i in range(20):
#     print('Drawing image', i+1, '/20...')
#     _, test_row = next(patient_df2.sample(1).iterrows())
#     fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
#     c_img = read_hu_test(test_row['kaggle_path'])
#     ax1.imshow(c_img, vmin=-1200, vmax=600, cmap='gray')
#     ax1.add_collection(PatchCollection(create_boxes(test_row), edgecolors=(1,0,0,1), facecolors=(0,1,0,0), linewidths=2))
#     ax1.set_title('Predicted lesion type: {}'.format(lesion_type_dict[test_row['pred']]), fontsize=24)
#     plt.show()
