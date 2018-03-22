from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.utils import plot_model
import tensorflow as tf
import keras
import json
import time
import sys
import os
import cv2


from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

train_data_dir = './dataset/training'
validation_data_dir = './dataset/testing'
validation_steps = 10
learning_rate = 0.0001

img_width, img_height = 150, 150
steps_per_epoch = 27
nb_epoch = 40
batch_size =  10

# vgg_features = VGGFace(include_top=False, input_shape=(img_width, img_height, 3), pooling='avg') # pooling: None, avg or max

nb_class = 4
hidden_dim = 512

vgg_model = VGGFace(include_top=False, input_shape=(img_width, img_height, 3))
last_layer = vgg_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
out = Dense(nb_class, activation='softmax', name='fc8')(x)
model = Model(vgg_model.input, out)

optimizer = keras.optimizers.Adam(lr=learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=5.,
    # horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    # color_mode='grayscale',
    seed=7,
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    # color_mode='grayscale',
    seed=7,
)

model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=nb_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    # callbacks=[tensorboard],
)

model.save_weights('./models/vgg-face.h5')
outfile = open('./models/vgg-face.json',
               'w')  # of course you can specify your own file locations
json.dump(model.to_json(), outfile)
outfile.close()