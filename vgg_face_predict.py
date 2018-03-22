from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
K.set_image_data_format('channels_first')
from keras.callbacks import TensorBoard
from keras.utils import plot_model
import tensorflow as tf
import keras
import json
import time
import sys
import os
import cv2
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import glob
import numpy as np
from keras.engine import  Model
from keras.layers import Input, Layer

keras.backend.set_image_data_format('channels_last');


img_width, img_height = 95, 95
names = {
    0 : 'Chris_patt',
    1 : 'Dwayne_Johnson',
    2 : 'Jeniffer_Lawrance',
    3 : 'Laurence_Fishburne'
}
nb_class = 4
hidden_dim = 512


_model_file = './models/vgg-face.json';
infile = open(_model_file)
model = keras.models.model_from_json(json.load(infile))
model.load_weights('./models/vgg-face.h5')


# model = VGGFace() # default : VGG16 , you can use model='resnet50' or 'senet50'
# vgg_model = VGGFace(include_top=False, input_shape=(img_width, img_height,3), classes=4)
# last_layer = vgg_model.get_layer('pool5').output
# x = Flatten(name='flatten')(last_layer)
# x = Dense(hidden_dim, activation='relu', name='fc6')(x)
# x = Dense(hidden_dim, activation='relu', name='fc7')(x)
# out = Dense(nb_class, activation='softmax', name='fc8')(x)
# model = Model(vgg_model.input, out)

# model = VGGFace(weights='./models/vgg-face.h5',input_shape=(224, 224, 3))


if len(sys.argv) < 3:
        print('train.py <dir> <modelname>')
        exit()

_model =  sys.argv[2]
_dir =  sys.argv[1]

dir_files = glob.glob(_dir+'*.jpg')



for _file in dir_files:
    file_path = _file
    img = image.load_img(file_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1)  # or version=2
    preds = model.predict(x)
    print('Predicted:', utils.decode_predictions(preds))

