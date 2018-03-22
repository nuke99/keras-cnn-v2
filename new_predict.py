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
from fr_utils import *
from facemodel import *
from multiprocessing.dummy import Pool

# from utils import preprocess_input



from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import glob


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# dimensions of our images
img_width, img_height = 96, 96

names = {
    0 : 'Chris_patt',
    1 : 'Dwayne_Johnson',
    2 : 'Jeniffer_Lawrance',
    3 : 'Laurence_Fishburne'
}

if len(sys.argv) < 3:
        print('train.py <dir> <modelname>')
        exit()

_model =  sys.argv[2]
_dir =  sys.argv[1]

model_name = _model

# load the model we saved
# model = load_model('./models/1516228873.3_fixedepos_model.h5')
infile = open('./models/'+model_name+'_convnet_model.json')
# model = keras.models.model_from_json(json.load(infile))
model = faceRecoModel(input_shape=(3, 96, 96))


#1516262902.05_last-1_convnet_weights.h5
model.load_weights('./models/'+model_name+'_convnet_weights.h5')
# model.load_weights('./1520110529.21_new-test-3_weights.best.hdf5')


def prepare_database():
    database = {}

    # load all the images of individuals to recognize into the database
    for _file in glob.glob(_dir+'*.jpg'):


        identity = os.path.splitext(os.path.basename(_file))[0]
        print _file
        database[identity] = img_path_to_encoding(_file, model)

    return database

database = prepare_database()





dir_files = glob.glob(_dir+'*.jpg')

for _file in dir_files:
    file_path = _file
    

    img = image.load_img(file_path, 
    target_size=(img_width, img_height),
    
    )
   
    # img = img.convert('L')
    
    x = image.img_to_array(img)
    x = x /255.0
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])

    print _file

    classes = (model.predict(images))
    print classes

    encoding = img_to_encoding(images, model)
    min_dist = 100
    identity = None
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(db_enc - encoding)
        print('distance for %s is %s' %(name, dist))
        if dist < min_dist:
            min_dist = dist
            identity = name
            if min_dist > 0.52:
                print "nope"
            else:
                print str(identity)


    # p_classes = model.predict_classes(images)
    # print names[p_classes[0]]
    
