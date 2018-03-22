from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Input, Layer
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.utils import plot_model
import tensorflow as tf
import mode as model
import keras
import json
import time
import sys
import cv2
import numpy as np
from keras.models import Model


from mode import create_model


if len(sys.argv) < 3:
        print('train.py modelname shutdown')
        exit()

file_name  = str(time.time())+"_"+sys.argv[1]
shutdown = sys.argv[2]

img_width, img_height = 150, 150
steps_per_epoch = 26
nb_epoch = 40
batch_size =  32

train_data_dir = './dataset/training'
validation_data_dir = './dataset/testing'
validation_steps = 10
learning_rate = 0.0001

# input_shape = shape(img_width, img_height, 3)



img_shape = shape=(150, 150, 3);

#input layer
nn_layer_1 = create_model();

in_a = Input(img_shape)
in_p = Input(img_shape)
in_n = Input(img_shape)


emb_a = nn_layer_1(in_a)
emb_p = nn_layer_1(in_p)
emb_n = nn_layer_1(in_n)


def triplet_generator():
    ''' Dummy triplet generator for API usage demo only.
    Will be replaced by a version that uses real image data later.
    :return: a batch of (anchor, positive, negative) triplets
    '''
    while True:
        a_batch = np.random.rand(4, 96, 96, 3)
        p_batch = np.random.rand(4, 96, 96, 3)
        n_batch = np.random.rand(4, 96, 96, 3)
        yield [a_batch , p_batch, n_batch], None


class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        a, p, n = inputs
        p_dist = K.sum(K.square(a - p), axis=-1)
        n_dist = K.sum(K.square(a - n), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

triplet_loss_layer = TripletLossLayer(alpha=0.2, name='triplet_loss_layer')([emb_a, emb_p, emb_n])


nn4_small2_train = Model([in_a, in_p, in_n], triplet_loss_layer)
generator = triplet_generator()

nn4_small2_train.compile(loss=None, optimizer='adam')
nn4_small2_train.fit_generator(generator, epochs=nb_epoch, steps_per_epoch=steps_per_epoch)


# optimizer = keras.optimizers.Adam(lr=learning_rate)

# model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])


# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     rotation_range=5.,
#     # horizontal_flip=True
#     )
#
# test_datagen = ImageDataGenerator(rescale=1. / 255)
#
#
# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical',
#     color_mode='grayscale',
#     seed=7,
#     )
#
# validation_generator = test_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical',
#     color_mode='grayscale',
#     seed=7,
#     )
#
# label_map = (train_generator.class_indices)
#
#
# tensorboard = TensorBoard(log_dir="logs/"+file_name+"_{}".format(time.time()))
#
#
# model.fit_generator(
#         train_generator,
#         steps_per_epoch=steps_per_epoch,
#         epochs=nb_epoch,
#         validation_data=validation_generator,
#         validation_steps=validation_steps,
#         callbacks=[tensorboard],
# )

# to save model architecture
outfile = open('./models/'+file_name+'_convnet_model.json', 'w') # of course you can specify your own file locations
json.dump(nn4_small2_train.to_json(), outfile)
outfile.close()

# to save model weights
nn4_small2_train.save_weights('./models/'+file_name+'_convnet_weights.h5')

# to load model architecture
infile = open('./models/'+file_name+'_convnet_model.json')
model = keras.models.model_from_json(json.load(infile))
infile.close()

# to load model weights
# model.load_weights('./models/'+file_name+'_convnet_weights.h5')

# model.save('./models/'+file_name+'_model.h5')

# if(shutdown == True) {
#     os.system('sudo shutdown now -P')
# }