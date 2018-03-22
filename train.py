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
import cv2


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

input_shape = (img_width, img_height, 1)

model = Sequential()


#input layer
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=input_shape))
model.add(Activation('relu'))
model.add(Dropout(0.25))


#convolusion layer
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# second converlusional layer
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.25))

# third convolusion layer
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#output layer
model.add(Flatten()) #<-fully connected
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4)) # <- outputs
model.add(Activation('softmax'))


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
    color_mode='grayscale',
    seed=7,
    )

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    seed=7,
    )

label_map = (train_generator.class_indices)


tensorboard = TensorBoard(log_dir="logs/"+file_name+"_{}".format(time.time()))


model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[tensorboard],
)

# to save model architecture
outfile = open('./models/'+file_name+'_convnet_model.json', 'w') # of course you can specify your own file locations
json.dump(model.to_json(), outfile)
outfile.close()

# to save model weights
model.save_weights('./models/'+file_name+'_convnet_weights.h5')

# to load model architecture
infile = open('./models/'+file_name+'_convnet_model.json')
model = keras.models.model_from_json(json.load(infile))
infile.close()

# to load model weights
model.load_weights('./models/'+file_name+'_convnet_weights.h5')

# model.save('./models/'+file_name+'_model.h5')

# if(shutdown == True) {
#     os.system('sudo shutdown now -P')    
# }