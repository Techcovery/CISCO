from __future__ import print_function
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation
from keras.layers.convolutional import *
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pylab as plt

batch_size = 128
num_classes = 10
epochs = 10

# input image dimensions
img_x, img_y = 224,224
num_classes = 2

train_path='cats-and-dogs/train'
test_path='cats-and-dogs/valid'

train_batches = ImageDataGenerator.flow_from_directory(train_path, target_size=(224,224),classes=['dog','cat'],batch_size=64)
test_batches = ImageDataGenerator.flow_from_directory(test_path, target_size=(224,224),classes=['dog','cat'],batch_size=64)

input_shape = (img_x, img_y, 3)
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model.fit_generator(train_batches,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=test_batches,
          callbacks=[history])
score = model.evaluate_generator(test_batches, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
