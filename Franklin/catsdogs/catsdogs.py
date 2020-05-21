# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:55:22 2020

@author: fjesudha
"""
import os

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator


IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
BATCH_SIZE = 64

input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)

def get_model():
    model = Sequential()
    
    model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=input_shape, activation='relu'))
    model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
        
    model.compile(loss='binary_crossentropy',
                optimizer=RMSprop(lr=0.0001),
                metrics=['accuracy'])
    
    return model
    
if __name__ == '__main__':
    path = ""
    training_data_dir = "training" # 10 000 * 2
    validation_data_dir = "validation" # 2 500 * 2
    
    test_data_dir = "test" # 12 500
    
    training_data_generator = ImageDataGenerator(
                                                rescale=1./255,
                                                shear_range=0.1,
                                                zoom_range=0.1,
                                                horizontal_flip=True)
    validation_data_generator = ImageDataGenerator(rescale=1./255)
    test_data_generator = ImageDataGenerator(rescale=1./255)

    training_generator = training_data_generator.flow_from_directory(
                                        training_data_dir,
                                        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                        batch_size=BATCH_SIZE,
                                        class_mode="binary")
    
    validation_generator = validation_data_generator.flow_from_directory(
                                        validation_data_dir,
                                        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                        batch_size=BATCH_SIZE,
                                        class_mode="binary")
    
#    test_generator = test_data_generator.flow_from_directory(
#                                        test_data_dir,
#                                        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
#                                        batch_size=1,
#                                        class_mode="binary", 
#                                        shuffle=False)
#                                    
    model = get_model()

    print(os.getcwd())
    print(training_generator)    
    
    history = model.fit_generator(training_generator, steps_per_epoch=len(training_generator),
                                      validation_data=validation_generator, 
                                      validation_steps=len(validation_generator), 
                                      epochs=20, verbose=1)
    
    acc = model.evaluate_generator(validation_generator, steps=len(validation_generator), verbose=1)
    print('> %.3f' % (acc * 100.0))
