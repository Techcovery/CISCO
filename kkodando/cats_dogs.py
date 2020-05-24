from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
from keras import backend as K
from matplotlib import pyplot


# dimensions of our images.
img_width, img_height = 150, 150
# We have total of 11500 training images for each category
train_data_dir = 'dogscats/train'
validation_data_dir = 'dogscats/valid'
#nb_train_samples = 2000, nb_validation_samples = 800
#accuracy: 0.6988
#nb_train_samples = 8000, nb_validation_samples = 800
#_accuracy: 0.7925

#accuracy: 0.8050
#_accuracy: 0.8325
# accuracy: 0.8625
nb_train_samples = 8000
nb_validation_samples = 800
epochs = 10
batch_size = 16

#Greyscale images have single channel. RGB have 3 channels
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

#Training data generator
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

#Testing data generator
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

ind = train_generator.class_indices

print('Indices....', ind)

# Callbacks
from keras.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)

callbacks = [earlystop]

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

steps = nb_train_samples // batch_size
print('Steps per epoch--------------------: ',steps)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=callbacks
)
model.save('cats_dogs_model')

# plot training history
pyplot.plot(history.history['accuracy'],color='b', label='Training Accuracy')
pyplot.plot(history.history['val_accuracy'],color='r', label='Testing Accuracy')
pyplot.legend()
pyplot.show()

import numpy as np
from keras.preprocessing import image

img_pred = image.load_img('dogscats/valid/dogs/dog.9804.jpg', target_size = (150, 150))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)
rslt = model.predict(img_pred)

if rslt[0][0] == 1:
    prediction = "dog"
else:
    prediction = "cat"

print(prediction)