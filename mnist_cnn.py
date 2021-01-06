import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D

#Tutorial url: https://towardsdatascience.com/build-your-own-convolution-neural-network-in-5-mins-4217c2cf964f

"""
Convolution - take small filter on image and perform ops (matrix multiplication, etc.)
Pooling - downsampling 
    - reduces required computation
    - avoids overfitting

Early layers - detect things like edges
Later layers - detect more complex features

Dense layers - predict labels

Dropout layer - reduces overfitting

Flatten layer - expands 3D vector to 1D
"""

batch_size = 128

#Images are from 0-9, 10 options
num_classes = 10

#Rounds of training
epochs = 12

#Images are 28*28
img_rows, img_cols = 28, 28

#Each raw data point is 28 1D arrays of 28 elements, making a 28*28 2D array
#Each pic is in grayscale, meaning each number is 0-255, rather than having 3
#separate RGB values
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

#Target variables go from 1, 2, 3 etc, to essentially one-hot vectors
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#Model building

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

