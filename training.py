# Train the model

import pickle
import numpy as np
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam


# Reload the data

def reload_data(pickle_file):
    print('reload: ', pickle_file)
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        X_train = pickle_data['X_train']
        y_train = pickle_data['y_train']
        del pickle_data  # Free up memory
    return X_train, y_train

X_train, y_train = reload_data('./pre-data.pickle')
X_train, y_train = shuffle(X_train, y_train)
print('X_train shape: ', X_train.shape, 'y_train shape: ',y_train.shape)


# Model architecture

nvidia = Sequential()
nvidia.add(Lambda(lambda x: x/255. - 0.5, input_shape=(80, 80, 3)))
nvidia.add(Cropping2D(cropping=((35, 13), (0, 0))))
nvidia.add(Convolution2D(24, 3, 3, subsample=(2, 2), activation='relu'))
nvidia.add(Convolution2D(36, 3, 3, subsample=(2, 2), activation='relu'))
nvidia.add(Convolution2D(48, 3, 3, activation='relu'))
nvidia.add(Convolution2D(64, 3, 3, activation='relu'))
nvidia.add(Convolution2D(64, 3, 3, activation='relu'))
nvidia.add(Dropout(0.5))
nvidia.add(Flatten())
nvidia.add(Dense(100))
nvidia.add(Dense(50))
nvidia.add(Dense(10))
nvidia.add(Dense(1))


# Training method

# Hyperparameters
LEARNING_RATE = 1e-4
EPOCHS = 5

# Training
nvidia.compile(loss='mse', optimizer=Adam(LEARNING_RATE))
nvidia.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=EPOCHS)
nvidia.save('model.h5')

