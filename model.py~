
# import modules

import csv
import cv2
import numpy as np
import os
import pickle
from tqdm import tqdm
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
print("import modules")


# Lead data file

lines = []
with open('./sample_data/data/driving_log.csv') as sample_file:
    sample_file.readline()
    reader = csv.reader(sample_file)
    for line in reader:
        lines.append(line)
print(len(lines))


# Load data and preprocess

images = []
angles = []
pbar = tqdm(total=len(lines))
for line in lines:
    center_path = './sample_data/data/IMG/' + line[0].split('/')[-1]
    center_img = cv2.imread(center_path)
    center_img = cv2.resize(center_img, None, fx=0.25, fy=0.5)
    center_img = cv2.cvtColor(center_img, cv2.COLOR_BGR2RGB)
    center_angle = float(line[3])
    images.append(center_img)
    angles.append(center_angle)
    
    left_path = './sample_data/data/IMG/' + line[1].split('/')[-1]
    left_img = cv2.imread(left_path)
    left_img = cv2.resize(left_img, None, fx=0.25, fy=0.5)
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    left_angle = float(line[3]) + 0.15
    images.append(left_img)
    angles.append(left_angle)
    
    right_path = './sample_data/data/IMG/' + line[2].split('/')[-1]
    right_img = cv2.imread(right_path)
    right_img = cv2.resize(right_img, None, fx=0.25, fy=0.5)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    right_angle = float(line[3]) - 0.15
    images.append(right_img)
    angles.append(right_angle)
    pbar.update(1)

pbar.close()    
X_train = np.array(images)
y_train = np.array(angles)
X_train, y_train = shuffle(X_train, y_train)
print(X_train.shape, y_train.shape)


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
