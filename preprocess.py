# Data preprocessing

import csv
import cv2
import numpy as np
import os
import pickle
from sklearn.utils import shuffle

# Read data
lines = []
with open('./sample_data/data/driving_log.csv') as sample_file:
    sample_file.readline()
    reader = csv.reader(sample_file)
    for line in reader:
        lines.append(line)
print(len(lines))


# Get images/features and turning angles/labels
images = []
angles = []
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
    left_angle = float(line[3]) + 0.10
    images.append(left_img)
    angles.append(left_angle)
    
    right_path = './sample_data/data/IMG/' + line[2].split('/')[-1]
    right_img = cv2.imread(right_path)
    right_img = cv2.resize(right_img, None, fx=0.25, fy=0.5)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    right_angle = float(line[3]) - 0.10
    images.append(right_img)
    angles.append(right_angle)
    
X_train = np.array(images)
y_train = np.array(angles)
print(X_train.shape, y_train.shape)


# Save the preprocessing data
def save_data(pickle_file):
    if not os.path.isfile(pickle_file):
        print('Saving data to pickle file...')
        try:
            with open(pickle_file, 'wb') as pfile:
                pickle.dump(
                    {
                        'X_train': X_train,
                        'y_train': y_train
                    },
                    pfile, protocol=2)
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise
    print('Data cached in pickle file.')
    
save_data('./pre-data.pickle')






