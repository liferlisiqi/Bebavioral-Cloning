# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results

[//]: # (Image References)

[center]: ./examples/center.jpg "center"
[left]: ./examples/left.jpg "left"
[right]: ./examples/right.jpg "right"
[center_bgr]: ./examples/center_bgr.jpg "center"
[left_bgr]: ./examples/left_bgr.jpg "left"
[right_bgr]: ./examples/right_bgr.jpg "right"
[center_resize]: ./examples/center_resize.jpg "center_resize"
[left_resize]: ./examples/left_resize.jpg "left_resize"
[right_resize]: ./examples/right_resize.jpg "right_resize"
[center_rgb]: ./examples/center_rgb.jpg "center_resize"
[left_rgb]: ./examples/left_rgb.jpg "left_resize"
[right_rgb]: ./examples/right_rgb.jpg "right_resize"


---
### Files Submitted & Code Quality

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 showing the testing video on track 1
* writeup.md summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model

The data is normalized in the model using a Keras lambda layer and croped from (80, 80) to (80, 32).
```sh
nvidia.add(Lambda(lambda x: x/255. - 0.5, input_shape=(80, 80, 3)))
nvidia.add(Cropping2D(cropping=((35, 13), (0, 0))))
```

Based on the model of Nvidia, my model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 64. And the The model includes RELU layers to introduce nonlinearity.
```sh
nvidia.add(Convolution2D(24, 3, 3, subsample=(2, 2), activation='relu'))
nvidia.add(Convolution2D(36, 3, 3, subsample=(2, 2), activation='relu'))
nvidia.add(Convolution2D(48, 3, 3, activation='relu'))
nvidia.add(Convolution2D(64, 3, 3, activation='relu'))
nvidia.add(Convolution2D(64, 3, 3, activation='relu'))
```

The model contains dropout layer between convolution layers and fully connected layers in order to reduce overfitting. 
```sh
nvidia.add(Dropout(0.5))
```

After convolution layer, I add four fully connected layers from 100 to 1.
```sh
nvidia.add(Flatten())
nvidia.add(Dense(100))
nvidia.add(Dense(50))
nvidia.add(Dense(10))
nvidia.add(Dense(1))
```


### Data preprocessing

##### 1. Original data
![alt text][left] ![alt text][center] ![alt text][right]

![alt text][left_bgr] ![alt text][center_bgr] ![alt text][right_bgr]

##### 2. Genometric transformation
![alt text][left_resize] ![alt text][center_resize] ![alt text][right_resize]

##### 3. Change colorspaces
![alt text][left_rgb] ![alt text][center_rgb] ![alt text][right_rgb]

##### 4. Data shuffling
I finally randomly shuffled the data set and put Y% of the data into a validation set.
```sh
X_train, y_train = shuffle(X_train, y_train)
```

### Training

The model choosed mean square error(MSE) to be loss function and used an keras.optimizers.Adam optimizer, the learning rate can be tuned manually with explanation.
```sh
LEARNING_RATE = 1e-4
nvidia.compile(loss='mse', optimizer=Adam(LEARNING_RATE))
```

The model was trained and validated on different data sets(8:2) to ensure that the model was not overfitting. The training epoch was set 5, since the model is goog enough by then,
```sh
EPOCHS = 5
nvidia.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=EPOCHS)
```

Finally, the model was save as model.h5 for testing.
```sh
nvidia.save('model.h5')
```


### Testing result
The model was tested by running it through the simulator and the vehicle is able to drive autonomously around the track 1 without leaving the road. The [video](https://youtu.be/bXbnlHCgiVU) can be watched on Youtube too.
