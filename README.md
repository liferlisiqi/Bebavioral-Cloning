# Behaviorial Cloning

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains all the files for the Behavioral Cloning Project.

In this project, I will use  deep neural networks and convolutional neural networks to clone driving behavior. I will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle. 

I'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track in the simulator provided by Udacity.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior. 
* Design, train and validate a model that predicts a steering angle from image data.
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results.

Dependencies and environment
---
This lab requires:

* Docker image [lsq-v1](https://github.com/liferlisiqi/Udacity-CarND-term1/blob/master/P3-behavioral-cloneing/lsq-v1) for training. 
* Docker image [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit) for testing. 
* [Simulator](https://github.com/udacity/self-driving-car-sim) for testing and collecting data.

### Details About Files In This Directory

### `lsq-v1`
Although the docker image `CarND Term1 Starter Kit` provided by Udacity can be used to train model, it is based on CPU. In order to train model on GPU with Tensorflow/Keras, I write this dockerfile. Usage of this docker image contains two steps: build docker image and run docker image, once you built docker image you don't have to do it again.
```sh
Step1: build docker image
docker build -t=lsq:v1 -f=lsq-v1 .

Step2: run docker image
(jupyter) nvidia-docker run -it --rm -p 8888:8888 -v `pwd`:/notebooks lsq:v1
(bash) nvidia-docker run -it --rm -v `pwd`:/notebooks lsq:v1 bash
```

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.


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


Model
---
The data is normalized in the model using a Keras lambda layer and croped from (80, 80) to (80, 32).
```sh
nvidia.add(Lambda(lambda x: x/255. - 0.5, input_shape=(80, 80, 3)))
nvidia.add(Cropping2D(cropping=((35, 13), (0, 0))))
```

Based on the model of Nvidia, my model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 64. And the The model includes RELU layers to introduce nonlinearity. The Original filter of the first two layers are 5x5, which is not suitable for my model, so I change to 3x3.
```sh
nvidia.add(Convolution2D(24, 3, 3, subsample=(2, 2), activation='relu'))
nvidia.add(Convolution2D(36, 3, 3, subsample=(2, 2), activation='relu'))
nvidia.add(Convolution2D(48, 3, 3, activation='relu'))
nvidia.add(Convolution2D(64, 3, 3, activation='relu'))
nvidia.add(Convolution2D(64, 3, 3, activation='relu'))
```

To combat the overfitting, I add dropout layer(0.5) between convolution layers and fully connected layers. 
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

The final model architecture is as following:
| Layer         		|     Description	        					| Input     | Output     | Activation |
|:---------------------:|:---------------------------------------------:|:---------:|:----------:|:-----------:
| Lambda             	| Normalize imagine from 0~255 to -0.5~0.5      | 80x80x3   | 80x80x3    |  		  |
| Cropping            	| Crop imagine from (80, 80) to (80, 32)        | 80x80x3   | 80x32x3    |  		  |
| Convolution       	| kernel: 3x3; stride:2x2; padding: valid  	    | 80x32x3   | 39x15x24   | Relu       |
| Convolution       	| kernel: 3x3; stride:2x2; padding: valid 	    | 39x15x24  | 19x7x36    | Rule       |
| Convolution       	| kernel: 3x3; stride:1x1; padding: valid 	    | 19x7x36   | 17x5x48    | Relu       |
| Convolution       	| kernel: 3x3; stride:1x1; padding: valid 	    | 17x5x48   | 15x3x64    | Relu       |  
| Convolution       	| kernel: 3x3; stride:1x1; padding: valid 	    | 15x3x64   | 13x1x64    | Relu       |
| Dropout				| Avoiding overfitting      					| 13x1x64   | 13x1x64    |  		  |
| Flatten				| Input 13x1x64 -> Output 832					| 13x1x64   | 832        |  		  |
| Fully connected		| connect every neurel with next layer 		    | 832       | 100        |  		  |
| Fully connected		| connect every neurel with next layer	        | 100       | 50         |  		  |
| Fully connected		| connect every neurel with next layer  		| 50        | 10         |  		  |
| Fully connected		| output a prediction of steering angle  		|


Data preprocessing
---
I use the sample data provided by Udacity to train my model, the sample data contains imagines capture by three camera(left/center/right) and a .csv file recording the steering angles. All the imagine captured by three camera are used to train my model and steering angles for left and right  are moddified to correct the behavioral of vihicle.
##### 1. Original data
The original imagine is in RGB and in (160, 320).  
![alt text][left] ![alt text][center] ![alt text][right]  
But cv2.imread() read imagines as BGR and this will have a great impact.  
![alt text][left_bgr] ![alt text][center_bgr] ![alt text][right_bgr]

##### 2. Genometric transformation
For training efficiency and accurancy, I change the shape of imagine from (160, 320) to (80, 80)  
![alt text][left_resize] ![alt text][center_resize] ![alt text][right_resize]

##### 3. Change colorspaces
Then, I change the colorspace of the resized imagine.  
![alt text][left_rgb] ![alt text][center_rgb] ![alt text][right_rgb]

##### 4. Angle modification
The steering angles(labels) for left/right camera are modified by adding +/-0.15 to make the vihicle performs better.
```sh
left_angle = float(line[3]) + 0.15
right_angle = float(line[3]) - 0.15
```

##### 5. Data shuffling
Finally, I randomly shuffled the data set and put Y% of the data into a validation set.
```sh
X_train, y_train = shuffle(X_train, y_train)
```

Training
---
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


Testing result
---
The model was tested by running it through the simulator and the vehicle is able to drive autonomously around the track 1 without leaving the road. The [video](https://youtu.be/bXbnlHCgiVU) can be watched on Youtube too.  
Moreover, I've tried to use generator to read data, but the vehicle performs much worse, thus, I give up using generator.

