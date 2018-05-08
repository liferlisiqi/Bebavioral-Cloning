# Behaviorial Cloning

Overview
---
In this project, I will use deep neural networks and convolutional neural networks to clone driving behavior by End-to-End learning. I will train, validate and test a model using Keras. The model will output a steering angle to guide an autonomous vehicle. The data I use it captured by three cameras, so there will be a step to combine three images to train the model.

The Project goals/steps
---
* Use the simulator to collect data of good driving behavior. 
* Design, train and validate a model that predicts a steering angle from image data.
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results.

Dependences
---
This project requires three environments, the first for training, the second for testing and the last for simulation:

* The training environment requires:

The already obtained training data captured from the simulator can be download [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip). After your model is trained, `model.h5` will be saved for testing.

* The testing environment requires:

The following command is used for connecting to simulator to testing your model.
```sh
python drive.py model.h5
(if you're using Docker)docker run -it --rm -p 4567:4567 -v `pwd`:/src udacity/carnd-term1-starter-kit python drive.py model.h5
```

* The simulation environment  
The simulator is provided by Udacity, more information can be found in this [repository](https://github.com/udacity/self-driving-car-sim). 


Ps: There are two docker images may help you.
* Docker image [lsq-v1](https://github.com/liferlisiqi/Udacity-CarND-term1/blob/master/P3-behavioral-cloneing/lsq-v1) for training on GPU. 
* Docker image [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit) for training on CPU and testing. 

#### More details

#### `lsq-v1`
In order to train model on GPU with Tensorflow/Keras, I write this dockerfile. Usage of this docker image contains two steps: build docker image and run docker image, once you built docker image you don't have to do it again.
```sh
Step1: build docker image
docker build -t=lsq:v1 -f=lsq-v1 .

Step2: run docker image
(jupyter) nvidia-docker run -it --rm -p 8888:8888 -v `pwd`:/notebooks lsq:v1
(bash) nvidia-docker run -it --rm -v `pwd`:/notebooks lsq:v1 bash
```
Ps: this docker image can only be used for training until now.
 
#### `drive.py`
Once the model has been saved, it can be used with drive.py using this command:
```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

When your model can correctly predict the steering angle, you can aving a video of the autonomous agent by:

```sh
python drive.py model.h5 video1
```
The fourth argument, `video1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten. The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

#### `video.py`
```sh
python video.py video1
```
Creates a video based on images found in the `video1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `video1.mp4`.


[//]: # (Image References)
[bgr]: ./result_images/bgr.jpg
[rgb]: ./result_images/rgb.jpg
[rgb_resize]: ./result_images/rgb_resize.jpg

Data preprocessing
---
I use the sample data provided by Udacity to train my model, the sample data contains imagines capture by three camera(left/center/right) and a .csv file recording the steering angles. All the imagines captured by three cameras are used to train my model, which will predict steering angles for left\right to correct the behavioral of vihicle.
##### 1. Original data
The images read through cv2.imread() are in BGR color space as following:  
![alt text][bgr]   
And plt.imread() read imagines in RGB color space:    
![alt text][rgb]  
Since the testing images are in RGB color space, so it it important that we should use *plt.imread()* to read training data.  

##### 2. Genometric transformation
For training efficiency, I change the shape of imagine from (160, 320) to (80, 80), meanwhile we should also resize image when testing by modifing the `drive.py` at line 64. 
![alt text][rgb_resize]

##### 3. Angle modification
This is the core operation of using images captured by three cameras to train and test the model. There is another operation to process the image, which is to crop the image from (80,80) to (80,32) with keras. And this is some kinds of data augmentation, through adding the images captured by left and right camera, we have three times of training data. More important, the labels for left and right images are modified to be used as tuning situation.
The steering angles(labels) for left/right camera are modified by adding +/-0.10 to make the vihicle performs better.
```sh
left_angle = float(line[3]) + 0.10
right_angle = float(line[3]) - 0.10
```

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
  
The whole model architecture is as following.  

| Layer         	|     Description	        					| Output     | Activation |
|:-----------------:|:---------------------------------------------:|:----------:|:----------:|
| Lambda            | Normalize imagine from [0,255] to [-0.5,0.5]  | 80x80x3    |  		  |
| Cropping          | Crop imagine from (80, 80) to (80, 32)        | 80x32x3    |  	      |
| Convolution       | kernel: 3x3; stride:2x2; padding: valid  	    | 39x15x24   | Relu       |
| Convolution       | kernel: 3x3; stride:2x2; padding: valid 	    | 19x7x36    | Rule       |
| Convolution       | kernel: 3x3; stride:1x1; padding: valid 	    | 17x5x48    | Relu       |
| Convolution       | kernel: 3x3; stride:1x1; padding: valid 	    | 15x3x64    | Relu       |
| Convolution       | kernel: 3x3; stride:1x1; padding: valid 	    | 13x1x64    | Relu       |
| Dropout		    | Avoiding overfitting      					| 13x1x64    |  		  |
| Flatten		    | Input 13x1x64 -> Output 832				    | 832        |  	      |
| Fully connected	| connect every neurel with next layer 		    | 100        |  	      |
| Fully connected	| connect every neurel with next layer	        | 50         |  	      |
| Fully connected	| connect every neurel with next layer  		| 10         | 		      |
| Fully connected	| connect every neurel with next layer  		| 1          | 		      |


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
### train with generator
Another efficient way to train the model is using generator provided by Keras, it allows you to do real-time data augmentation on images on CPU in parallel to training your model on GPU.([More details](https://faroit.github.io/keras-docs/1.2.2/models/sequential/))

Testing result
---
The model was tested by running it through the simulator and the vehicle is able to drive autonomously around the track 1 without leaving the road. The [result video](https://youtu.be/gs2o7dtvt-E) can be watched on Youtube too.  

Summary
---
In this project, I've trained a model to guide an autonomous car running in a simulator by End-to-End learning. The autonomous car owns three cameras: left, center and right, thus, all the images captured by these camera are used to train the model after some preprocess. The result show that it can perfected running in the simulator by itself.

References
---
[End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)
[Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim)

