[//]: # (Image References)

[image1]: ./images/key_pts_example.png "Facial Keypoint Detection"

# Facial Keypoint Detection

## Project Overview

In this project, Using computer vision techniques and deep learning architectures to build a facial keypoint detection system. 

Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition.

![Facial Keypoint Detection][image1]

### Project Files:

__Notebook 1__ : Loading and Visualizing the Facial Keypoint Data

__Notebook 2__ : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

__Notebook 3__ : Facial Keypoint Detection Using Haar Cascades and your Trained CNN


### Local Environment Instructions

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/kalyancv/Facial_Keypoints.git
cd Facial_Keypoints
```

2. Create (and activate) a new environment, named `cv` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n cv python=3.6
	source activate cv
	```
	- __Windows__: 
	```
	conda create --name cv python=3.6
	activate cv
	```
3. Install PyTorch and torchvision; this should install the latest version of PyTorch.
	
	- __Linux__ or __Mac__: 
	```
	conda install pytorch torchvision -c pytorch 
	```
	- __Windows__: 
	```
	conda install pytorch-cpu -c pytorch
	pip install torchvision
	```

6. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```


### Data

All of the data you'll need to train a neural network is in the Facial_Keypoints repo, in the subdirectory `data`. In this folder are training and tests set of image/keypoint data, and their respective csv files. This will be further explored in Notebook 1: Loading and Visualizing Data.


LICENSE: This project is licensed under the terms of the MIT license.
