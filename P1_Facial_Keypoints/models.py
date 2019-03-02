## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #input (1X224X224) 
        self.in_planes = 32
        self.conv1 = nn.Conv2d(1, 32,  kernel_size = 5, stride = 1, padding=1)  #32x224x224
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x111x111 
        
        self.conv2 = nn.Conv2d(32, 64,  kernel_size = 4, stride = 1, padding=0)  #128x111x111
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 64x54x54 
        
        self.conv3 = nn.Conv2d(64, 128,  kernel_size = 3, stride = 1, padding=1) #64x54x54
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 128x27x27
        
        self.conv4 = nn.Conv2d(128, 256,  kernel_size = 2, stride = 1, padding=1) #128x27x27
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # 256x14x14
        
        self.conv5 = nn.Conv2d(256, 256,  kernel_size = 1, stride = 1, padding=1) #256x14x14
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2, 2)  # 256x8x8
        
        self.fc1 =  nn.Linear(256*8*8, 512)
        self.fc2 =  nn.Linear(512, 256)
        self.fc3 =  nn.Linear(256, 68*2)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.3)
        #self.dropout5 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout(0.4)
        self.dropout6 = nn.Dropout(0.4)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, mu\ltiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.dropout1(self.pool1(F.elu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(F.elu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(F.elu(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(F.elu(self.bn4(self.conv4(x)))))
        x = self.dropout5(self.pool5(F.elu(self.bn5(self.conv5(x)))))
        
        # Flattening
        x = x.view(x.size(0), -1)
        x = self.dropout5(F.elu(self.fc1(x)))
        x = self.dropout6(F.elu(self.fc2(x)))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

