# Image-Captioning
The objective this project is to convert a given input image into a natural language description. The encoder-decoder framework is widely used for this task. The image encoder is a convolutional neural network (CNN). The decoder is a long short-term memory (LSTM) network.

In this project, we will use the dataset of image-caption pairs to train a CNN-RNN model to automatically generate images from captions.

[//]: # (Image References)

[image1]: ./images/encoder-decoder.png "CNN-RNN model architecture"

![CNN-RNN model architecture][image1]

### CNN Encoder
The encoder that provide to uses the pre-trained ResNet-50 architecture (with the final fully-connected layer removed) to extract features from a batch of pre-processed images. The output is then flattened to a vector, before being passed through a Linear layer to transform the feature vector to have the same size as the word embedding.

[image2]: ./images/encoder.png "CNN Encoder"

![CNN Encoder][image2]

### RNN Decoder

For language based model using (RNN/LSTM network), which transulate the features and feature vector giving by image based model (EncoderCNN) to a natural sentence.

For training LSTM model, we need predifined label and target text.

[image3]: ./images/decoder.png "RNN Decoder"

![RNN Decoder][image3]

### Implementations
1. [Data Analysis](https://github.com/kalyancv/Image-Captioning/edit/master/0_Dataset.ipynb)
  The Microsoft Common Objects in COntext (MS COCO) dataset is a large-scale dataset for scene understanding. The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms.
2. [Preliminaries](https://github.com/kalyancv/Image-Captioning/edit/master/1_Preliminaries.ipynb) In this notebook, how to load and pre-process data from the COCO dataset. and also design a CNN-RNN model for automatically generating image captions.
    *  Explore the Data Loader
    *  Use the Data Loader to Obtain Batches
    *  Experiment with the CNN Encoder
    *  Implement the RNN Decoder
3. [Training](https://github.com/kalyancv/Image-Captioning/edit/master/2_Training.ipynb) In this notebook, train the CNN-RNN model by specifying hyperparameters.
    *  Training Setup
    *  Training model
4. [Inference](https://github.com/kalyancv/Image-Captioning/edit/master/3_Inference.ipynb) In this notebook, trained model to generate captions for images in the test dataset.  
    *  Load Test Dataset
    *  Load Trained Models
    *  Finish the Sampler
    *  Clean up Captions
    *  Generate Predictions!
