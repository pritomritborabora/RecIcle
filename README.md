# RecIcle
In this project we have built an image classifier which is able to distinguish between images of paper, cardboard, metal, glass and plastic.The dataset used contains 2527 different images belonging to one of the classes mentioned above. We have taken into consideration the small size of the dataset that we are working with and tried to come up with approaches to obtain good performances .We have implemented a convolutional neural network with Keras as framework and augmented different parameters of the network to improve the performance of the system.The CNN has three layers and  134 neurons. Each layer is followed by a max pooling layer which is attached to a fully connected layer of 64 neurons.We used CUDA extension for efficient implementation of the convolution operation and  “dropout” regularization method for reducing overfitting. Additionally, we have analyzed the effect of preprocessing the images. We obtained 80% accuracy which is significantly better that 75% accuracy of previous state of the art.

HOW TO RUN THE CODE:
--------------------

PREPARATION:
- install Python 3
- install Tensorflow with "pip install tensorflow"
- install Keras with "pip install keras"
- install Sklearn with "pip instal scikit-learn"
- install OpenCV with "pip install opencv-python"
- install Numpy with "pip install numpy"
- Download the dataset from https://drive.google.com/open?id=0B3P9oO5A3RvSSm9MdU91MUgxcW8
- Put it in ./data/original
- Create ./data/train
- Create ./data/validation
- Create ./data/test

OPTIONS:

python net.py <weight.h5> [-i <image>] [-e <epochs>]
 [-s <image size>] [-b <batch size>] [--train] [--test]

 <weight.h5>: Weight file to be loaded or saved. With h5 extension.
 
 -i <image>: Image filename to be predicted
 
 --train: Run the training procedure
 
 --test: Run the evaluation procedure
 
 -e <epochs>: Number of epochs when training
 
 -s <image size>: Size in pixels of the input images
 
 -b <batch size>: Batch size when training

 For example:
 
 python net.py best.h5 -i trash.png
 
 python net.py fix.h5 --train --test -e 100
