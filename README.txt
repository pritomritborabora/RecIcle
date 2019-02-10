----------------------------------------
Image Classification for Trash Recyclability 
using a Convolutional Neural Network
----------------------------------------

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