from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

# Check https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

batch_size = 16
img_width, img_height = 64, 64

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size = (img_width, img_height),  # all images will be resized to 150x150
        batch_size = batch_size,
        class_mode='categorical') 


validation_generator = validation_datagen.flow_from_directory(
        'data/validation',
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'categorical')