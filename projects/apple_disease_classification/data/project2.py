
"""
help(os.listdir)

Help on built-in function listdir in module nt:

listdir(path=None)
    Return a list containing the names of the files in the directory.
    
    path can be specified as either str, bytes, or a path-like object.  If path is bytes,
      the filenames returned will also be bytes; in all other circumstances
      the filenames returned will be str.
    If path is None, uses the path='.'.
    On some platforms, path may also be specified as an open file descriptor;\
      the file descriptor must refer to a directory.
      If this functionality is unavailable, using it raises NotImplementedError.
    
    The list is in arbitrary order.  It does not include the special
    entries '.' and '..' even if they are present in the directory.
"""
"""
from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('data/train/cats/cat.0.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely
"""
"""
model.add(layers.Dense(256, activation = tf.keras.layers.LeakyReLU(alpha=0.3)))
"""
"""
#!/usr/bin/env python

# DEFINE IMPORTS HERE

# IMPLEMENT RUNNABLE CODE INSIDE THIS MAIN 
def main():
    pass

# DO NOT IMPLEMENT ANYTHING HERE
if __name__ == main():
    main()    
"""

""""

# - Er worden twee DCNN modellen gebruikt, de eerste als generator, de tweede als discriminator

# - Leaky Relu wordt in beide modellen gebruikt als activatiefunctie ipv de normale Relu, waardoor het trainen sneller gaat en het "Zero-Death" effect wegvalt

# - In het generator model maken de onderzoekers gebruik van Tanh in de output layer

# - Het 'eigen' model scoort hier beter dan de pre-trained modellen

# - ResNet50 komt naar voren als het meest effectieve pre-trained model waarmee het eigen model vergeleken is (96% Overall Accuracy)

# - MiniVVGNet valt voor mij af omdat deze de laagste Overall Accuracy genereert in dit voorbeeld

# - ADAM komt in alle gevallen naar voren als de meest effectieve optimizer met de hoogste overall accuracy

# - Bij het trainen van het model worden slechts 35 epochs gebruikt, bij het 'gebruik' slechts 10 epochs

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, Sequential
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import vgg16
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential



data_dir = 'projects/apple_disease_classification/data/Train'

img_height = 360
img_width = 360
batch_size = 32


train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


import matplotlib.pyplot as plt

class_names = train_ds.class_names
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

    plt.show()



tf.keras.Layers.Rescaling(scale=1./255, offset=0.0)

model = keras.models.Sequential()

model.add(layers.Input(shape=(350, 350, 3)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.1))

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(4))

model.summary()
