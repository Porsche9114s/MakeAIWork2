{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'tensorflow'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-1-b3dd24ab6b17>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      9\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpreprocessing\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mLabelEncoder\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     10\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpreprocessing\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mOneHotEncoder\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 11\u001b[1;33m \u001b[1;32mimport\u001b[0m \u001b[0mtensorflow\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0mtf\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     12\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     13\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0mkeras\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mlayers\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'tensorflow'"
     ]
    }
   ],
   "source": [
    "import cv2 \n",
    "import os \n",
    "import glob \n",
    "import numpy as np\n",
    "\n",
    "\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "import tensorflow as tf\n",
    "\n",
    "from keras import layers\n",
    "from keras.preprocessing import image\n",
    "from keras.applications.imagenet_utils import preprocess_input\n",
    "from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D\n",
    "from keras.layers import AveragePooling2D, MaxPooling2D, Dropout\n",
    "from keras.models import Model\n",
    "\n",
    "import keras.backend as K\n",
    "from keras.models import Sequential\n",
    "\n",
    "from tensorflow.keras.applications.vgg16 import VGG16"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}