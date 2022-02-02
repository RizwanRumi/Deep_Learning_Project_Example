# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
##
# https://www.kaggle.com/uysimty/get-start-image-classification/notebook
# ##

import warnings


import matplotlib

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import os

print(os.listdir("../ImageClassificationForDigitRecognizer/dataset"))

FAST_RUN = False
batch_size = 32
epochs = 100
if FAST_RUN:
    epochs = 1

# Import data
train_data = pd.read_csv("../ImageClassificationForDigitRecognizer/dataset/train.csv")
test_data = pd.read_csv("../ImageClassificationForDigitRecognizer/dataset/test.csv")

# Data exploration
#print(train_data.columns)

# show image

def show_image(train_image, label, index):
    image_shaped = train_image.values.reshape(28, 28)
    plt.subplot(3, 6, index+1)
    plt.imshow(image_shaped, cmap=plt.cm.gray)
    plt.title(label)

plt.figure(figsize=(18, 8))
sample_image = train_data.sample(18).reset_index(drop=True)
for index, row in sample_image.iterrows():
    label = row['label']
    image_pixels = row.drop('label')
    show_image(image_pixels, label, index)
plt.tight_layout()
#plt.show()

# Data preparation

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

x = train_data.drop(columns=['label']).values.reshape(train_data.shape[0],28,28,1)
y = to_categorical(train_data['label'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

print('test')


