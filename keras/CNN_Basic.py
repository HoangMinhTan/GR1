from keras.models import Sequential
from keras.layers import  Dense, Conv2D, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import  to_categorical
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape


plt.imshow(X_train[0])