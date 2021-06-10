import matplotlib.pyplot as plt
from keras.utils import to_categorical
import numpy as np
from keras.datasets import cifar10

(train_X, train_Y), (test_X, test_Y) = cifar10.load_data()

print('Training data shape : ', train_X.shape, train_Y.shape)

print('Testing data shape : ', test_X.shape, test_Y.shape)

classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)
