import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist

(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_Y.shape))
print('X_test:  ' + str(test_X.shape))
print('Y_test:  ' + str(test_Y.shape))

# for i in range(9):
#     pyplot.subplot(330 + 1 + i)
#     pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
#     pyplot.show()

# normalization

train_X = train_X / 255
test_X = test_X / 255

train_Y_cat = keras.utils.to_categorical(train_Y, 10)
test_Y_cat = keras.utils.to_categorical(test_Y, 10)

plt.figure(figsize=(10, 5))

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_X[i], cmap=plt.cm.binary)

plt.show()

model = keras.Sequential([
 tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
 tf.keras.layers.Dense(128, activation='relu'),
 tf.keras.layers.Dense(10, activation='softmax'),
])
