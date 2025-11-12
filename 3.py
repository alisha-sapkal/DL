#importing packages
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np


# loading and pre-processing the image data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() #splitting training and testing data
input_shape = (28, 28, 1)


#making sure that the values are float so that we can get decimal points after division
x_train = x_train.reshape(x_train.shape[0], 28,28, 1)
x_test = x_test.reshape(x_test.shape[0], 28,28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


#normalizing the RGB codes by dividing it to the max RGB value
x_train = x_train/255
x_test = x_test/255
print("Shape of Training: ", x_train.shape)
print("Shape of Testing: ", x_test.shape)


#Defining the model arch
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))
model.summary()


#training the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)


#estimating the model performance
test_loss, test_acc=model.evaluate(x_test, y_test)
print("Test Loss: ", test_loss)
print("Test Accuracy: ", test_acc)


#showing image at position[] from dataset:
image = x_train[0]
plt.imshow(np.squeeze(image), cmap='gray')
plt.show()


#predicting the class of image:
image = image.reshape(1, image.shape[0], image.shape[1],  image.shape[2])
predict_model = model.predict([image])
print("Predicted class: {}".format(np.argmax(predict_model)))