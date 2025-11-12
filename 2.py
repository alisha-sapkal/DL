#importing the packages
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import random

#Load training and testing dataset
mnist = tf.keras.datasets.mnist #import mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data() #spliting
x_train = x_train / 255 #data was in range 0-255, we wanted in 0-1
x_test = x_test / 255

#Define the network architechture using keras
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.summary()


#Train the model using SGD - stochastic gradient decent
model.compile(optimizer="sgd",
              loss = "sparse_categorical_crossentropy",
              metrics=['accuracy'])
history=model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)


#Evaluate the network
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Loss=%.3f" %test_loss)
print("Accuracy=%.3f" %test_acc)


n = random.randint(0, 9999)
plt.imshow(x_test[n])
plt.show()
predicted_value = model.predict(x_test)
plt.imshow(x_test[n])
plt.show()
print("Predicted Value:",predicted_value[n])


#Plotting the training oss and accuracy
#acuuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()