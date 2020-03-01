# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 14:55:06 2019

@author: HP
"""

import tensorflow as tf
mnist= tf.keras.datasets.mnist # mnist is a datasset of 28x28 images of handwritten digits and their labels
(x_train, y_train), (x_test, y_test)= mnist.load_data()#unpacks images to x_train/x_test and labels tp y_tain
x_train= tf.keras.utils.normalize(x_train, axis=1)#scales data bw 0 and 1
x_test=  tf.keras.utils.normalize(x_test, axis=1)#scales data bw 0 and 1

model= tf.keras.models.Sequential() # a basic feed forward model
model.add(tf.keras.layers.Flatten())#takes our 28x28 and makes it 1x784
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))#a simple fully-connected layer, 128 units,relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))#a simple fully-connected layer, 128 units,relu activation
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))#our output layer. 10 units for 10 classes


model.compile(optimizer='adam',#good default optimizer to start with
              loss='sparse_categorical_crossentropy', #how will we calculate our error, Neural network aims to minimise the error
              metrics=['accuracy'])#what to track
model.fit(x_train,y_train,epochs=3)#train the model
val_loss,val_acc=model.evaluate(x_test,y_test)#evaluate the output of sample data with model
print(val_loss)#model's loss(error)
print(val_acc)#model's accuracy
#model.save('epic_num_reader.model')

#new_model=tf.keras.models.load_model('epic_num_reader.model')
new_model=model
predictions= new_model.predict(x_test)
print(predictions)
import numpy as np
import matplotlib.pyplot as plt
print(np.argmax(predictions[1]))

#there's your prediction, let's look at the input 
plt.imshow(x_test[1],cmap=plt.cm.binary)
plt.show()
