#!/usr/bin/env python
# coding: utf-8

# In[11]:


pip install tensorflow


# In[9]:


import os
import shutil
import random
import itertools
import matplotlib.pyplot as plt
plt.show()
import numpy as np
import tensorflow as tf
import matplotlib as mpl
from keras import backend
from tensorflow import keras
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Activation
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import decode_predictions, preprocess_input


data_dir = "D:/2023sumdpl302m/devset_images"


labels = ['Flooding', 'No Flooding']
train_path = 'D:/book/DPL302m/2023falldpl30xm/devset_images'
#valid_path = 'D:/2023sumdpl302m/validclean'
test_path = 'D:/book/DPL302m/2023falldpl30xm/testset_images'


train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224,224), batch_size=10)
#valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    #directory=valid_path, target_size=(224,224), batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)

mobile = tf.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False)
mobile.summary()


x = mobile.layers[-12].output
x

# Create global pooling, dropout and a binary output layer, as we want our model to be a binary classifier, 
# i.e. to classify flooding and no flooding
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
output = Dense(units=2, activation='sigmoid')(x)
# Construct the new fine-tuned mode
model = Model(inputs=mobile.input, outputs=output)
# Freez weights of all the layers except for the last five layers in our new model, 
# meaning that only the last 12 layers of the model will be trained.
for layer in model.layers[:-23]:
    layer.trainable = False
model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


num_epochs = 10
for epoch in range(num_epochs):
    print("Epoch:", epoch+1)
    for i, (x, y) in enumerate(train_batches):
        try:
            model.train_on_batch(x, y)
        except Exception as e:
            print("Error loading image batch at index:", i)
            print("Error message:", str(e))
        if i == len(train_batches)-1:
            break


# In[ ]:





# In[ ]:





# In[ ]:




