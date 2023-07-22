#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models,layers
import numpy as np
import tensorflow as tf
import math
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import  BatchNormalization
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.offsetbox as offsetbox
from PIL import Image
import os


# In[2]:


df = tf.keras.preprocessing.image_dataset_from_directory(r"C:\Users\soura\OneDrive\Desktop\check2",
                                                        shuffle=True ,
                                                        image_size = (256,256),
                                                        batch_size = 32)


# In[3]:


classes =df.class_names
classes


# In[4]:


train_size = 0.7
len(df)*train_size


# In[8]:


train_ds = df.take(44)
len(train_ds)


# In[10]:


test_ds = df.skip(44)
len(test_ds)


# In[11]:


val_ds = test_ds.take(19)
len(val_ds)


# In[12]:


train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[13]:


resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(256,256),
  layers.experimental.preprocessing.Rescaling(1./255),
])


# In[14]:


@tf.function
def preprocess_data(x, y):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    x = tf.image.rot90(x, k=k)
    x = tf.image.random_brightness(x, max_delta=0.1)
    x = tf.image.random_contrast(x, lower=0.8, upper=1.2)
    x = tf.image.random_hue(x, max_delta=0.1)
    x = tf.image.per_image_standardization(x)

    return x, y
batch_size = 32


preprocessed_train_ds = train_ds.map(preprocess_data).prefetch(buffer_size=tf.data.AUTOTUNE)
preprocessed_val_ds = val_ds.map(preprocess_data).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[15]:


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))


# In[16]:


initial_learning_rate = 0.001
decay_rate = 0.1
decay_steps = 10

num_classes = 2
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x) 
x = Dense(512, activation='relu')(x) 
x = Dropout(0.5)(x) 
x = Dense(256, activation='relu')(x) 
x = Dropout(0.5)(x)  
x = Dense(128, activation='relu')(x)  
x = Dropout(0.5)(x)  
x = Dense(64, activation='relu')(x)  
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False


# In[17]:


model.summary()


# In[18]:


def exponential_decay(epoch, initial_lr=0.001, decay_rate=0.1, decay_steps=10):
    return initial_lr * math.pow(decay_rate, epoch / decay_steps)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay)


# In[19]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Set the initial learning rate as you desire
             loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[20]:


BATCH_SIZE = 32
history = model.fit(
    preprocessed_train_ds,
    batch_size=BATCH_SIZE,
    validation_data=preprocessed_val_ds,
    epochs=20,
    callbacks=[lr_scheduler],
    verbose=1
)


# In[ ]:




