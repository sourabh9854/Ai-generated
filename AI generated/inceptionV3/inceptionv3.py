#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import models,layers
from tensorflow.keras.regularizers import l2
import math


# In[2]:


df = tf.keras.preprocessing.image_dataset_from_directory(r"C:\Users\soura\OneDrive\Desktop\check1",
                                                        shuffle=True ,
                                                        image_size = (256,256),
                                                        batch_size = 32)


# In[3]:


train_size = 0.7
len(df)*train_size


# In[4]:


train_ds = df.take(219)
len(train_ds)


# In[5]:


test_ds = df.skip(219)
len(test_ds)


# In[6]:


val_ds = test_ds.take(94)
len(val_ds)


# In[7]:


train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[8]:


resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(224,224),
  layers.experimental.preprocessing.Rescaling(1./255),
])


# In[9]:


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


# In[10]:


num_classes = 2
input_shape = (256,256, 3)
initial_learning_rate = 0.001
decay_rate = 0.1
decay_steps = 10


# In[11]:


base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False


# In[12]:


def exponential_decay(epoch, initial_lr=0.001, decay_rate=0.1, decay_steps=10):
    return initial_lr * math.pow(decay_rate, epoch / decay_steps)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay)


# In[13]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Set the initial learning rate as you desire
             loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[14]:


BATCH_SIZE = 32
history = model.fit(
    preprocessed_train_ds,
    batch_size=BATCH_SIZE,
    validation_data=preprocessed_val_ds,
    epochs=10,
    callbacks=[lr_scheduler],
    verbose=1
)


# In[16]:


import matplotlib.pyplot as plt
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

final_train_acc = train_acc[-1]
final_val_acc = val_acc[-1]

epochs = range(1, len(train_acc) + 1)
plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.text(epochs[-1], final_train_acc, f'Train Acc: {final_train_acc:.4f}', ha='right', va='center', color='blue')
plt.text(epochs[-1], final_val_acc, f'Val Acc: {final_val_acc:.4f}', ha='right', va='center', color='red')

plt.show()


# In[ ]:




