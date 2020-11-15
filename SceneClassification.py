#!/usr/bin/env python
# coding: utf-8

# In[4]:

import keras
keras.__version__


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
from sklearn import preprocessing
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.utils import shuffle
from keras.applications.imagenet_utils import preprocess_input


# In[ ]:


import os, shutil
import pandas as pd


# In[4]:


path = os.getcwd()
print(path)


# In[5]:


base_dir = path

# Directories for our training pictures
train_dir = os.path.join(base_dir, 'train')

# Labels
train_labels = pd.read_csv(base_dir + "/train.csv")

# Test Images List
test_images_list = pd.read_csv(base_dir + "/test.csv")


# In[6]:


image_count = len(os.listdir(train_dir))
print(image_count)
samples = len(train_labels)
print (samples)
test_samples = len(test_images_list)
print (len(train_labels["image_name"]))
print (len(train_labels["label"]))


# In[7]:


def process_images(train_labels, image_dir, samples, is_labelled = True):
    X_train = np.zeros((samples, 100, 100, 3))
    img_list = train_labels["image_name"]
    labels = train_labels["label"]
    
    idx = 0
    Y_train = [None for i in range(samples)]
    
    for i in range(len(img_list)):
        img_no = img_list[i]
        label = labels[i] if is_labelled else -1
        img = image.load_img(image_dir + "/" + img_no, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = x.reshape((1,) + x.shape)
        X_train[idx] = x
        Y_train[idx] = label
        idx += 1 
        if (idx % 500 == 0):
            print ("Processing {} Label {}".format(img_no, label))
    
    Y_train = to_categorical(Y_train)
    return X_train, Y_train


# In[8]:


X_train, Y_train = process_images(train_labels, train_dir, samples)


# In[9]:


X_train = X_train / 255.


# In[10]:


plt.imshow(X_train[1])
plt.show()
print ("Label", np.argmax(Y_train[1], axis=0))


# In[11]:


def process_test_images(test_images_list, image_dir, samples):
    X_train = np.zeros((samples, 100, 100, 3))
    idx = 0
    for img_no in test_images_list["image_name"]:
        img = image.load_img(image_dir + "/" + img_no, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = x.reshape((1,) + x.shape)
        X_train[idx] = x
        idx += 1 
        if (idx % 500 == 0):
            print ("Processing {}".format(img_no))
    
    return X_train


# In[12]:


X_test = process_test_images(test_images_list, train_dir, test_samples)


# In[13]:


X_test = X_test / 255.


# In[14]:


print (X_train.shape)
print (Y_train.shape)

print (X_test.shape)


# In[15]:


from collections import Counter
# Checking label distribution
_labels = np.argmax(Y_train, axis = 1)
counter = Counter(_labels)
plt.bar(list(counter.keys()), list(counter.values()))
plt.show()


# In[16]:


# Shuffling
X_train, Y_train = shuffle(X_train, Y_train)


# In[17]:


X_val = X_train[13035:]
Y_val = Y_train[13035:]

X_train = X_train[:13035]
Y_train = Y_train[:13035]


# In[18]:


print ("X train shape", X_train.shape)
print ("Y train shape", Y_train.shape)

print ("X val shape", X_val.shape)
print ("Y val shape", Y_val.shape)


# In[19]:


# Checking distribution in the validation set
from collections import Counter
# Checking label distribution
_labels = np.argmax(Y_val, axis = 1)
counter = Counter(_labels)
plt.bar(list(counter.keys()), list(counter.values()))
plt.show()


np.save("X_train.npy",X_train,allow_pickle=True)
np.save("X_val.npy",X_val,allow_pickle=True)
np.save("X_test.npy",X_test,allow_pickle=True)
np.save("Y_train.npy",Y_train,allow_pickle=True)
np.save("Y_val.npy",Y_val,allow_pickle=True)

'''
# ## Creating a Naive Model as a Baseline

# In[20]:


from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(100, 100, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='softmax'))
model.add(layers.Dense(Y_train.shape[1], activation='softmax'))


# In[21]:


from keras import optimizers

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', 'categorical_accuracy'])


# In[22]:


model.summary()


# In[23]:


history = model.fit(X_train, 
                    Y_train, 
                    batch_size=100, 
                    epochs= 10,
                    steps_per_epoch=100,
                    validation_data=(X_val, Y_val))


# In[24]:


import pickle 
with open('naive_model_train_history.pickle', 'wb') as pickle_file:
    pickle.dump(history.history, pickle_file)


# In[25]:


model.save('naive_model.h5')
'''
