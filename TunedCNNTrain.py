import keras
from keras import layers, models,optimizers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal

x_train = np.load("X_train.npy")
x_val = np.load("X_val.npy")
y_train = np.load("Y_train.npy")
y_val = np.load("Y_val.npy")

print(x_train.shape)
print(x_val.shape)
print(y_train.shape)

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu',input_shape=(100, 100, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='softmax'))
model.add(layers.Dense(y_train.shape[1], activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', 'categorical_accuracy'])
model.summary()
history = model.fit(x_train,
                    y_train,
                    batch_size=100,
                    epochs= 50,
                    steps_per_epoch=x_train.shape[0]//100,
                    validation_data=(x_val, y_val))

np.save("tuned_model_train_history.npy", history.history,allow_pickle=True)
model.save('tuned_model.keras')