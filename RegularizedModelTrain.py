import numpy as np
from keras import layers
from keras import models
from keras.regularizers import l2

x_train = np.load("X_train.npy")
x_val = np.load("X_val.npy")
y_train = np.load("Y_train.npy")
y_val = np.load("Y_val.npy")

print(x_train.shape)
print(x_val.shape)
print(y_train.shape)


model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu',input_shape=(100, 100, 3), kernel_regularizer=l2(0.001)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu',kernel_regularizer=l2(0.001)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu',kernel_regularizer=l2(0.001)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu',kernel_regularizer=l2(0.001)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(rate=0.2))
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
                    epochs= 85,
                    steps_per_epoch=x_train.shape[0]//100,
                    validation_data=(x_val, y_val))

np.save("regularized_train_history.npy", history.history,allow_pickle=True)
model.save('regularized_model.keras')