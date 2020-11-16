import keras
from keras import layers, models,optimizers
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal




X_test = np.load("X_test.npy")
model = load_model("tuned_model.keras")

#plt.imshow(X_test[1])
#plt.show()
prediction = model.predict(X_test)
prediction = np.argmax(prediction,axis=1)

width=10
height=10
rows = 3
cols = 3
axes=[]
fig=plt.figure()

for a in range(rows*cols):
    b = X_test[a]
    axes.append( fig.add_subplot(rows, cols, a+1) )
    subplot_title=("Predicted"+ str(prediction[a]))
    axes[-1].set_title(subplot_title)
    plt.imshow(b)
fig.tight_layout()
plt.show()