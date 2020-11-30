import keras
from keras import layers, models,optimizers
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal




X_test = np.load("X_test.npy")
model = load_model("scene_classification_transfer_learning_1.keras")

dict = {0:"Buildings",1:"Forest",2:"Glacier",3:"Mountains",4:"Sea",5:"Street"}

width=10
height=10
rows = 3
cols = 3
off =40
axes=[]
fig=plt.figure()
prediction = model.predict(X_test[off:off+rows*cols])
prediction = np.argmax(prediction,axis=1)



for a in range(off,off+rows*cols):
    b = X_test[a]
    axes.append( fig.add_subplot(rows, cols, a+1-off) )
    subplot_title=(str(dict.get(prediction[a-off])))
    axes[-1].set_title(subplot_title)
    plt.axis('off')
    plt.imshow(b)
fig.tight_layout()
plt.show()