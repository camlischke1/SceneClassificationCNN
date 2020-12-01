import requests
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model
import keras

# takes the url of the image as an input (must be an http url)
# returns the tensor form of the image with shape (1, 100, 100, 3) to be fed to the final model
# and the raw RGB image 
def process_google_image(url):
  response = requests.get(url)
  img = Image.open(BytesIO(response.content))
  img_tensor = image.img_to_array(img)
  img_tensor = image.smart_resize(img_tensor, (100, 100))
  img_tensor = np.array(img_tensor)
  img_tensor = img_tensor / 255.
  img_tensor = img_tensor.reshape((1,) + img_tensor.shape)
  return img_tensor, img

# example
#url = "https://cdn.vox-cdn.com/thumbor/Mt2SHO8KepxiyP-dIB4qruN_dNE=/0x0:1500x1001/1200x0/filters:focal(0x0:1500x1001):no_upscale()/cdn.vox-cdn.com/uploads/chorus_asset/file/10177719/20_Kensinger_325_327_Canal_Street_DSC_7353.jpg"
#url = "http://t0.gstatic.com/images?q=tbn:ANd9GcQaFn4K4e24DD4bE4-Ev6PfWgq4FcPs1zdLaKs9C6roiIgGqbwtsNZnNAyxGE3Ybmk-Pw9lisTy4G7B1lhKkaM"

#url = "https://www.thefinancialfreedomproject.com/wp-content/uploads/2018/08/climb-your-money-mountain-to-financial-freedom-1024x682.jpg"
#url = "https://www.nps.gov/grte/planyourvisit/images/drive_bonney.jpg?maxwidth=1200&maxheight=1200&autorotate=false"

#imageB
url = "https://cache.desktopnexus.com/thumbseg/2199/2199997-bigthumbnail.jpg"
#imageA
#url = "https://wallpaperstock.net/wallpapers/thumbs1/44189hd.jpg"

test_tensor, real_image = process_google_image(url)
print (test_tensor.shape)

model= load_model("regularized_model.keras")
dict = {0:"Buildings",1:"Forest",2:"Glacier",3:"Mountains",4:"Sea",5:"Street"}

prediction = model.predict(test_tensor)
print(prediction)
prediction = np.argmax(prediction,axis=1)

plt.imshow(real_image)
plt.title("Predicted as " + str(dict.get(prediction[0])))
plt.show()

