import cv2
import numpy as np 
import pandas as pd
from PIL import Image
from keras.models import load_model
import imutils


model = load_model('alzheimer.hdf5')

image = 'nonDem10.jpg'

image = cv2.imread(image)
image = cv2.resize(image,(176,176))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

batch = np.expand_dims(image,axis=0)
print(batch.shape) # (1, 28, 28)
batch = np.expand_dims(batch,axis=3)
print(batch.shape) # (1, 28, 28,1)
preds = model.predict(batch)

print(preds)