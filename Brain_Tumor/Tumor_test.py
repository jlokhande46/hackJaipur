import cv2
import numpy as np 
import pandas as pd
from PIL import Image
from keras.models import load_model
import imutils
from keras.applications.vgg16 import VGG16, preprocess_input


def crop_imgs(img, add_pixels_value=0):

    set_new = []

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)


    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

        
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

        
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    ADD_PIXELS = add_pixels_value
    new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
    set_new.append(new_img)

    return np.array(set_new)


def preprocess_imgs(img, img_size):

    set_new = []
    
    img = cv2.resize(img,dsize=img_size,interpolation=cv2.INTER_CUBIC)
    set_new.append(preprocess_input(img))
    return np.array(set_new)

model = load_model('Brain_model.h5')

image = 'NO.jpeg'
image = cv2.imread(image)

img = preprocess_imgs(image,(224,224))
print(img.shape)


preds = model.predict(img)

preds = preds.tolist()
if preds[0][0]>0.5:
    print("Yes")
else:
    print("NO")
