from flask import render_template, jsonify, Flask, redirect, url_for, request
from app import app
import pickle5 as cPickle
import numpy as np 
import pandas as pd 
import cv2
from skimage import feature
from imutils import build_montages
from imutils import paths
import random
import os
from keras.applications.vgg16 import VGG16, preprocess_input
# import urllib2
# Just disables the warning, doesn't enable AVX/FMA
#import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.applications.resnet50 import ResNet50
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# UPLOAD_FOLDER = '/home/sachin/Desktop/AI_Startup_Prototype/flaskSaaS-master/app/static/img'
UPLOAD_FOLDER = '/workspace/HackNITR/flaskSaaS-master/app/static/img'
@app.route('/')


@app.route('/upload')
def upload_file2():
   return render_template('index.html')	
	
@app.route('/uploaded', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
      print("path:",path)
      f.save(path)
      print("2",os.getcwd())
      #changes start
      with open('parkinson.cpickle', 'rb') as f:
            model = cPickle.load(f)
      test_image = path
      image = cv2.imread(test_image)
      output = image.copy()
      output = cv2.resize(output, (128, 128))

      # pre-process the image in the same manner we did earlier
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      image = cv2.resize(image, (200, 200))
      image = cv2.threshold(image, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
      features = quantify_image(image)
      preds = model.predict([features]).tolist()
      print(type(preds[0]))
      print(preds)
      if preds[0]==1:
          likely="More chances to be suffering from Parkinsons"
      else:
          likely="Less chances to be suffering from Parkinsons"
      return render_template('uploaded.html',disease="Parkinsons", title='Success', predictions=likely ,path=os.path.split(path))

@app.route('/uploaded_al', methods = ['GET', 'POST'])
def upload_file1():
   if request.method == 'POST':
      f = request.files['file']
      path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
      print("path:",path)
      f.save(path)
      print("2",os.getcwd())
      
      model = load_model('alzheimer.hdf5',compile=False)

      image = path

      image = cv2.imread(image)
      image = cv2.resize(image,(176,176))
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      batch = np.expand_dims(image,axis=0)
      print(batch.shape) # (1, 28, 28)
      batch = np.expand_dims(batch,axis=3)
      print(batch.shape) # (1, 28, 28,1)
      preds = model.predict(batch)
      print(preds)
      l = ["Mild Demented","Moderate Demented","Non Demented","Very Mild Demented"]
      for i,j in enumerate(preds[0]):
          if(j==1):
              position=i
              break

      result=l[position]
      return render_template('uploaded.html',disease="Alzheimer's", title='Success', predictions=result ,path=os.path.split(path))

def crop_imgs(img, add_pixels_value=0):

      set_new = []

      gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
      gray = cv2.GaussianBlur(gray, (5, 5), 0)


      thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
      thresh = cv2.erode(thresh, None, iterations=2)
      thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
      cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)
      c = max(cnts, key=cv2.contourArea)

        # find the extreme points
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

@app.route('/uploaded_tu', methods = ['GET', 'POST'])

def upload_file3():
    if request.method == 'POST':
      f = request.files['file']
      path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
      print("path:",path)
      f.save(path)
      print("2",os.getcwd())
      
      model = load_model('Brain_model.h5')

      image = path
      image = cv2.imread(image)
      img = preprocess_imgs(image,(224,224))
      print(img.shape)


      preds = model.predict(img)
      preds = preds.tolist()
      if preds[0][0]>0.5:
        result="Tumor is present"
      else:
        result="Tumor is absent"


      return render_template('uploaded.html',disease="Tumor", title='Success', predictions=result ,path=os.path.split(path))


#quantify_image
def quantify_image(image):
	# compute the histogram of oriented gradients feature vector for
	# the input image
	features = feature.hog(image, orientations=9,
		pixels_per_cell=(10, 10), cells_per_block=(2, 2),
		transform_sqrt=True, block_norm="L1")

	# return the feature vector
	return features
@app.route('/index')
def index():
    return render_template('index.html', title='Home')

@app.route('/map')
def map():
    return render_template('map.html', title='Map')


@app.route('/map/refresh', methods=['POST'])
def map_refresh():
    points = [(random.uniform(48.8434100, 48.8634100),
               random.uniform(2.3388000, 2.3588000))
              for _ in range(random.randint(2, 9))]
    return jsonify({'points': points})


@app.route('/contact')
def contact():
    return render_template('contact.html', title='Contact')
