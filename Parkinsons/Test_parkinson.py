import _pickle as cPickle
import numpy as np 
import pandas as pd 
import cv2
from skimage import feature
from imutils import build_montages
from imutils import paths

def quantify_image(image):
	# compute the histogram of oriented gradients feature vector for
	# the input image
	features = feature.hog(image, orientations=9,
		pixels_per_cell=(10, 10), cells_per_block=(2, 2),
		transform_sqrt=True, block_norm="L1")

	# return the feature vector
	return features
with open('Trained_model/parkinson.cpickle', 'rb') as f:
    model = cPickle.load(f)
test_image = "dataset/spiral/testing/healthy/V55HE15.png"

image = cv2.imread(test_image)
output = image.copy()
output = cv2.resize(output, (128, 128))

	# pre-process the image in the same manner we did earlier
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (200, 200))
image = cv2.threshold(image, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
features = quantify_image(image)
preds = model.predict([features])

print(preds)