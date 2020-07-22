import argparse
from keras.models import load_model
from create_model import CRNN
from config import config
from pyimagesearch.io import HDF5DatasetGenerator
from keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
import numpy as np
import cv2
import itertools
import os
import imutils

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def fastdecode(y_pred, chars):
    results_str = ""
    confidence = 0.0

    for i,one in enumerate(y_pred[0]):

        if one<config.NUM_CLASSES and (i==0 or (one!=y_pred[0][i-1])):
            results_str+= chars[one]

    return results_str
def decode_label(label, chars):
	results_str = ""
	for i, num in enumerate(label[0]):
		if num != config.NUM_CLASSES:
			results_str += chars[num]
	return results_str 

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model",
	help = "path to pre-trained model")
ap.add_argument("-i", "--images", 
	help = "path to image folder ")
args = vars(ap.parse_args())

print('loading model...')
model = CRNN.build(width=config.WIDTH, height=config.HEIGHT, depth=1,
		classes=config.NUM_CLASSES, training=0)
model.load_weights(args["model"])
iap = ImageToArrayPreprocessor()
dic = {}
dic[0] = ' '
with open('dic.txt', encoding="utf-8") as dict_file:
	for i, line in enumerate(dict_file):
		if i == 0:
			continue
		(key, value) = line.strip().split('\t')
		dic[int(key)] = value
dict_file.close()

acc = 0
total = 0
paths = os.listdir(args["images"])


width = config.HEIGHT
height = config.WIDTH
k1 = width/height

print('predicting...')
for i in range(0, 50, 1):
    mark = False
    string = ''
    for j in range(0, 10):
        name = 'test_image/' + str(i) + '_' + str(j) + '.jpg'
        if os.path.exists(name):
            mark = True

            image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)

            k2 = image.shape[1]/image.shape[0]
            if k2 > k1:		
                resized = imutils.resize(image, width = width)
                zeros = np.zeros((height - resized.shape[0], width))
                results = np.concatenate((resized, zeros), axis=0)
            else:
                resized = imutils.resize(image, height = height)
                zeros = np.zeros((height, width - resized.shape[1]))
                results = np.concatenate((resized, zeros), axis=1)

            image = imutils.rotate_bound(results, 90)
            image = image/255.0
            image = [image]
            image = iap.preprocess(image)  


            predict = model.predict([[image]])	
            predict = np.argmax(predict, axis=2)
            print(len(predict))		

            res = fastdecode(predict, dic)
            string = string + res + ' '
    if mark:
        print(string)


