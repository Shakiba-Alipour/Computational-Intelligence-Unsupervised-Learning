import numpy as np
import cv2
import sklearn.cluster
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt

imgs = [410][2]

def loadImages():
  imgCount = 0
  personCount = 0
  while (True):
    personCount += 1
    for i in range(1,10):
      #read image and convert it to rgb
      img = cv2.imread('ORL/' + imgCount + '_' + personCount + '.jpg')
      img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      if img is not None:
        imgCount += 1
        #reshape the image to a 2D array
        img = img.reshape((-1,3))
        #convert to float
        img = np.float32(img)
        #append image to the list
        imgs[imgCount][0] = img
        imgs[imgCount][1] = personCount
    if (imgCount == 410):
      break
