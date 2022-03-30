from importlib.resources import path
import numpy as np
import cv2
import sklearn.cluster
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt


def loadImages():
  imgCount = 0
  personCount = 0
  imgs = list()
  while (True):
    personCount += 1
    for i in range(1, 11):
      imgCount += 1
      #read image and convert it to rgb
      path = 'ORL\\' + str(imgCount) + '_' + str(personCount) + '.jpg'
      img = cv2.imread(path)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      if img is not None:
        #append image to the list
        imgs.append([img, personCount])
        print(img.shape)
      else:
        personCount -= 1
        imgCount -= 1
    if (imgCount == 410):
      break
  return imgs


imgs = loadImages()