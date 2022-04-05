from importlib.resources import path
import numpy as np
import cv2
from sklearn.cluster import *
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from PIL import Image

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
      img = np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
      if img is not None:
        #reshape
        pxl = img.reshape((-1,3))
        #convert to float
        pxl = np.float32(pxl)
        #append image to the list
        imgs.append([pxl, personCount])
        #print(img.shape)
      else:
        personCount -= 1
        imgCount -= 1
    if (imgCount == 410):
      break
  return imgs


def main():
  #the below line of code defines the criteria for the algorithm to stop running,
  #which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
  #becomes 85%
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

  images = loadImages()

  #KMeans
  kmeans_labels = KMeans().fit(images).predict()

  #DBSCAN
  dbscan_labels = DBSCAN().fit_predict(images)

  #single link
  single_link_labels = AgglomerativeClustering(linkage = 'single').fit_predict(images)

  #single link
  complete_link_labels = AgglomerativeClustering(linkage = 'complete').fit_predict(images)  

  #single link
  group_average_link_labels = AgglomerativeClustering(linkage = 'average').fit_predict(images)

