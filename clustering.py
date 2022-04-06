from enum import unique
import numpy as np
import cv2
from sklearn.cluster import *
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


def showPlot(dataset, labels):
	clusters = unique(labels)

	for cluster in clusters:
		# get row indexes for samples with this cluster
		row_ix = np.where(labels == cluster)
		# create scatter of these samples
		plt.scatter(dataset[row_ix, 0], dataset[row_ix, 1])

	# show the plot
	plt.show()

def main():

	images = loadImages()

	#KMeans
	kmeans_labels = KMeans().fit_predict(images)
	showPlot(images, kmeans_labels)

	#DBSCAN
	dbscan_labels = DBSCAN().fit_predict(images)
	showPlot(images, dbscan_labels)

	#single link
	single_link_labels = AgglomerativeClustering(linkage = 'single').fit_predict(images)
	showPlot(images, single_link_labels)

	#single link
	complete_link_labels = AgglomerativeClustering(linkage = 'complete').fit_predict(images)
	showPlot(images, complete_link_labels)

	#single link
	group_average_link_labels = AgglomerativeClustering(linkage = 'average').fit_predict(images)
	showPlot(images, group_average_link_labels)

