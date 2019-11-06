import numpy as np
import cv2
import Problem3BOW as Processing

if __name__ == "__main__":
	shp = (64, 64)
	
	label = ["banana", "fountain", "reef", "tractor"]
	imgTrn, labTrn = [], []
	imgTst, labTst = [], []
	for i in range(4):
		for j in range(500):
			## read the image file
			filename = "./p3_data/" + label[i] + "/" + label[i] + "_%03d.JPEG" % (j + 1)

			## organize the training data
			if j < 375:
				imgTrn.append(cv2.imread(filename, cv2.IMREAD_COLOR))
				labTrn.append(i + 1)
			
			## organize the testing data
			if j > 374:
				imgTst.append(cv2.imread(filename, cv2.IMREAD_COLOR))
				labTst.append(i + 1)

	imgTrn, labTrn = np.array(imgTrn), np.array(labTrn)
	imgTst, labTst = np.array(imgTst), np.array(labTst)
	
	patTrn, divTrn, patTst, divTst = Processing.divide(imgTrn, labTrn, imgTst, labTst)
	
	cen, ret = Processing.k_means   (shp, patTrn, divTrn)
	# ret = np.load("ret.npy")
	# cen = np.load("cen.npy")

	bow, now = Processing.bagOfWords(shp, patTrn, divTrn, patTst, divTst, cen)
	# bow = np.load("bow.npy")
	# now = np.load("now.npy")
	
	Processing.k_nearest(shp, imgTrn, labTrn, patTst, labTst, bow, now)
