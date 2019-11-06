import numpy as np
import cv2
import Problem2PCA as Processing

if __name__ == "__main__":
	shp = (56, 46)

	imgTrn, labTrn = [], []
	imgTst, labTst = [], []
	for i in range(40):
		for j in range(10):
			## read the image file
			filename = "./p2_data/" + str(i+1) + "_%d.png" % (j + 1)

			## organize the training data
			if j < 6:
				imgTrn.append(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))
				labTrn.append(i + 1)
			
			## organize the testing data
			if j > 5:
				imgTst.append(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))
				labTst.append(i + 1)

	imgTrn, labTrn = np.array(imgTrn), np.array(labTrn)
	imgTst, labTst = np.array(imgTst), np.array(labTst)
	
	Processing.mean       (shp, imgTrn)
	Processing.eigen      (shp, imgTrn, 4)
	Processing.reconstruct(shp, imgTrn, imgTrn[0], [3, 45, 140, 229])
	Processing.k_nearest  (shp, imgTrn, labTrn, imgTst, labTst, [3, 45, 140])
