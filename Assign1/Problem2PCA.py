import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def mean(shp, img):
	res = np.mean(img, axis = 0)

	# plt.title("meanFace")
	# plt.imshow(res, cmap='gray')
	# plt.show()
	filename = "mean.png"
	cv2.imwrite(filename, res)
	print ("\n[Done] Calculating for mean face image!")

def eigen(shp, img, num):
	# create PCA model and get the eigenspace vector
	pca = PCA()
	mean = np.mean(img, axis = 0)
	diff = img - mean
	diff = np.reshape(diff, (240, shp[0] * shp[1]))
	modl = pca.fit(diff)

	# normalize the result and scale it between 0 and 255
	for k in range(num):
		res = modl.components_[k]
		res -= np.min(res)
		res /= np.max(res)
		res *= 255
		res = np.reshape(res, shp)

		# plt.title("meanFace")
		# plt.imshow(res, cmap='gray')
		# plt.show()
		filename = "eigen%d.png" % (k + 1)
		cv2.imwrite(filename, res)
		print ("[Done] Calculating for eigen face image %d!" % k)

def reconstruct(shp, imgOri, imgTar, lst):
	# create PCA model and get the eigenspace vector
	pca = PCA()
	mean = np.mean(imgOri, axis = 0)
	diff = imgOri - mean
	diff = np.reshape(diff, (240, shp[0] * shp[1]))
	modl = pca.fit(diff)
	
	# project the specific image on the eigenspace
	diff = imgTar - mean
	diff = np.reshape(diff, (1, shp[0] * shp[1]))
	tran = pca.transform(diff)
	
	for i in lst:
		res = (tran[:,:i] @ modl.components_[:i]) + mean.reshape(1,-1)
		res = np.reshape(res, shp)
		
		# plt.title("meanFace")
		# plt.imshow(res, cmap='gray')
		# plt.show()
		filename = "reconstruct_%d.png" % i
		cv2.imwrite(filename, res)
		print ("\n[Done] Reconstructing the image with %d images!" % i)
		errors(imgTar, res)

def errors(imgOri, imgTar):
	# compute the mean square error
	mse = np.mean((imgOri - imgTar)**2)
	print ("[Done] Computing the mean square error (MSE) which is %4.6f " % mse)

def k_nearest(shp, imgTrn, labTrn, imgTst, labTst, lst):
	# create PCA model, get the eigenspace vector
	pca = PCA()
	mean = np.mean(imgTrn, axis = 0)
	diff = imgTrn - mean
	diff = np.reshape(diff, (240, shp[0] * shp[1]))
	modl = pca.fit(diff)
	tran = pca.transform(diff)

	# apply k-neighbors classifier with k = 1, 3, 5
	knn = KNeighborsClassifier()
	par = {"n_neighbors": [1, 3, 5]}
	clf = GridSearchCV(knn, par, cv = 3)
	
	# find the best results from classifier with training images
	num = -1
	arg = -1
	bst = -1
	print("\n                   k = 1      k = 3      k = 5")
	for ind in lst:
		clf.fit(tran[:,:ind], labTrn)
		scr = clf.cv_results_["mean_test_score"]
		if bst < np.max(scr):
			bst = np.max(scr)
			arg = np.argmax(scr)
			num = ind
		print("       n = %3d" % ind, scr)
	print("[Done] Computing the training results by various k and n!")

	# project the test images on the eigenspace
	diff = imgTst - mean
	diff = np.reshape(diff, (160, shp[0] * shp[1]))
	test = pca.transform(diff)
	
	# apply k-neighbors classifier with hyperparameters after tuning
	k = arg * 2 + 1
	n = num
	knn = KNeighborsClassifier(n_neighbors = k)
	knn.fit(tran[:,:n], labTrn)
	
	# compute the final accuracy score
	res = knn.predict(test[:,:n])
	acc = accuracy_score(y_pred = res, y_true = labTst)
	print ("[Done] Predicting the testing results with accuracy %5f!" % acc)