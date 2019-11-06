import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import random

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def divide(imgTrn, labTrn, imgTst, labTst):
	# divide the training and testing images into 16 parts
	patTrn, divTrn = [], []
	patTst, divTst = [], []
	for k in range(1500):
		for tmpi in range(4):
			for tmpj in range(4):
				patTrn.append(imgTrn[k, tmpi*16:(tmpi+1)*16, tmpj*16:(tmpj+1)*16])
				divTrn.append(labTrn[k])

				if k > 499: continue
				patTst.append(imgTst[k, tmpi*16:(tmpi+1)*16, tmpj*16:(tmpj+1)*16])
				divTst.append(labTst[k])

	num = len(imgTrn) // 4
	ind = 32
	# ind = random.randint(0, num)
	
	for k in range(4):
		pat1 = patTrn[16*(num*k+ind) + 2]
		pat2 = patTrn[16*(num*k+ind) + 4]
		pat3 = patTrn[16*(num*k+ind) + 14]

		# plt.title("patch")
		# plt.imshow(patTrn[16*(num*k+ind) + 15])
		# plt.show()

		filename1 = "patch%d_1.png" % (num*k+ind)
		filename2 = "patch%d_2.png" % (num*k+ind)
		filename3 = "patch%d_3.png" % (num*k+ind)
		cv2.imwrite(filename1, pat1)
		cv2.imwrite(filename2, pat2)
		cv2.imwrite(filename3, pat3)

	print ("[Done] Dividing the data!")
	return patTrn, divTrn, patTst, divTst

def k_means(shp, pat, lab):
	# flatten the numpuy array of patches
	pat = np.reshape(pat, (24000, shp[0] * shp[1] // 16 * 3))

	# apply k-means algorithm
	kmn = KMeans(n_clusters = 15)
	ret = kmn.fit_predict(pat)
	cen = kmn.cluster_centers_

	# create PCA model, and project the patches on the eigenspace
	pca = PCA(n_components = 3)
	modl = pca.fit_transform(pat, lab)
	proj = pca.transform(cen)

	lst = [0, 3, 5, 9, 12, 14]
	tmpc = []
	tmpr = [[], [], [], [], [], []]
	
	# select 6 clusters randomly
	for k in range(6):
		# lst[k] = random.randint(0, 15)
		tmpc.append(proj[lst[k]])
		
		for i in range(24000):
			if ret[i] != lst[k]: continue
			tmpr[k].append(modl[i])

		tmpr[k] = np.array(tmpr[k])
	tmpc = np.array(tmpc)

	plt.figure(figsize = (10, 8))
	ax = plt.axes(projection='3d')

	# ax.scatter(tmpc[:,0], tmpc[:,1], tmpc[:,2], c = "black", marker = "x")	
	# ax.scatter(tmpr[0][:,0], tmpr[0][:,1], tmpr[0][:,2], c = "red")
	# ax.scatter(tmpr[1][:,0], tmpr[1][:,1], tmpr[1][:,2], c = "orange")
	# ax.scatter(tmpr[2][:,0], tmpr[2][:,1], tmpr[2][:,2], c = "yellow")
	# ax.scatter(tmpr[3][:,0], tmpr[3][:,1], tmpr[3][:,2], c = "green")
	# ax.scatter(tmpr[4][:,0], tmpr[4][:,1], tmpr[4][:,2], c = "blue")
	# ax.scatter(tmpr[5][:,0], tmpr[5][:,1], tmpr[5][:,2], c = "magenta")
	# plt.savefig("3dimension.png")
	# plt.show()

	# plt.title("tangentPlane I")
	# plt.scatter(tmpr[0][:,0], tmpr[0][:,1], c = "red")
	# plt.scatter(tmpr[1][:,0], tmpr[1][:,1], c = "orange")
	# plt.scatter(tmpr[2][:,0], tmpr[2][:,1], c = "yellow")
	# plt.scatter(tmpr[3][:,0], tmpr[3][:,1], c = "green")
	# plt.scatter(tmpr[4][:,0], tmpr[4][:,1], c = "blue")
	# plt.scatter(tmpr[5][:,0], tmpr[5][:,1], c = "magenta")
	# plt.scatter(tmpc[:,0], tmpc[:,1], c = "black", marker = "x")	
	# plt.savefig("tangentPlaneI.png")
	# plt.show()
	
	# plt.title("tangentPlane II")
	# plt.scatter(tmpr[0][:,1], tmpr[0][:,2], c = "red")
	# plt.scatter(tmpr[1][:,1], tmpr[1][:,2], c = "orange")
	# plt.scatter(tmpr[2][:,1], tmpr[2][:,2], c = "yellow")
	# plt.scatter(tmpr[3][:,1], tmpr[3][:,2], c = "green")
	# plt.scatter(tmpr[4][:,1], tmpr[4][:,2], c = "blue")
	# plt.scatter(tmpr[5][:,1], tmpr[5][:,2], c = "magenta")
	# plt.scatter(tmpc[:,1], tmpc[:,2], c = "black", marker = "x")	
	# plt.savefig("tangentPlaneII.png")
	# plt.show()

	# plt.title("tangentPlane III")
	# plt.scatter(tmpr[0][:,0], tmpr[0][:,2], c = "red")
	# plt.scatter(tmpr[1][:,0], tmpr[1][:,2], c = "orange")
	# plt.scatter(tmpr[2][:,0], tmpr[2][:,2], c = "yellow")
	# plt.scatter(tmpr[3][:,0], tmpr[3][:,2], c = "green")
	# plt.scatter(tmpr[4][:,0], tmpr[4][:,2], c = "blue")
	# plt.scatter(tmpr[5][:,0], tmpr[5][:,2], c = "magenta")
	# plt.scatter(tmpc[:,0], tmpc[:,2], c = "black", marker = "x")	
	# plt.savefig("tangentPlaneIII.png")
	plt.show()
	
	np.save("ret", ret)
	np.save("cen", cen)
	print ("[Done] Implementing k-means algorithm!")
	return cen, ret

def bagOfWords(shp, patTrn, labTrn, patTst, labTst, cen):
	# flatten the numpuy array of patches
	patTrn = np.reshape(patTrn, (24000, shp[0] * shp[1] // 16 * 3))
	patTst = np.reshape(patTst, ( 8000, shp[0] * shp[1] // 16 * 3))

	# create PCA model, and project the patches on the eigenspace
	# pca = PCA(n_components = 3)
	# modl = pca.fit_transform(patTrn, labTrn)
	# proj = pca.transform(cen)

	ind = 30
	# ind = random.randint(0, num)
	bow = np.zeros((1500, 15))
	now = np.zeros(( 500, 15))
	for k in range(1500):
		arrTrn = np.zeros((16, 15))
		arrTst = np.zeros((16, 15))

		## compute each patch feature vectors
		for i in range(16):
			tmpTrn = np.reshape(patTrn[(16*k+i) % 24000], (1, shp[0] * shp[1] // 16 * 3))
			tmpTst = np.reshape(patTst[(16*k+i) %  8000], (1, shp[0] * shp[1] // 16 * 3))

			for j in range(15):
				arrTrn[i][j] = np.linalg.norm(tmpTrn - cen[j])
				arrTst[i][j] = np.linalg.norm(tmpTst - cen[j])
				arrTrn[i][j] = np.reciprocal(arrTrn[i][j])
				arrTst[i][j] = np.reciprocal(arrTst[i][j])

		## normalize the patch feature vectors
		arrTrn[i] = arrTrn[i] / np.linalg.norm(arrTrn[i])
		arrTst[i] = arrTst[i] / np.linalg.norm(arrTst[i])
		bow[k % 1500] = np.max(arrTrn, axis = 0)
		now[k %  500] = np.max(arrTst, axis = 0)

		if k % 375 == ind:
			filename = "bowVisualize%d.png" % k
			plt.bar(np.arange(15), height = bow[k], color = "gray")
			plt.savefig(filename)
			# plt.show()

	np.save("bow", bow)
	np.save("now", now)
	print ("[Done] Computing bagOfWords of training images!")
	return bow, now

def k_nearest (shp, patTrn, labTrn, patTst, labTst, bow, now):
	# flatten the numpuy array of patches
	patTrn = np.reshape(patTrn, (24000, shp[0] * shp[1] // 16 * 3))
	patTst = np.reshape(patTst, ( 8000, shp[0] * shp[1] // 16 * 3))

	pca = PCA()
	modl = pca.fit_transform(bow, labTrn)
	tran = pca.transform(bow)
	test = pca.transform(now)

	# apply k-neighbors classifier
	knn = KNeighborsClassifier(n_neighbors = 5)
	knn.fit(tran, labTrn)

	res = knn.predict(test)
	acc = accuracy_score(y_pred = res, y_true = labTst)
	print (acc)