import math
import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
	img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
	img = np.array(img)

	# gaussian filter with sigma = 2 ln 2
	new = np.zeros(img.shape)
	
	sig = 2 * math.log(2)
	msk = [[2, 1, 2], [1, 0, 1], [2, 1, 2]]
	msk = np.array(msk)
	msk = np.exp(msk * -1 / sig**2)
	msk = msk / msk.sum()

	new = signal.convolve2d(img, msk, boundary='symm', mode='same')
	cv2.imwrite("img.png", img)
	cv2.imwrite("new.png", new)

	# derivative with [-1, 0, 1]
	derx1 = np.zeros(img.shape)
	dery1 = np.zeros(img.shape)
	derx2 = np.zeros(new.shape)
	dery2 = np.zeros(new.shape)

	msk = [-1, 0, 1]
	msk = np.array(msk)

	for i in range(1, img.shape[0] - 1):
		for j in range(1, img.shape[1] - 1):
			orix = img[i - 1: i + 2, j].reshape(1, 3) * msk
			oriy = img[i, j - 1: j + 2].reshape(1, 3) * msk
			tmpx = new[i - 1: i + 2, j].reshape(1, 3) * msk
			tmpy = new[i, j - 1: j + 2].reshape(1, 3) * msk

			derx1[i][j] = orix.sum()
			dery1[i][j] = oriy.sum()
			derx2[i][j] = tmpx.sum()
			dery2[i][j] = tmpy.sum()

	cv2.imwrite("derx.png", derx1)
	cv2.imwrite("dery.png", dery1)

	# compute gradient magnitude
	grad1 = np.zeros(img.shape)
	grad2 = np.zeros(new.shape)

	grad1 = np.sqrt(derx1**2 + dery1**2)
	grad2 = np.sqrt(derx2**2 + dery2**2)

	cv2.imwrite("grad1.png", grad1)
	cv2.imwrite("grad2.png", grad2)
