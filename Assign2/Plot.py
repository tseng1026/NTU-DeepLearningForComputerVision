import os
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
	f = open("warning.txt", "r")

	k = 0
	pltx = []
	trny = []
	valy = []
	cont = f.readlines()
	for k in range(20):
		pltx.append(int(cont[k * 4 + 1][14:]))
		trny.append(float(cont[k * 4 + 2][34:]))
		valy.append(float(cont[k * 4 + 3][34:]))


	plt.figure()
	plt.xticks(np.arange(0, 25, 5))
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.plot(pltx, trny, color="red")
	plt.savefig("plt1.png")

	plt.figure()
	plt.xticks(np.arange(0, 25, 5))
	plt.xlabel("Epoch")
	plt.ylabel("mIOU")
	plt.plot(pltx, valy, color="blue")
	plt.savefig("plt2.png")