import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

import matplotlib.pyplot as plt

import Parsing
import Loading
import Model

def mean_iou(predt, label):
	res = 0.0
	for k in range(9):
		tp_fp = np.sum(predt == k)
		tp_fn = np.sum(label == k)
		tp    = np.sum((predt == k) * (label == k))
		
		iou = tp / (tp_fp + tp_fn - tp)
		res += iou
	return res / 9

if __name__=='__main__':
	gpu = torch.cuda.is_available()
	torch.cuda.empty_cache()
	
	# parsing the arguments
	args = Parsing.Args()
	dataname = args.d
	warnname = args.w
	modlname = args.m
	mode = args.mode

	train = Loading.LoadData(dataname, 0)
	valid = Loading.LoadData(dataname, 1)

	train = Loading.DataSet(train, 0)
	valid = Loading.DataSet(valid, 1)
	train = DataLoader(train, batch_size=32, shuffle=True)
	valid = DataLoader(valid, batch_size=32, shuffle=True)
	print ("[Done] Loading all data (training and validation)!")

	# define loss function and optimizer
	if mode == "baseline": model = Model.Baseline()
	if mode == "improved": model = Model.Improved()
	if gpu: model = model.cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
	print ("[Done] Initializing model and parameters!")

	pltx = []
	trny = []
	valy = []
	best = -1.0
	for epoch in range(40):
		print ("\n###### Epoch: {:2d}".format(epoch + 1))
		f = open(warnname, "a")
		f.write("\n###### Epoch: {:2d}\n".format(epoch + 1))
		f.close()

		# set to training mode
		model.train()

		train_loss = []
		for ind, (img, lab) in enumerate(train):
			optimizer.zero_grad()
			
			# preprocess the image data
			if gpu: 
				img = img.cuda()
				lab = lab.cuda()
			out = model(img)
			
			# compute the loss value
			loss = criterion(out, lab)
			loss.backward()
			train_loss.append(loss.item())

			optimizer.step()

		print("[Done] Computing training loss:   {:.4f}".format(np.mean(train_loss)))
		f = open(warnname, "a")
		f.write("[Done] Computing training loss:   {:.4f}\n".format(np.mean(train_loss)))
		f.close()

		# set to evaluation mode
		# with torch.no_grad():
		model.eval()

		valid_scre = 0.0
		predt = np.zeros((1, 352, 448)).astype("float32")
		label = np.zeros((1, 352, 448)).astype("float32")
		for ind, (img, lab) in enumerate(valid):

			# preprocess the image data
			if gpu: 
				img = img.cuda()
				lab = lab.cuda()
			out = model(img)
			
			# compute the mIOU value
			pred = torch.max(out, dim=1)[1].cuda()			
			pred  = pred.cpu().numpy().squeeze().astype("float32")
			labl  = lab .cpu().numpy().squeeze().astype("float32")
			predt = np.concatenate((predt, pred), axis=0)
			label = np.concatenate((label, labl), axis=0)

		predt = predt[1:]
		label = label[1:]
		valid_scre = mean_iou(predt, label)
		print("[Done] Computing validation mIOU: {:.4f}".format(valid_scre))
		f = open(warnname, "a")
		f.write("[Done] Computing validation mIOU: {:.4f}\n".format(valid_scre))
		f.close()

		# update the best model
		if best < np.mean(valid_scre):
			best = np.mean(valid_scre)
			torch.save(model.state_dict(), os.path.join("./", modlname))


		# plot the graph
		pltx.append(epoch)
		trny.append(np.mean(train_loss))
		valy.append(np.mean(valid_scre))

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
