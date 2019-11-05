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
import Model

import os
import glob
import random
import torch
from   torch.utils.data import Dataset
from   torchvision      import transforms
from   PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def LoadData(dataname):
	img = sorted(glob.glob(os.path.join(dataname, "*.png")))

	data = list(img)
	return data

class DataSet(Dataset):
	def __init__(self, data):
		self.data = data
		self.transform = transforms.Compose([
						 transforms.ToTensor(),
						 transforms.Normalize(MEAN, STD)
						 ])

	def __len__(self):
		return len(self.data)

	def __getitem__(self, ind):
		pth = self.data[ind]
		img = Image.open(pth)
		img = self.transform(img)

		img = img.type(torch.FloatTensor)
		return img, pth


if __name__=='__main__':
	gpu = torch.cuda.is_available()
	torch.cuda.empty_cache()
	
	# parsing the arguments
	args = Parsing.Args()
	dataname = args.d
	predname = args.p
	modlname = args.m
	mode = args.mode

	test = LoadData(dataname)

	test = DataSet(test)
	test = DataLoader(test, batch_size=1, shuffle=False)
	print ("[Done] Loading all data (testing)!")

	# define loss function and optimizer
	if mode == "baseline": model = Model.Baseline()
	if mode == "improved": model = Model.Improved()
	if gpu: model = model.cuda()

	checkpoint = torch.load(modlname)
	model.load_state_dict(checkpoint)
	print ("[Done] Initializing model and parameters!")

	# set to evaluation mode
	# with torch.no_grad():
	model.eval()

	for ind, (img, pth) in enumerate(test):

		# preprocess the image data
		if gpu: 
			img = img.cuda()
		out = model(img)
		
		# compute the mIOU value
		pred = torch.max(out, dim=1)[1].cuda()
		pred  = pred.cpu().numpy().squeeze().astype("float32")

		result = Image.fromarray((pred).astype(np.uint8))
		result.save(predname + pth[0][-8:])

		# mask = np.zeros((pred.shape)).astype("float32")
		# mask = np.where(pred == 0, mask, 0)
		# mask = np.where(pred == 1, mask, 100)
		# mask = np.where(pred == 2, mask, 120)
		# mask = np.where(pred == 3, mask, 140)
		# mask = np.where(pred == 4, mask, 160)
		# mask = np.where(pred == 5, mask, 180)
		# mask = np.where(pred == 6, mask, 200)
		# mask = np.where(pred == 7, mask, 230)
		# mask = np.where(pred == 8, mask, 255)

		# result = Image.fromarray((mask).astype(np.uint8))
		# result.save(predname + "_" + pth[0][-8:])