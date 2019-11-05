import os
import glob
import random
import torch
from   torch.utils.data import Dataset
from   torchvision      import transforms
from   PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def LoadData(dataname, train_or_valid):
	# 0 for train, 1 for valid
	if train_or_valid == 0: dataname += "train/"
	if train_or_valid == 1: dataname += "val/"

	img = sorted(glob.glob(os.path.join(dataname + "img/", "*.png")))
	lab = sorted(glob.glob(os.path.join(dataname + "seg/", "*.png")))

	data = list(zip(img, lab))
	random.shuffle(data)
	# if train_or_valid == 0: data = data[:128]
	# if train_or_valid == 1: data = data[:100]
	
	return data

class DataSet(Dataset):
	def __init__(self, data, train_or_valid):
		self.data = data

		if train_or_valid == 0:
			self.transform1 = transforms.Compose([
							  transforms.RandomHorizontalFlip(0.5),
							  transforms.ToTensor(),
							  ])

		if train_or_valid == 1:
			self.transform1 = transforms.Compose([
							  transforms.ToTensor(),
							  ])

		self.transform2 = transforms.Compose([
						  transforms.Normalize(MEAN, STD)
						  ])

	def __len__(self):
		return len(self.data)

	def __getitem__(self, ind):
		img = Image.open(self.data[ind][0])
		lab = Image.open(self.data[ind][1])

		img = self.transform1(img)
		lab = self.transform1(lab)
		img = self.transform2(img)
		lab = lab.view(lab.size(1), lab.size(2)) * 255

		img = img.type(torch.FloatTensor)
		lab = lab.type(torch.LongTensor)
		return img, lab
