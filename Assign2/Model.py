import torch.nn as nn
import torchvision
import torchvision.models as models

class Baseline(nn.Module):
	def __init__(self):
		super(Baseline, self).__init__()
		self.resnet18 = models.resnet18(pretrained=True)
		self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-2])

		self.conv1 = nn.Sequential(
					 nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
					 nn.ReLU()
		)

		self.conv2 = nn.Sequential(
					 nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
					 nn.ReLU()
		)

		self.conv3 = nn.Sequential(
					 nn.ConvTranspose2d(128,  64, kernel_size=4, stride=2, padding=1, bias=False),
					 nn.ReLU()
		)

		self.conv4 = nn.Sequential(
					 nn.ConvTranspose2d( 64,  32, kernel_size=4, stride=2, padding=1, bias=False),
					 nn.ReLU()
		)

		self.conv5 = nn.Sequential(
					 nn.ConvTranspose2d( 32,  16, kernel_size=4, stride=2, padding=1, bias=False),
					 nn.ReLU()
		)

		self.conv = nn.Conv2d(16, 9, kernel_size=1, stride=1, padding=0, bias=True) 
		
	def forward(self, img):
		tmp = self.resnet18(img)
		tmp = self.conv1(tmp)
		tmp = self.conv2(tmp)
		tmp = self.conv3(tmp)
		tmp = self.conv4(tmp)
		tmp = self.conv5(tmp)
		mod = self.conv(tmp)
		return mod

class Improved(nn.Module):
	def __init__(self):
		super(Improved, self).__init__()
		self.resnet18 = models.resnet18(pretrained=True)
		self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-2])

		self.conv1 = nn.Sequential(
					 nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
					 nn.LeakyReLU(),
					 nn.BatchNorm2d(256),
					 # nn.Dropout2d()
		)

		self.conv2 = nn.Sequential(
					 nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
					 nn.LeakyReLU(),
					 nn.BatchNorm2d(128),
					 # nn.Dropout2d()
		)

		self.conv3 = nn.Sequential(
					 nn.ConvTranspose2d(128,  64, kernel_size=4, stride=2, padding=1, bias=False),
					 nn.LeakyReLU(),
					 nn.BatchNorm2d(64),
					 # nn.Dropout2d()
		)

		self.conv4 = nn.Sequential(
					 nn.ConvTranspose2d( 64,  32, kernel_size=4, stride=2, padding=1, bias=False),
					 nn.LeakyReLU(),
					 nn.BatchNorm2d(32),
					 # nn.Dropout2d()
		)

		self.conv5 = nn.Sequential(
					 nn.ConvTranspose2d( 32,  16, kernel_size=4, stride=2, padding=1, bias=False),
					 nn.LeakyReLU(),
					 nn.BatchNorm2d(16),
					 # nn.Dropout2d()
		)

		self.conv  = nn.Conv2d(16, 9, kernel_size=1, stride=1, padding=0, bias=True)

	def forward(self, img):
		tmp = self.resnet18(img)
		tmp = self.conv1(tmp)
		tmp = self.conv2(tmp)
		tmp = self.conv3(tmp)
		tmp = self.conv4(tmp)
		tmp = self.conv5(tmp)
		mod = self.conv(tmp)
		return mod