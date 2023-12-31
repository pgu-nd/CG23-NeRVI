from utils import *
import numpy as np
import torch
import skimage 
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage import data,img_as_float
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os




class ViewSynthesis_CNN():
	def __init__(self,args):
		self.batch_size = args.batch_size
		self.res = 512 ### image resolution
		self.data_path = '/afs/crc.nd.edu/user/p/pgu/Research/../'
		self.iso = [-0.90, -0.80, -0.70, -0.60, -0.50, -0.40, -0.30]
		self.test_iso = [-0.85,-0.75,-0.65,-0.55,-0.45,-0.35]
		#self.iso[4] = 0.0

	def GetMask(self,image):
		mask = np.sum(image,axis=0,keepdims=True)/3.0
		mask[mask!=1.0] = 0
		mask = 1-mask
		return mask

	def ReadData(self):
		self.view_parms = []
		self.imgs = []
		self.masks = []
		self.start = [0,1,2,3,4,5,6]
		self.test_start = [1,2,3,4,5,6]
		idx = 0
		for iso in self.iso:
			print('iso value', iso)
			iso_ = iso-self.iso[0]/(self.iso[-1]-self.iso[0])
			iso_ -= 0.5
			iso_ *= 2.0
			for theta in range(self.start[idx],180,12):
				theta_ = theta/179.0
				theta_ -= 0.5
				theta_ *= 2.0 ### normaize theta parameter
				for phi in range(self.start[idx],360,12):
					phi_ = phi/359.0
					phi_ -= 0.5
					phi_ *= 2.0 ### normaize theta parameter
					self.view_parms.append([iso_,phi_,theta_])
					img = img_as_float(imread(self.data_path+'/save-iso-'+'{:4f}'.format(iso)+'-theta-'+'{:6f}'.format(theta)+'-phi-'+'{:6f}'.format(phi)+'.png')) ### read image
					img = img.transpose(2,0,1)
					mask = self.GetMask(img) ### compute the forground and background of the image
					self.imgs.append(img)
					self.masks.append(mask)
			idx += 1
		idx_test =0
		for iso in self.test_iso:
			print('test_iso value', iso)
			iso_ = iso-self.test_iso[0]/(self.test_iso[-1]-self.test_iso[0])
			iso_ -= 0.5
			iso_ *= 2.0
			for theta in range(self.test_start[idx_test],180,9):
				theta_ = theta/179.0
				theta_ -= 0.5
				theta_ *= 2.0 ### normaize theta parameter
				for phi in range(self.test_start[idx_test],360,9):
					phi_ = phi/359.0
					phi_ -= 0.5
					phi_ *= 2.0 ### normaize theta parameter
					self.view_parms.append([iso_,phi_,theta_])
					img = img_as_float(imread(self.data_path+'/save-iso-'+'{:4f}'.format(iso)+'-theta-'+'{:6f}'.format(theta)+'-phi-'+'{:6f}'.format(phi)+'.png')) ### read image
					img = img.transpose(2,0,1)
					mask = self.GetMask(img) ### compute the forground and background of the image
					self.imgs.append(img)
					self.masks.append(mask)
			idx_test += 1

		self.imgs = np.asarray(self.imgs)
		self.masks = np.asarray(self.masks)
		self.view_parms = np.asarray(self.view_parms)
		print(self.imgs.shape)

	def GetTrainingData(self):
		training_data_input = torch.FloatTensor(self.view_parms)
		training_data_output_imgs = torch.FloatTensor(self.imgs)
		training_data_output_masks = torch.FloatTensor(self.masks)
		
		data = torch.utils.data.TensorDataset(training_data_input,training_data_output_imgs,training_data_output_masks) ### wrap the params, images, and masks
		train_loader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=True)
		return train_loader

