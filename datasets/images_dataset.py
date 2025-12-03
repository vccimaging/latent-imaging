from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import numpy as np
import torch
import torchvision.transforms as transforms
import os

#/ibex/ai/home/medeirmv/datasets/ffhq/ffhq_image

class ImagesDataset(Dataset):

	def __init__(self, source_root, other_root, opts, target_transform=None, source_transform=None, masks=None, transforms_mask=None, train=True):

		self.opts = opts
		if train:
			path = source_root
			# Add celeba training images to the training set 28k images
			self.source_paths = sorted(data_utils.make_dataset(path))
			self.target_paths = sorted(data_utils.make_dataset(path))

	
			path_celeb_train = other_root
			celebimages = sorted(data_utils.make_dataset(path_celeb_train))
			self.source_paths += celebimages
			self.target_paths += celebimages

		else:
			
			path_celeb_test = source_root
			# utilizes the celeba testset with 2k images
			self.source_paths = sorted(data_utils.make_dataset(path_celeb_test))
			self.target_paths = sorted(data_utils.make_dataset(path_celeb_test))


		self.source_transform = source_transform
		self.target_transform = target_transform
		self.transforms_mask = transforms_mask
		self.opts = opts

		if self.opts.ini_w == "ft":
			if not "celeb" in source_root:

				path_train = 'DMD_DATA/FineTune/'+str(self.opts.encoding_size)+'/Train/'
				self.real_measurement = sorted([os.path.join(path_train, f) for f in os.listdir(path_train) if os.path.isfile(os.path.join(path_train, f))])

				path_gt = 'DMD_DATA/DMD_TRAINSET/gt/'
				self.images = sorted([os.path.join(path_gt, f) for f in os.listdir(path_gt) if os.path.isfile(os.path.join(path_gt, f))])

			else:
				path_train = 'DMD_DATA/FineTune/'+str(self.opts.encoding_size)+'/Test/'
				self.real_measurement = sorted([os.path.join(path_train, f) for f in os.listdir(path_train) if os.path.isfile(os.path.join(path_train, f))])

				path_gt = 'DMD_DATA/DMD_TESTSET/gt/'
				self.images = sorted([os.path.join(path_gt, f) for f in os.listdir(path_gt) if os.path.isfile(os.path.join(path_gt, f))])

				new_list = []
				for j in range(len(self.real_measurement)):
					new_list.append(self.real_measurement[j].replace("FineTune/"+str(self.opts.encoding_size)+"/Test/", "DMD_TESTSET/gt/").replace(".npz",".jpg"))
				self.images = new_list
				

	def __len__(self):
		if self.opts.ini_w == "ft":
			return len(self.real_measurement)
		else:
			return len(self.source_paths)

	def __getitem__(self, index):
		if not self.opts.ini_w == "ft":
			from_path = self.source_paths[index]
			from_im = Image.open(from_path)
			from_im = from_im.convert('RGB')

			to_path = self.target_paths[index]
			to_im = Image.open(to_path).convert('RGB')

			
			im_augmented = self.source_transform(to_im)
			not_augmented = self.target_transform(from_im)

			return not_augmented, im_augmented, im_augmented
			
			
		else:

			loaded = np.load(self.real_measurement[index])
			measure = torch.from_numpy(loaded['array'])
			gt = Image.open(self.images[index])
			gt = gt.convert('RGB')
			gt = self.source_transform(gt)

			return gt, gt, measure
