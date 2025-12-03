from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import numpy as np
import torch
import os

class ImagesDataset_CLS(Dataset):

	def __init__(self, source_root, other_root, opts, target_transform=None, source_transform=None, masks=None, transforms_mask=None, train=True):
		self.opts = opts
		if train:
	
			path_celeb_train = source_root
			celebimages = sorted(data_utils.make_dataset(path_celeb_train))
			self.source_paths = celebimages
			self.target_paths = celebimages
			
			names = [n.split("/")[-1] for n in self.source_paths]

			attributes_gt = []
			images_names = []
			with open('datasets/list_attr_celeba.txt', 'r') as file:
				# Loop through each line in the file
				for line in file:
					try:

						a = line.split()
						name = a[0]
						if name in names:
							print(name)
							images_names.append(name)
							list_att = a[1:]
							list_att_float = [float(char) for char in list_att]
							attributes_gt.append(torch.tensor(list_att_float))

					except:
						print("Line with less attributes")		

			self.attributes_gt = torch.stack(attributes_gt)		
			assert images_names == names

		else:
			
			path_celeb_train = source_root
			celebimages = sorted(data_utils.make_dataset(path_celeb_train))
			self.source_paths = celebimages
			self.target_paths = celebimages
			names = [n.split("/")[-1] for n in self.source_paths]
			
			attributes_gt = []
			images_names = []
			with open('datasets/list_attr_celeba.txt', 'r') as file:
				# Loop through each line in the file
				for line in file:
					try:

						a = line.split()
						name = a[0]
						if name in names:
							print(name)
							images_names.append(name)
							list_att = a[1:]
							list_att_float = [float(char) for char in list_att]
							tensor_list_att = attributes_gt.append(torch.tensor(list_att_float))

					except:
						print("Line with less attributes")		

			self.attributes_gt = torch.stack(attributes_gt)
						
			assert images_names == names

		self.source_transform = source_transform
		self.target_transform = target_transform
		self.transforms_mask = transforms_mask
		self.opts = opts

		if self.opts.ini_w == "ft":
			if not "celeb" in source_root:

				path_train = 'DMD_DATA/DMD_TRAINSET/measurement/'
				self.real_measurement = sorted([os.path.join(path_train, f) for f in os.listdir(path_train) if os.path.isfile(os.path.join(path_train, f))])

				path_gt = 'DMD_DATA/DMD_TRAINSET/gt/'
				self.images = sorted([os.path.join(path_gt, f) for f in os.listdir(path_gt) if os.path.isfile(os.path.join(path_gt, f))])

			else:
				path_train = 'DMD_DATA/DMD_TESTSET/measurement/'
				self.real_measurement = sorted([os.path.join(path_train, f) for f in os.listdir(path_train) if os.path.isfile(os.path.join(path_train, f))])

				path_gt = 'DMD_DATA/DMD_TESTSET/gt/'
				self.images = sorted([os.path.join(path_gt, f) for f in os.listdir(path_gt) if os.path.isfile(os.path.join(path_gt, f))])
				

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


			return not_augmented, im_augmented, self.attributes_gt[index]
			
		else:

			loaded = np.load(self.real_measurement[index])
			measure = torch.from_numpy(loaded['array'])
			gt = Image.open(self.images[index])
			gt = gt.convert('RGB')
			gt = self.source_transform(gt)

			return gt, gt, measure
