from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import torchvision.transforms as transforms



class ImagesDataset_SEG(Dataset):

	def __init__(self, source_root, other_root, opts, target_transform=None, source_transform=None, masks=None, transforms_mask=None, train=True):
		self.opts = opts
		self.mask_root = masks
		if train:
			path_celeb_train = source_root
			celebimages = sorted(data_utils.make_dataset(path_celeb_train))
			self.source_paths = celebimages
			self.target_paths = celebimages

		else:
			
			path_celeb_train = source_root
			celebimages = sorted(data_utils.make_dataset(path_celeb_train))
			self.source_paths = celebimages
			self.target_paths = celebimages

		self.source_transform = source_transform
		self.target_transform = target_transform
		self.transforms_mask = transforms_mask
		self.opts = opts


		self.mask_transform = transforms.Compose([
		transforms.ToTensor()            # Convert to PyTorch tensor
		])

				

	def __len__(self):
		if self.opts.ini_w == "ft":
			return len(self.real_measurement)
		else:
			return len(self.source_paths)

	def __getitem__(self, index):

		from_path = self.source_paths[index]
		name = (self.source_paths[index]).split("/")[-1][:-4]+".png"

		mask_path = self.mask_root+name

		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB')

		
		not_augmented = self.source_transform(from_im)

		mask = Image.open(mask_path).convert('L')
		mask = (self.mask_transform(mask)*255.0).long()


		return not_augmented, not_augmented, mask
			
	