dataset_paths = {
	#Face Datasets (In the paper: FFHQ - train, CelebAHQ - test)
	'ffhq': 'datasets/ffhq/ffhq_images_256x256/',
	'celeba_test': 'celeba_hq/val/better/image_256x256/',
	'celeba_train': 'celeba_hq/train/both_masked_256x256/',
	'celeba_mask': 'celeb_repo/CelebAMask-HQ/face_parsing/Data_preprocessing/CelebAMaskHQ-mask/',
	'wild_train': 'afhq/train/wild_masked/',
	'wild_test': 'afhq/val/wild_masked/',
	'dog_train': 'afhq/train/dog_mask/image/',
	'dog_test': 'afhq/val/dog_mask/image/',
    'cat_train': 'afhq/train/cat_mask/image/',
	'cat_test': 'afhq/val/cat_mask/image/',

}

model_paths = {
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pt'
}
