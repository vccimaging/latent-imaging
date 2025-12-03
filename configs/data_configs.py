from configs import transforms_config
from configs.paths_config import dataset_paths

DATASETS = {
	'ffhq_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'ffhq': dataset_paths['ffhq'],
		'celeba_test': dataset_paths['celeba_test'],
		'celeba_train': dataset_paths['celeba_train'],
		'celeba_mask': dataset_paths['celeba_mask'],
	},
	'wild': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['wild_train'],
		'train_target_root': dataset_paths['wild_train'],
		'test_source_root': dataset_paths['wild_test'],
		'test_target_root': dataset_paths['wild_test'],
        'celeba_mask': dataset_paths['ffhq'],
		'ffhq_mask': dataset_paths['celeba_test'],
	},
    'cat': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['cat_train'],
		'train_target_root': dataset_paths['cat_train'],
		'test_source_root': dataset_paths['cat_test'],
		'test_target_root': dataset_paths['cat_test'],
        'celeba_mask': dataset_paths['ffhq'],
		'ffhq_mask': dataset_paths['celeba_test'],
	},
    'dog': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['dog_train'],
		'train_target_root': dataset_paths['dog_train'],
		'test_source_root': dataset_paths['dog_test'],
		'test_target_root': dataset_paths['dog_test'],
        'celeba_mask': dataset_paths['ffhq'],
		'ffhq_mask': dataset_paths['celeba_test'],
    }
}
