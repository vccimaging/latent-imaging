from argparse import ArgumentParser
from configs.paths_config import model_paths

def str2bool(v):
    return v.lower() in ('yes', 'y', 'true', 't', '1')
class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
        self.parser.add_argument('--dataset_type', default='ffhq_encode', type=str,
                                 help='Type of dataset/experiment to run')
        self.parser.add_argument('--batch_size', default=4, type=int, help='training batch size')
        self.parser.add_argument('--encoding_size', default=512, type=int, help='encode size in')
        self.parser.add_argument('--encoding_size_out', default=512, type=int, help='generative model latent size')

        self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=2, type=int,
                                 help='Number of test/inference dataloader workers')

        #Optimizers
        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--learning_rate_mask', default=0.0001, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
        self.parser.add_argument('--optim_name_mask', default='lion', type=str, help='Which optimizer to use')
        self.parser.add_argument('--start_from_latent_avg', action='store_true',
                                 help='Whether to add average latent vector to generate codes from encoder.')

        #Losses
        self.parser.add_argument('--lpips_type', default='alex', type=str, help='LPIPS backbone')
        self.parser.add_argument('--lpips_lambda', default=0.0, type=float, help='LPIPS loss multiplier factor')
        self.parser.add_argument('--id_lambda', default=0.0, type=float, help='ID loss multiplier factor')
        self.parser.add_argument('--l2_lambda', default=0.0, type=float, help='L2 loss multiplier factor')
        self.parser.add_argument('--w_norm_lambda', default=0.0, type=float, help='latent loss')
        self.parser.add_argument('--energy_loss_lambda', default=0.0, type=float, help='energy masks loss')
        self.parser.add_argument('--energy_a', default=10, type=int, help='90% of the energy')
        self.parser.add_argument('--energy_b', default=90, type=int, help='10% of the energy')
        self.parser.add_argument('--dino', default=1.0, type=float, help='dino features')
        self.parser.add_argument('--encoder_lambda', default=0.0, type=float, help='similarity with the latent loss')



        self.parser.add_argument('--quantizing_strategy', default="detach", type=str, help='quantizing strategy')
        self.parser.add_argument('--generative_model', default="styleganxl", type=str, help='generative model used')
        
        
        self.parser.add_argument('--stylegan_size', default=1024, type=int,
                                 help='size of pretrained StyleGAN Generator')
        self.parser.add_argument('--checkpoint_path', default=None, type=str, help='path to checkpoint')
        self.parser.add_argument('--verbose', default=True, type=str2bool, help='')


        #Downstream Tasks
        self.parser.add_argument('--cls', default=False, type=str2bool, help='classification')
        self.parser.add_argument('--landmarks', default=False, type=str2bool, help='landmarks')
        self.parser.add_argument('--segment', default=False, type=str2bool, help='segmentation')

        

        self.parser.add_argument('--ini_w', default="pass", type=str, help='')
        self.parser.add_argument('--norm_w', default="pass", type=str, help='')
        self.parser.add_argument('--clip_w', default="clamp", type=str, help='')
        self.parser.add_argument('--bits', default=1, type=int, help='number of bits quantization')
        self.parser.add_argument('--styles', default=18, type=int, help='styles')
        self.parser.add_argument('--w_dropout', default=0.0, type=float, help='dropout w')
        self.parser.add_argument('--dropout', default=0.0, type=float, help='drpoout')

        self.parser.add_argument('--max_steps', default=300000, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=1000, type=int,
                                 help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int,
                                 help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
        self.parser.add_argument('--save_interval', default=None, type=int, help='Model checkpoint interval')


        #Save additional training info to enable future training continuation from produced checkpoints
        self.parser.add_argument('--save_training_data', action='store_true',
                                 help='Save intermediate training data to resume training from the checkpoint')
        self.parser.add_argument('--sub_exp_dir', default=None, type=str, help='Name of sub experiment directory')
        self.parser.add_argument('--keep_optimizer', action='store_true',
                                 help='Whether to continue from the checkpoint\'s optimizer')
        self.parser.add_argument('--resume_training_from_ckpt', default=None, type=str,
                                 help='Path to training checkpoint, works when --save_training_data was set to True')
        self.parser.add_argument('--update_param_list', nargs='+', type=str, default=None,
                                 help="Name of training parameters to update the loaded training checkpoint")
        
        #From intermediate latent to final latent model
        self.parser.add_argument('--b_coarse_n', default=16, type=int, help='depth coarse')
        self.parser.add_argument('--b_mid_n', default=32, type=int, help='depth of mid')
        self.parser.add_argument('--b_fine_n', default=64, type=int, help='depth of fine')
        self.parser.add_argument('--b_multiply', default=512, type=int, help='style mid project dim multiply by this')
        self.parser.add_argument('--b_attention_features', default=128, type=int, help='0 no attention, 64/128/256 attention dim')
        self.parser.add_argument('--b_residual', default=True, type=str2bool, help='use or not residual')
        self.parser.add_argument('--b_final_equal', default=False, type=str2bool, help='equalized linear or nothing')
        self.parser.add_argument('--b_projection', default="gmlp", type=str, help='linear, gmlp')
        self.parser.add_argument('--b_proj_aggregation', default="sum", type=str, help='linear, gmlp')
        self.parser.add_argument('--b_out_strategy', default="gmlp", type=str, help='none, gmlp, linear, equal')
        self.parser.add_argument('--b_out_ff', default=2, type=int, help='linear, gmlp')
        self.parser.add_argument('--b_out_depth', default=1, type=int, help='linear, gmlp')

    def parse(self):
        opts = self.parser.parse_args()
        return opts
