import os
import random
import matplotlib
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")
matplotlib.use('Agg')
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common, train_utils
from criteria import id_loss, moco_loss, w_norm
from configs import data_configs

from datasets.images_cls import ImagesDataset_CLS
from datasets.images_seg import ImagesDataset_SEG
from datasets.images_dataset import ImagesDataset
from utils.segmentation_utils import FocalLoss
from utils.metrics_downstream import detect_faces
from criteria.lpips.lpips import LPIPS

from models.pSp_o import pSp_o
from training.ranger import Ranger
import torch.nn.functional as F
from lion_pytorch import Lion

from criteria.splice.DoesFS.splice_utils.splice import Splice
import facer

random.seed(0)
torch.manual_seed(0)



class Coach:
    def __init__(self, opts, prev_train_checkpoint=None):
        
        self.opts = opts

        if (self.opts.landmarks+self.opts.segment+self.opts.cls) > 1:
            raise ValueError("This implementation only train one downstream task at a time")
        
        self.global_step = 0
        self.device = 'cuda:0'

        self.net = pSp_o(self.opts).to(self.device)

        self.face_parser = facer.face_parser('farl/celebm/448', device=self.device) # optional "farl/celebm/448"
        self.face_landmark = facer.face_aligner('farl/ibug300w/448', device=self.device)
        self.face_detector = facer.face_detector('retinaface/mobilenet', device="cuda:0")

        if self.opts.ini_w == "ft":
            if self.opts.cls:
                self.face_attr = facer.face_attr("farl/celeba/224", device=self.device)

        # Initialize loss
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type=self.opts.lpips_type).to(self.device).eval()
        if self.opts.id_lambda > 0:
            if 'ffhq' in self.opts.dataset_type or 'celeb' in self.opts.dataset_type:
                self.id_loss = id_loss.IDLoss().to(self.device).eval()
            else:
                self.id_loss = moco_loss.MocoLoss(opts).to(self.device).eval()
        self.mse_loss = nn.MSELoss().to(self.device).eval()
        self.wnorm = w_norm.WNormLoss()
        if self.opts.dino > 0:
            self.dino_loss = Splice(device=self.device)
        if self.opts.energy_loss_lambda > 0:
            self.energy_gt = []
            energy = self.opts.energy_a
            for en in range(self.opts.encoding_size):
                if energy > self.opts.energy_b:
                    energy = self.opts.energy_a
                self.energy_gt.append(energy/100)
                energy += 1   
            
            self.energy_gt = torch.tensor(self.energy_gt)
            perm = torch.randperm(self.energy_gt.size(0))
            self.energy_gt = self.energy_gt[perm]
        self.loss_linear_cls = nn.BCEWithLogitsLoss()
        self.loss_segmentation = FocalLoss()

        # Initialize optimizer
        if self.opts.ini_w != "ft":
            self.optimizer, self.optimizer_mask = self.configure_optimizers()
        else:
            self.optimizer = self.configure_optimizers()

        
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                        batch_size=self.opts.batch_size,
                                        shuffle=True,
                                        num_workers=int(self.opts.workers),
                                        drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                        batch_size=self.opts.test_batch_size,
                                        shuffle=False,
                                        num_workers=int(self.opts.test_workers),
                                        drop_last=True)

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

        if prev_train_checkpoint is not None:
            self.load_from_train_checkpoint(prev_train_checkpoint)
            prev_train_checkpoint = None


    def load_from_train_checkpoint(self, ckpt):
        print('Loading previous training data...')
        self.global_step = ckpt['global_step'] + 1
        self.best_val_loss = ckpt['best_val_loss']
        self.net.load_state_dict(ckpt['state_dict'])

        if self.opts.keep_optimizer:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        print(f'Resuming training from step {self.global_step}')

    def train(self):
        self.net.train()
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss_dict = {}
        
                # if self.opts.ini_w == "ft":
                #     x, y, y_hat, latent, l_cam, mask = self.forward(batch)
                #     loss, encoder_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent,l_cam, mask)
                # else:
                x, y, y_hat, latent, l_cam, mask = self.forward(batch)
                loss, encoder_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent,l_cam, mask)
                loss_dict = {**loss_dict, **encoder_loss_dict}

                if not self.opts.ini_w == "ft":
                
                    self.optimizer.zero_grad()
                    self.optimizer_mask.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer_mask.step()
                else :
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 25 == 0):
                    self.parse_and_log_images(id_logs, x, y, y_hat, title='images/train/faces')
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1

                #if self.opts.scheduler and self.global_step > 1000:
                    #self.scheduler_lcam.step()
                    #self.scheduler_mask.step()


    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            cur_loss_dict = {}
            with torch.no_grad():
                if self.opts.ini_w == "ft":
                    x, y, y_hat, latent, l_cam, mask = self.forward(batch)
                    loss, cur_encoder_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent,l_cam, mask)
                else:
                    x, y, y_hat, latent, l_cam, mask = self.forward(batch)
                    loss, cur_encoder_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent,l_cam, mask)
                cur_loss_dict = {**cur_loss_dict, **cur_encoder_loss_dict}
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            self.parse_and_log_images(id_logs, x, y, y_hat,
                                      title='images/test/faces',
                                      subscript='{:04d}'.format(batch_idx))
            if self.global_step == 0 and batch_idx >= 4:
                self.net.train()
                return None

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict


    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        checkpoint_path_name = os.path.join(self.checkpoint_dir, str(self.global_step)+'.pt')
        torch.save(save_dict, checkpoint_path)
        torch.save(save_dict, checkpoint_path_name)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    '**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))
    
    
    def set_requires_grad_true(self, module):
        for param in module.parameters():
            param.requires_grad = True
    
    def configure_optimizers(self):

 
        params = list(self.net.latent_camera.parameters())
            
        if self.opts.cls:
            params = list(self.net.l_cls.parameters())
        if self.opts.segment:
            params = list(self.net.l_seg.parameters())
        if self.opts.landmarks:
            params = list(self.net.l_land.parameters())

        if not self.opts.ini_w == "ft":
            params_mask = list(self.net.mask.parameters())
            
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        elif self.opts.optim_name == 'lion':
            optimizer = Lion(params, lr=self.opts.learning_rate, weight_decay=1e-2)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)

        if not self.opts.ini_w == "ft":
            if self.opts.optim_name_mask == 'adam':
                optimizer_mask = torch.optim.Adam(params_mask, lr=self.opts.learning_rate_mask)
            elif self.opts.optim_name_mask == 'lion':
                optimizer_mask = Lion(params_mask, lr=self.opts.learning_rate_mask, weight_decay=1e-2)
            else:
                optimizer_mask = Ranger(params_mask, lr=self.opts.learning_rate_mask)

            return optimizer, optimizer_mask
        else:
            return optimizer

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception('{} is not a valid dataset_type'.format(self.opts.dataset_type))
        print('Loading dataset for {}'.format(self.opts.dataset_type))
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()

        if self.opts.cls and self.opts.ini_w != "ft":
            train_dataset = ImagesDataset_CLS(source_root=dataset_args['celeba_train'],
                                      other_root=None,
                                      source_transform=transforms_dict['transform_source'],
                                      target_transform=transforms_dict['transform_gt_train'],
                                      opts=self.opts,
                                      masks=dataset_args['celeba_mask'],
                                      transforms_mask=transforms_dict['transform_mask'],
                                      train=True)
            test_dataset = ImagesDataset_CLS(source_root=dataset_args['celeba_test'],
                                     other_root=None,
                                     source_transform=transforms_dict['transform_source'],
                                     target_transform=transforms_dict['transform_test'],
                                     opts=self.opts,
                                     masks=dataset_args['celeba_mask'],
                                     transforms_mask=transforms_dict['transform_mask'],
                                     train=False)
        elif (self.opts.landmarks or self.opts.segment) and self.opts.ini_w != "ft":
            train_dataset = ImagesDataset_SEG(source_root=dataset_args['celeba_train'],
                                      other_root=None,
                                      source_transform=transforms_dict['transform_source'],
                                      target_transform=transforms_dict['transform_gt_train'],
                                      opts=self.opts,
                                      masks=dataset_args['celeba_mask'],
                                      transforms_mask=transforms_dict['transform_mask'],
                                      train=True)
            test_dataset = ImagesDataset_SEG(source_root=dataset_args['celeba_test'],
                                     other_root=None,
                                     source_transform=transforms_dict['transform_source'],
                                     target_transform=transforms_dict['transform_test'],
                                     opts=self.opts,
                                     masks=dataset_args['celeba_mask'],
                                     transforms_mask=transforms_dict['transform_mask'],
                                     train=False)
        else:
            train_dataset = ImagesDataset(source_root=dataset_args['ffhq'],
                                        other_root=dataset_args['celeba_train'],
                                        source_transform=transforms_dict['transform_source'],
                                        target_transform=transforms_dict['transform_gt_train'],
                                        opts=self.opts,
                                        masks=None,
                                        transforms_mask=transforms_dict['transform_mask']   ,
                                        train=True)
            test_dataset = ImagesDataset(source_root=dataset_args['celeba_test'],
                                        other_root=None,
                                        source_transform=transforms_dict['transform_source'],
                                        target_transform=transforms_dict['transform_test'],
                                        opts=self.opts,
                                        masks=None,
                                        transforms_mask=transforms_dict['transform_mask'],
                                        train=False)
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset

    def calc_loss(self, x, y, y_hat, latent, l_cam, mask=None):
        loss_dict = {}
        loss = 0.0
        id_logs = None

        if self.opts.energy_loss_lambda > 0:
            if self.opts.encoder_lambda > 0:
                energy_pred = torch.sum(latent[0],dim=0)/(latent[0].shape[0])
            else:
                energy_pred = torch.sum(latent,dim=0)/((latent.shape[0]))
            energy_loss = F.l1_loss(energy_pred,self.energy_gt.to(energy_pred.device))
            loss_dict['energy_loss'] = float(energy_loss)
            loss += energy_loss * self.opts.energy_loss_lambda
            
        if self.opts.ini_w == "ft":
            
            with torch.no_grad():
                gt = self.net.encoder(y)

            loss_latent = F.l1_loss(gt.detach(),l_cam[0])
            loss_dict['loss_latent'] = float(loss_latent)
            loss += loss_latent

        if self.opts.segment:
            with torch.no_grad():
                input_seg = ((x.detach() + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
                detect = detect_faces(input_seg, self.face_detector)
                faces = self.face_parser(input_seg,detect)
                seg_logits = faces['seg']['logits']
                seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
                vis_seg_probs = seg_probs.argmax(dim=1).long()

            loss_seg = self.loss_segmentation(latent[2][0], vis_seg_probs.long().to(latent[2][0].device))
            loss_dict['loss_segmentation'] = float(loss_seg)
            loss += loss_seg

            
        if self.opts.landmarks:
            with torch.no_grad():
                input_seg = ((x.detach() + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
                detect = detect_faces(input_seg, self.face_detector)
                faces_land = (self.face_landmark(input_seg,detect)['alignment'])/255.0

            loss_landmarks = F.mse_loss(latent[2][0], faces_land.to(latent[2][0].device))
            loss_dict['loss_landmarks'] = float(loss_landmarks)
            loss += loss_landmarks
            
        if self.opts.cls:
            if not self.opts.ini_w == "ft":
                attributes_gt = (mask+1)/2
                correct_predictions = torch.sum(torch.round(F.sigmoid(latent[2][0])) == attributes_gt.to(latent[2][0].device),1)
                accuracy = ((correct_predictions / 40) * 100).mean()
                loss_linear_cls = self.loss_linear_cls(latent[2][0], attributes_gt.to(latent[2][0].device))
                loss_dict['linear_class_percentage_right'] = float(accuracy)
                loss_dict['loss_linear_class'] = float(loss_linear_cls)
                loss += loss_linear_cls * 1.0#self.opts.encoder_lambda

            else:
                correct_predictions = torch.sum(torch.round(F.sigmoid(latent[2][0])) == torch.round(F.sigmoid(latent[1])).detach(),1)
                accuracy = ((correct_predictions / 40) * 100).mean()
                loss_linear_cls = self.loss_linear_cls(latent[2][0], torch.round(F.sigmoid(latent[1])).detach().float().to(latent[2][0].device))
                loss_dict['linear_class_percentage_right'] = float(accuracy)
                loss_dict['loss_linear_class'] = float(loss_linear_cls)
                loss += loss_linear_cls * 1.0#self.opts.encoder_lambda

        if self.opts.dino > 0:
            loss_dino = self.dino_loss.calculate_sim_loss(y_hat, y, mode='f', layer_num=11)
            loss_dict['loss_dino'] = float(loss_dino)
            loss += loss_dino * self.opts.dino

        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda

        if self.opts.w_norm_lambda > 0:
            loss_w_norm = self.wnorm(l_cam, self.net.latent_avg)
            loss_dict['loss_w_norm'] = float(loss_w_norm)
            loss += loss_w_norm * self.opts.w_norm_lambda
            
        if self.opts.encoder_lambda > 0 and self.opts.ini_w != "ft":

            loss_encoder = F.l1_loss(l_cam[0], latent[1])#loss_encoder / count
            loss_dict['loss_encoder'] = float(loss_encoder)
            loss += loss_encoder * self.opts.encoder_lambda

        if self.opts.id_lambda > 0:  # Similarity loss
            loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x, mask)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_id * self.opts.id_lambda
        
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat , y)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda

        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def forward(self, batch):

        x, y, mask = batch
        x, y = x.to(self.device).float(), y.to(self.device).float()

        if self.opts.ini_w == "ft":
            y_hat, latent, l_cam = self.net.forward(mask.to(self.device), return_latents=True, global_step=self.global_step)
        else:
            y_hat, latent, l_cam = self.net.forward(x, return_latents=True, global_step=self.global_step)

        return x, y, y_hat, latent, l_cam, mask

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=2):
        im_data = []

        for i in range(display_count):
            cur_im_data = {
                'input_face': common.log_input_image(x[i], self.opts),
                'target_face': common.tensor2im(y[i]),
                'output_face': common.tensor2im(y_hat[i]),
            }
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
        else:
            path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        
        mask = self.net.mask.state_dict()
        latent_camera = self.net.latent_camera.state_dict()
        latent_avg = self.net.latent_avg

        # Checking if we are training or loading a checkpoint
        if self.opts.segment or self.net.seg_flag:
            l_seg = self.net.l_seg.state_dict()
        else:
            l_seg = None
        if self.opts.landmarks or self.net.land_flag:
            l_land = self.net.l_land.state_dict()
        else:
            l_land = None
        if self.opts.cls or self.net.cls_flag:
            l_cls = self.net.l_cls.state_dict()
        else:
            l_cls = None

        save_dict = {
            'mask': mask,
            'landmarks': l_land,
            'cls': l_cls,
            'segmentation': l_seg,
            'latent_avg': latent_avg,
            'latent_camera': latent_camera,
            'opts': vars(self.opts)
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = self.net.latent_avg

        if self.opts.save_training_data:  # Save necessary information to enable training continuation from checkpoint
            save_dict['global_step'] = self.global_step
            save_dict['optimizer'] = self.optimizer.state_dict()
            save_dict['best_val_loss'] = self.best_val_loss
        return save_dict
    
    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag