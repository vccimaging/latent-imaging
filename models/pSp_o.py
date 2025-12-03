# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# This code is modified from the original pSp paper by Matheus Souza.

import torch
from torch import nn
from models.encoders import psp_encoders
import numpy as np
from torch.nn import Module
from models.stylegan2.model import EqualLinear
from models.g_mlp import gMLPVision, Attention
from torch.fft import *
from models.masking import MaskGenerator
import torch.nn.functional as F
from models.genxl import generate_images as generate_images_xl

class Landmarks(nn.Module):
    def __init__(self):
        super(Landmarks, self).__init__()

        self.upsample_1 = nn.Upsample(size=(8, 8), mode='bilinear', align_corners=False)
        self.conv_l1 = nn.Conv2d(1024, 68, kernel_size=3, padding=0) 
        self.l1_latent = nn.Linear(18*512, 1024)  
        self.l1 = nn.Linear(64, 2)

        
    def forward(self, x, w):
        ft1 = x["L3_36_1024"].float()
        B,_,_,_ = ft1.shape
        l1_latent = self.l1_latent(w.view(B,-1))
        ft1 = self.conv_l1(ft1+l1_latent.unsqueeze(2).unsqueeze(2))
        ft1 = self.upsample_1(ft1)
        ft1 = self.l1(ft1.view(B,68,64))    
        return ft1
    
class ConvUpsampleModel(nn.Module):
    def __init__(self):
        super(ConvUpsampleModel, self).__init__()
        
        # Define the convolutional layers to project features to segmentation mask
        self.conv_l3 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)  
        self.upsample_36 = nn.Upsample(size=(52, 52), mode='bilinear', align_corners=False)
        self.conv_l9 = nn.Conv2d(1024, 919, kernel_size=3, padding=1)  
        self.upsample_52 = nn.Upsample(size=(84, 84), mode='bilinear', align_corners=False)
        self.conv_l11 = nn.Conv2d(919, 256, kernel_size=3, padding=1)  
        self.upsample_84 = nn.Upsample(size=(148, 148), mode='bilinear', align_corners=False)
        self.conv_l13 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  
        self.upsample_148 = nn.Upsample(size=(276, 276), mode='bilinear', align_corners=False)
        self.conv_l16 = nn.Conv2d(256, 19, kernel_size=3, padding=1)
        self.upsample_256 = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)

        
    def forward(self, x):
    
        ft1 = x["L3_36_1024"].float()
        ft2 = x["L9_52_1024"].float()
        ft3 = x["L11_84_1024"].float()
        ft4 = x["L13_148_919"].float()
        ft5 = x["L15_276_256"].float()
        ft6 = x["L16_256_256"].float()

        ft1 = self.upsample_36(ft1)
        ft2 = self.conv_l3(ft1 + ft2)
        ft2 = self.upsample_52(ft2)
        ft3 = self.conv_l9(ft2 + ft3)
        ft3 = self.upsample_84(ft3)
        ft4 = self.conv_l11(ft3 + ft4)
        ft4 = self.upsample_148(ft4)
        ft5 = self.conv_l13(ft4 + ft5)
        ft5 = self.upsample_256(ft5)
        out = self.conv_l16(ft5 + ft6)

        return out

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

class pSp_o(nn.Module):

    def __init__(self, opts):
        super(pSp_o, self).__init__()
        
        
        self.opts = opts
        img_size = 256
        self.number_layers = opts.styles

        self.mask = MaskGenerator(img_size*img_size, args=opts)
        self.latent_camera = Mapping2StyleGAN(opts=opts)

        self.seg_flag = False
        self.land_flag = False
        self.cls_flag = False

        self.l_cls = nn.Linear(512*18, 40)

        self.l_seg = ConvUpsampleModel()

        self.l_land = Landmarks()
        
        self.decoder, self.latent_avg, self.latent_std =  generate_images_xl(network_pkl='pretrained_models/ffhq256.pkl', opt=self.opts)
        
        self.encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        enc_w = torch.load('pretrained_models/encoder_styleganxl.pt')
        msg = self.encoder.load_state_dict(get_keys(enc_w, 'encoder'), strict=True)
        print("Inversion Encoder Loading:", msg)

        self.load_weights()

    def get_activation(self, layer_name, activations_dict):
        def hook(model, input, output):
            activations_dict[layer_name] = output
        return hook


    def extract_synthesis_features(self, synthesis_model, layers_to_extract, w):
        activations = {}

        # Register hooks for the specified layers
        hooks = []
        for i, (name, layer) in enumerate(synthesis_model.named_children()):
            if i in layers_to_extract:
                hook = layer.register_forward_hook(self.get_activation(name, activations))
                hooks.append(hook)
        
        # Forward pass through the synthesis network
        out = synthesis_model(w,  noise_mode="const")

        #TODO: Partial forward pass
        for hook in hooks:
            hook.remove()
        return activations,out,w

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            
            print('Loading from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')

            print("Loading mask weights from checkpoint")
            msg = self.mask.load_state_dict(ckpt['mask'], strict=True)
            print("Mask:", msg)

            print("Loading Latent Camera weights from checkpoint")
            msg = self.latent_camera.load_state_dict(ckpt['latent_camera'], strict=True)
            print("Latent Camera:", msg)

            if self.opts.generative_model == "styleganxl":
                self.decoder, self.latent_avg, self.latent_std =  generate_images_xl(network_pkl='pretrained_models/ffhq256.pkl', opt=self.opts)
            else:
                raise ValueError
            
            if ckpt['cls'] is not None:
                self.cls_flag = True
                msg = self.l_cls.load_state_dict(ckpt['cls'], strict=True)
                print("CLS:", msg)
            else:
                print("Training CLS from scratch")
            
            if ckpt['segmentation'] is not None:
                self.seg_flag = True
                msg = self.l_seg.load_state_dict(ckpt['segmentation'], strict=True)
                print("SEG:", msg)
            else:
                print("Training SEG from scratch")
           
            if ckpt['landmarks'] is not None:
                self.land_flag = True
                msg = self.l_land.load_state_dict(ckpt['landmarks'], strict=True)
                print("LAND:", msg)
            else:
                print("Training LANDMARKS from scratch")
            self.__load_latent_avg(ckpt)

        else:

            if self.opts.generative_model == "styleganxl":
                self.decoder, self.latent_avg, self.latent_std =  generate_images_xl(network_pkl='pretrained_models/ffhq256.pkl', opt=self.opts)
            else:
                raise ValueError
            

    def forward(self, x, input_code=False, randomize_noise=True,
                return_latents=False, global_step=0):
        
        # If x is a real measurement instead of an image it will just bypass the mask internally.
        masked_x, w = self.mask(x, global_step)

        if self.opts.ini_w != "ft":
            with torch.no_grad():
                gt_encoder = self.encoder(x).detach()
        else:
            gt_encoder = masked_x
                
        l_cam = self.latent_camera(masked_x)
        avg_latent = self.latent_avg.repeat(l_cam.shape[0], self.opts.styles, 1)

        downstream_tasks = []

        # During training we indenpendently train the segmentation, landmarks and classification
        if self.opts.segment and self.opts.landmarks and self.opts.cls:
            layers_to_extract = [4, 9, 11, 13, 15, 16]
            layers, images, _ = self.extract_synthesis_features(self.decoder.synthesis, layers_to_extract, avg_latent + l_cam)
            seg = self.l_seg(layers)
            downstream_tasks.append(seg)

            land = self.l_land(layers, avg_latent + l_cam )
            downstream_tasks.append(land)

            cls = self.l_cls(l_cam.view(l_cam.shape[0],-1))
            downstream_tasks.append(cls)

        elif self.opts.segment:
            layers_to_extract = [4, 9, 11, 13, 15, 16]
            layers, images, _ = self.extract_synthesis_features(self.decoder.synthesis, layers_to_extract, avg_latent + l_cam)
            seg = self.l_seg(layers)
            downstream_tasks.append(seg)

        elif self.opts.cls:
            cls = self.l_cls(l_cam.view(l_cam.shape[0],-1))
            images = self.decoder.synthesis(avg_latent + l_cam, noise_mode="const")
            downstream_tasks.append(cls) 

        elif self.opts.landmarks:
            layers_to_extract = [4]
            layers, images, _ = self.extract_synthesis_features(self.decoder.synthesis, layers_to_extract, avg_latent + l_cam)
            land = self.l_land(layers, avg_latent + l_cam )
            downstream_tasks.append(land)
        else:
            images = self.decoder.synthesis(avg_latent + l_cam, noise_mode="const")
            

        return images, [w,gt_encoder,downstream_tasks], [l_cam]


    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to("cuda")
        elif self.opts.start_from_latent_avg:
            # Compute mean code based on a large number of latents (10,000 here)
            with torch.no_grad():
                print('Computing average latent...')
                self.decoder = self.decoder.to("cuda:0")
                self.latent_avg, self.variance_avg = self.decoder.mean_latent(10000)#.to(self.opts.device))
        else:
            self.latent_avg = None
        if repeat is not None and self.latent_avg is not None:
            self.latent_avg = self.latent_avg.repeat(repeat, 1)

class StyleBlock(Module):
    def __init__(self, in_c, out_c, spatial, opts):
        super(StyleBlock, self).__init__()


        self.multiply = opts.b_multiply
        self.middle_features = opts.b_multiply
        self.attention_features = opts.b_attention_features
        self.residual = opts.b_residual
        self.final_equal = opts.b_final_equal

        num_pools = int(np.log2(spatial))

        self.norm = nn.LayerNorm(in_c)
        self.initial =  nn.Sequential(nn.Linear(in_c, self.middle_features),
                                      nn.SiLU())
        
        modules = []
        if self.attention_features != 0:
            for i in range(num_pools - 1):
                modules += [
                    nn.LayerNorm(self.middle_features),
                    Attention(self.middle_features, self.middle_features, self.attention_features),
                    nn.SiLU(),
                    nn.Dropout(opts.dropout)]
        else:
            for i in range(num_pools - 1):
                modules += [
                    nn.LayerNorm(self.middle_features),
                    nn.Linear(self.middle_features, self.middle_features),
                    nn.SiLU(),
                    nn.Dropout(opts.dropout)]
                
        self.convs = nn.Sequential(*modules)

        if self.final_equal:
            self.linear = EqualLinear(self.middle_features, out_c)
        else:
            self.linear = nn.Linear(self.middle_features, out_c)


    def forward(self, inp):
        inp = self.initial(self.norm(inp))
        if self.residual:
            x = self.convs(inp) + inp
        else:
            x = self.convs(inp)
        x = self.linear(x)
    
        return x
        
        
class Mapping2StyleGAN(Module):
    def __init__(self, opts=None):
        super(Mapping2StyleGAN, self).__init__()
        self.opts = opts

        self.styles = nn.ModuleList()

        self.coarse_ind = 3
        self.middle_ind = 7
        self.number_layers = opts.styles
        out_dim = opts.encoding_size_out
        in_dim = opts.encoding_size

        self.projection = opts.b_projection
        self.coarse_n = opts.b_coarse_n
        self.mid_n = opts.b_mid_n
        self.fine_n = opts.b_fine_n

        self.proj_aggregation = opts.b_proj_aggregation

        self.out_strategy = opts.b_out_strategy
        self.out_ff = opts.b_out_ff
        self.out_depth = opts.b_out_depth

        for i in range(self.number_layers):
            if i < self.coarse_ind:
                style = StyleBlock(in_dim, out_dim, self.coarse_n, opts)
            elif i < self.middle_ind:
                style = StyleBlock(in_dim, out_dim, self.mid_n, opts)
            else:
                style = StyleBlock(in_dim, out_dim, self.fine_n, opts)
            self.styles.append(style)
        
        if self.projection == "linear":
            
            if self.proj_aggregation == "sum":
                self.proj_input_0 = nn.Linear(out_dim,in_dim)
                self.proj_input_1 = nn.Linear(out_dim,in_dim)
            elif self.proj_aggregation == "concat":
                self.proj_input_0 = nn.Linear(out_dim*3,in_dim)
                self.proj_input_1 = nn.Linear(out_dim*4,in_dim)
            else:
                raise ValueError("Invalid projection aggregation")

        elif self.projection == "gmlp":

            self.proj_input_0_glp = gMLPVision(
                        image_size = 256,
                        patch_size = 3,
                        num_classes = 1000,
                        dim = out_dim,
                        depth = 1,
                        attn_dim=None,
                        prob_survival=1.0, 
                        ff_mult=2)
            self.proj_input_0_linear = nn.Linear(out_dim*3,in_dim)
            
            self.proj_input_1_glp = gMLPVision(
                        image_size = 256,
                        patch_size = 4,
                        num_classes = 1000,
                        dim = out_dim,
                        depth = 1,
                        attn_dim=None,
                        prob_survival=1.0, 
                        ff_mult=2)
            self.proj_input_1_linear = nn.Linear(out_dim*4,in_dim)
        else:
            raise ValueError("Invalid projection")    
        

        if self.out_strategy == "none":
            self.out = nn.Identity()
        elif self.out_strategy == "gmlp":
            self.out = gMLPVision(
                    image_size = 256,
                    patch_size = self.number_layers,
                    num_classes = 1000,
                    dim = out_dim,
                    depth = self.out_depth,
                    attn_dim=None,
                    prob_survival=1.0, 
                    ff_mult=self.out_ff
                    )
        elif self.out_strategy == "linear":
            self.out = nn.Linear(out_dim,out_dim)
        elif self.out_strategy == "equal":
            self.out = EqualLinear(out_dim,out_dim)
        else:
            raise ValueError("Invalid out strategy")
    

    def forward(self, x):

        latents = []
        l_acc = 0
        for j in range(self.coarse_ind):
            l = self.styles[j](x)
            if not self.proj_aggregation == "concat":
                l_acc += l
            latents.append(l)
        
        if self.projection == "linear":
            if self.proj_aggregation == "sum":
                x2 = self.proj_input_0(l_acc) + x
            elif self.proj_aggregation == "concat":
                x2 = torch.concat([latents[0],latents[1],latents[2]],1)
                x2 = self.proj_input_0(x2) + x

        elif self.projection == "gmlp":
            x2 = self.proj_input_0_glp(torch.concat([latents[0].unsqueeze(1),latents[1].unsqueeze(1),latents[2].unsqueeze(1)],1)) 
            b,p,c = x2.shape
            x2 = self.proj_input_0_linear(x2.view(b,p*c)) + x

        l_acc2 = 0
        for j in range(self.coarse_ind, self.middle_ind):
            l = self.styles[j](x2)
            if not self.proj_aggregation == "concat":
                l_acc2 += l
            latents.append(l)

        if self.projection == "linear":
            if self.proj_aggregation == "sum":
                x3 = self.proj_input_1(l_acc2) + x
            elif self.proj_aggregation == "concat":
                x3 = torch.concat([latents[3],latents[4],latents[5],latents[6]],1)
                x3 = self.proj_input_0(x3) + x
        elif self.projection == "gmlp":
            x3 = self.proj_input_1_glp(torch.concat([latents[3].unsqueeze(1),latents[4].unsqueeze(1),latents[5].unsqueeze(1),latents[6].unsqueeze(1)],1)) 
            b,p,c = x3.shape
            x3 = self.proj_input_1_linear(x3.view(b,p*c)) + x


        for j in range(self.middle_ind, self.number_layers):
            l = self.styles[j](x3)
            latents.append(l)

        out = torch.stack(latents, dim=1)
                
        return self.out(out)
    