import sys
sys.path.append(".")
sys.path.append("..")

import torch
import torch
import os
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

class quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input):
        qtz_level=1
        return (torch.round(input/input.max()*qtz_level) / qtz_level)*input.max()
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.clamp(grad_output, -1, 1)
        return grad_input
quantizer = quantize.apply

class MaskGenerator(nn.Module):
    def __init__(self,in_dim,args) -> None:
        super(MaskGenerator, self).__init__()

        self.args = args
        self.clamping = args.clip_w
        self.latent_size = args.encoding_size
        self.input_size = in_dim
        self.quantizing_strategy = args.quantizing_strategy
        self.verbose = args.verbose
        self.bits = args.bits
        self.weight_drop = nn.Dropout(args.w_dropout)
        self.input_drop = nn.Dropout(args.dropout)
    
        if args.ini_w == "normalized":
            self.weight = nn.Parameter(torch.rand(self.latent_size, in_dim)/self.latent_size, requires_grad=True)
        else:
            self.weight = nn.Parameter(torch.rand(self.latent_size, in_dim), requires_grad=True)


    def plotting_mask(self):

        folder_path = self.args.exp_dir + "/masks/"
        os.makedirs(folder_path, exist_ok=True)

        if self.quantizing_strategy == "detach":
            weight = self.quantizing_detach(self.weight.permute(1,0))
        else:
            weight = self.quantizing(self.weight.permute(1,0))

        for i in range(self.latent_size):
            plt.figure()
            img = weight[:,i].reshape(256,256).to("cpu").detach().numpy()
            plt.imshow(img, interpolation="none")
            plt.colorbar()
            plt.savefig(folder_path + "mask_" + str(i) + ".png")
            plt.close()

    def quantizing_detach(self, real):

        b = (2**self.bits) - 1

        if self.clamping == None:
            return real
        
        elif self.clamping == "clamp":
            real_clamp = torch.clamp(real, 0, 1)
            quantized = (real_clamp*b).round()/b
            return (quantized - real).detach() + real
        else:
            raise NotImplementedError("Quantization strategy not implemented.")
    
    def quantizing(self, real):
        real = torch.clamp(real, 0, 1)
        b = (2**self.bits) - 1
        quantized = quantizer(real)
        return quantized
    
    def forward(self, image, iter):
        if image.shape[1] == 512 or image.shape[1] == 256 or image.shape[1] == 128 or image.shape[1] == 64 or image.shape[1] == 32 or image.shape[1] == 16 or image.shape[1] == 8 or image.shape[1] == 4:
            # if its the real measurement we just by pass the mask.
            return image.float(), None
        else:
            image = self.input_drop(image)
            # From -1-1 to 0-1 because the stylegan2 expects values in the range of -1 to 1 but we operate in the range of 0 to 1.
            image = (image*0.5)+0.5

            batch,c,h,wid = image.shape

            if self.quantizing_strategy == "detach":
                
                weight = self.quantizing_detach(self.weight.permute(1,0))
                weight = self.weight_drop(weight)

            else:
                weight = self.weight.permute(1,0)

            out = torch.zeros((batch,1,self.latent_size)).to(image.device)

            r = image[:,0,:,:].view(-1,1,h*wid)
            b = image[:,2,:,:].view(-1,1,h*wid)
            g = image[:,1,:,:].view(-1,1,h*wid)
            out[:,:,:self.latent_size] = (r @ weight) + (g @ weight) + (b @ weight)
            out = out.squeeze(1)
         
            if self.verbose:
                # Plotting the mask every 10000 iterations just for debugging purposes.
                if iter % 10000 == 0 and iter != 0:
                     self.plotting_mask()
            return out, weight
