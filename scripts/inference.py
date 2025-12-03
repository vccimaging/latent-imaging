import torch
import numpy as np
import sys
sys.path.append(".")
sys.path.append("..")
from PIL import Image
from options.train_options import TrainOptions
from models.psp import pSp
from models.pSp_o import pSp_o
import torchvision.transforms as trans
import glob

SAVE_TENSOR = False

id_transform = trans.Compose([
        trans.Resize((256, 256)),
		trans.ToTensor(),
		trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
	])


def main():
    opts = TrainOptions().parse()
    opts.device = 'cuda:0'
    
    if opts.generative_model == "stylegan":
        net = pSp(opts).to(opts.device)
    elif opts.generative_model == "stylegano":
        net = pSp_o(opts).to(opts.device)
    else:
        raise ValueError("Invalid generative model")
    net.eval()
    path_out = "paper_imgs/out/"
    with torch.no_grad():
        if opts.dataset_type == "ffhq_encode":
            path = "paper_imgs/gt_faces/"
        elif opts.dataset_type == "dog":
            path = "paper_imgs/gt_dogs/"
        elif opts.dataset_type == "cat":
            path = "paper_imgs/gt_cats/"
        elif opts.dataset_type == "real":
            path = "paper_imgs/measurements/"
        else:
            raise ValueError("Invalid dataset type")

        if opts.dataset_type != "real":
            images = sorted(glob.glob(path+"*.jpg"))
        else:
            images = sorted(glob.glob(path+"*.npz"))

        for i,image in enumerate(images):
            print("Reconstructing Image: ",i)
            if opts.dataset_type == "real":
                image = np.load(image)
                image = torch.from_numpy(image['array']).to(opts.device)
                out, _, _ = net.forward(image.unsqueeze(0), return_latents=True, global_step=0)
            else:
                image = Image.open(image)
                image = id_transform(image).unsqueeze(0).to(opts.device)
                out, _, _ = net.forward(image,return_latents=True, global_step=0)

            out = (((out*0.5)+0.5)*255).clamp(0,255)
            out = out.squeeze(0).to("cpu").permute(1,2,0).detach().numpy().astype(np.uint8)
    
            Image.fromarray(out).save(path_out+str(i)+".png")


if __name__ == '__main__':
	main()
