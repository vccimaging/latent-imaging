# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""
import sys
sys.path.append(".")
sys.path.append("..")
import re
from typing import List, Tuple, Union

import models.styleganxl.dnnlib as dnnlib
import numpy as np
import torch

import models.styleganxl.legacy as legacy
#from models.stylegan2_official import gen_utils

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

def generate_images(network_pkl='pretrained_models/ffhq256.pkl',translate=parse_vec2('0,0'),opt=None,rotate=0):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        
        G = legacy.load_network_pkl(f)['G_ema']
        G = G.eval().requires_grad_(False).to(device)
    
    zs = torch.randn([10000, G.mapping.z_dim], device=device)
    cs = None

    w_stds = G.mapping(zs, cs)
    w_stds = w_stds.reshape(10, 1000, G.num_ws, -1)
    w_stds = w_stds.std(0).mean(0)[0]
    w_all_classes_avg = G.mapping.w_avg.mean(0)

    
    # Only For StyleganXL
    m = make_transform(translate, rotate)
    m = np.linalg.inv(m)
    G.synthesis.input.transform.copy_(torch.from_numpy(m))

    return G, G.mapping.w_avg, w_all_classes_avg 

# generate_images()
# #     # os.makedirs(outdir, exist_ok=True)

#     Generate images.
#     for seed_idx, seed in enumerate(seeds):
#         print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))

#         # Construct an inverse rotation/translation matrix and pass to the generator.  The
#         # generator expects this matrix as an inverse to avoid potentially failing numerical
#         # operations in the network.
#         if hasattr(G.synthesis, 'input'):
#             m = make_transform(translate, rotate)
#             m = np.linalg.inv(m)
#             G.synthesis.input.transform.copy_(torch.from_numpy(m))

#         w = gen_utils.get_w_from_seed(G, batch_sz, device, truncation_psi, seed=seed,
#                                       centroids_path=centroids_path, class_idx=class_idx)
#         #print(w[0,1,:]==w[0,2,:])

#         img = gen_utils.w_to_img(G, w, to_np=True)
#         PIL.Image.fromarray(gen_utils.create_image_grid(img), 'RGB').save(f'{outdir}/seed{seed:04d}.png')


# #----------------------------------------------------------------------------

# if __name__ == "__main__":
#     generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
