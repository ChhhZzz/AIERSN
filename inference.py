"""
This file runs the main training/val loop
"""

import json
import os
import pprint
import random
import shutil
import sys
import time
import warnings

import numpy as np
import torchvision.transforms as transforms
import cv2
from torchvision.utils import save_image
from configs import transforms_config
import torch
from torch.utils.data import DataLoader
from datasets.images_dataset import ImagesDataset
from models.main_network_4tst import styleSketch
from utils.change_w_layer import get_mix_latent

sys.path.append(".")
sys.path.append("..")

from options.tst_options import TstOptions  # noqa: E402
from training.coach import Coach  # noqa: E402


warnings.filterwarnings("ignore")


def main():

    # Fix random seed
    SEED = 2107456
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    opts = TstOptions().parse()

    opts.w = 'avg'
    opts.device = 'cuda:0'
    if os.path.exists(opts.exp_dir):
        shutil.rmtree(opts.exp_dir)
    os.makedirs(opts.exp_dir, exist_ok=True)
    opts_dict = vars(opts)
    pprint.pprint(opts_dict)

    net = styleSketch(opts)
    print(f'load weights from {opts.checkpoint_path}')
    ckpt = torch.load(opts.checkpoint_path, map_location='cpu')
    net.spatialNet.load_state_dict(ckpt['model'])
    net.to('cuda')

    transform_config = transforms_config.EncodeTransforms(opts)
    transform = transform_config.get_transforms()["transform_gt_train"]
    dataset = ImagesDataset(
        data_path=opts.data_path,
        transform=transform,
        opts=opts,
        use_w=False
    )
    dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=False, drop_last=False)

    net.spatialNet.use_w = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]

    start = time.time()
    for idx, img in enumerate(dataloader):
        with torch.no_grad():

            img = img.cuda()

            spatial_res, _ = net.spatialNet(img)
            w_avg = (net.latent_avg).unsqueeze(0).repeat(14, 1).unsqueeze(0)

            w_invert = net.w_encoder(img)
            if w_invert.ndim == 2:
                w_invert = w_invert + net.latent_avg.repeat(w_invert.shape[0], 1, 1)[:, 0, :]
            else:
                w_invert = w_invert + net.latent_avg.repeat(w_invert.shape[0], 1, 1)

            w_mix = get_mix_latent(w_avg, w_invert, net.spatialNet.use_w)
            g_img, _ = net.Generator.synthesis(w_mix, spatial_res, noise_mode="const")
            save_image(g_img[-1], f'inference/{str(idx).zfill(5)}.jpg', normalize=True, nrow=opts.batch_size)

    end = time.time()
    print(end-start, (end-start)/100.)




if __name__ == "__main__":
    main()
