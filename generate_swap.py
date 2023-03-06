"""
StyleMapGAN
Copyright (c) 2021-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import argparse
import pickle
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import utils, transforms
import numpy as np
from torchvision.datasets import ImageFolder
from training.dataset import *
from scipy import linalg
import random
import time
import os
from tqdm import tqdm
from copy import deepcopy
import cv2
from PIL import Image
from itertools import combinations
# need to modify
from training.model import Generator_globalatt_return_32 as Generator
from training.model import Encoder_return_32 as Encoder

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

cmap = np.array([(0, 0, 0), (255, 0, 0), (76, 153, 0),
                 (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
                 (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
                 (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153),
                 (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)],
                dtype=np.uint8)

class Colorize(object):
    def __init__(self, n=19):
        self.cmap = cmap
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

def save_image(img, path, normalize=True, range=(-1, 1)):
    utils.save_image(
        img,
        path,
        normalize=normalize,
        range=range,
    )

def save_image_list(img, path, normalize=True, range=(-1, 1)):
    nrow = len(img)
    utils.save_image(
        img,
        path,
        nrow=nrow,
        normalize=normalize,
        range=range,
    )

def save_images(imgs, paths, normalize=True, range=(-1, 1)):
    for img, path in zip(imgs, paths):
        save_image(img, path, normalize=normalize, range=range)


def make_noise(batch, latent_channel_size, device):
    return torch.randn(batch, latent_channel_size, device=device)


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


class Model(nn.Module):
    def __init__(self, device="cuda"):
        super(Model, self).__init__()
        self.g_ema = Generator(
            args.size,
            args.latent_channel_size,
            args.latent_spatial_size,
            lr_mul=args.lr_mul,
            channel_multiplier=args.channel_multiplier,
            normalize_mode=args.normalize_mode,
            small_generator=args.small_generator,
        )
        self.e_ema = Encoder(
            args.size,
            args.latent_channel_size,
            args.latent_spatial_size,
            channel_multiplier=args.channel_multiplier,
        )
        
    def tensor2label(self, label_tensor, n_label):
        label_tensor = label_tensor.cpu().float()
        if label_tensor.size()[0] > 1:
            label_tensor = label_tensor.max(0, keepdim=True)[1]
        label_tensor = Colorize(n_label)(label_tensor)
        label_numpy = label_tensor.numpy()

        return label_numpy


    def forward(self, input):
        trg = input[0]
        src = input[1]

        trg_src = torch.cat([trg, src], dim=0)
        # w = self.e_ema(trg_src)

        w, w_feat = self.e_ema(trg_src)
        w_feat_tgt = [torch.chunk(f, 2, dim=0)[0] for f in w_feat][::-1]

        trg_w, src_w = torch.chunk(w, 2, dim=0)

        fake_img = self.g_ema([trg_w, src_w, w_feat_tgt])


        return trg, src, fake_img


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mixing_type",
        type=str,
        default='examples'
    )
    parser.add_argument("--inter", type=str, default='pair')
    parser.add_argument("--ckpt", type=str, default='session/swap/checkpoints/500000.pt')
    parser.add_argument("--test_path", type=str, default='examples/img/')
    parser.add_argument("--test_txt_path", type=str, default='examples/pair_swap.txt')
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--save_image_dir", type=str, default="expr")

    args = parser.parse_args()

    ckpt = torch.load(args.ckpt)
    train_args = ckpt["train_args"]
    for key in vars(train_args):
        if not (key in vars(args)):
            setattr(args, key, getattr(train_args, key))
    print(args)

    dataset_name = args.inter
    args.save_image_pair_dir = os.path.join(
        args.save_image_dir, args.mixing_type, dataset_name, 'pair'
    )
    os.makedirs(args.save_image_pair_dir, exist_ok=True)

    args.save_image_single_dir = os.path.join(
        args.save_image_dir, args.mixing_type, dataset_name, 'single'
    )
    os.makedirs(args.save_image_single_dir, exist_ok=True)

    model = Model().half().to(device)
    model.g_ema.load_state_dict(ckpt["g_ema"])
    model.e_ema.load_state_dict(ckpt["e_ema"])
    model.eval()

    batch = args.batch

    device = "cuda"
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    test_dataset = SwapTestTxtDataset(args.test_path, args.test_txt_path, transform, suffix='.jpg')
    n_sample = len(test_dataset)
    sampler = data_sampler(test_dataset, shuffle=False)

    loader = data.DataLoader(
        test_dataset,
        batch,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    with torch.no_grad():
        for i, ([trg_img, trg_name],[src_img, src_name]) in enumerate(tqdm(loader, mininterval=1)):
            trg_img = trg_img.half().to(device)
            src_img = src_img.half().to(device)
            trg_img_n = trg_name
            src_img_n = src_name

            img_t, img_s, img_r1 = model([trg_img, src_img])

            for i_b, (imt, ims, imr1) in enumerate(zip(img_t, img_s, img_r1)):

                save_image_list(
                    [imt, ims, imr1],
                    f"{args.save_image_pair_dir}/{trg_img_n[i_b]}_{src_img_n[i_b]}.jpg",                    
                )
                # imr1_resize = F.interpolate(imr1.unsqueeze(0), (1024, 1024)).squeeze(0)
                save_image(imr1, f"{args.save_image_single_dir}/{trg_img_n[i_b]}.jpg")

