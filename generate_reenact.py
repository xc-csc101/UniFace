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
from training.model import Generator_32 as Generator
from training.model import Encoder_32 as Encoder
from training.pose import Encoder_Pose
from utils.flow_utils import flow_to_image, resize_flow

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


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
        padding=0
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

        self.e_ema_p = Encoder_Pose()

    def forward(self, input):
        src = input[0]
        drv = input[1]

        src_w = self.e_ema(src)
        flow, pose = self.e_ema_p(drv)
        fake_img = self.g_ema([src_w, flow])
        return src, drv, fake_img, resize_flow(flow, (256, 256))


if __name__ == "__main__":
    # python generate.py --ckpt expr/checkpoints/celeba_hq_256_8x8.pt --mixing_type local_editing --test_lmdb data/celeba_hq/LMDB_test --local_editing_part nose
    device = "cuda"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mixing_type",
        type=str,
        default='examples'
    )
    parser.add_argument("--inter", type=str, default='pair')
    parser.add_argument("--ckpt", type=str, default='session/reenactment/checkpoints/1000000.pt')
    parser.add_argument("--test_path", type=str, default='examples/img')
    parser.add_argument("--txt_path", type=str, default='examples/pair_reenact.txt')
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

    model = Model().to(device)
    model.g_ema.load_state_dict(ckpt["g_ema"])
    model.e_ema.load_state_dict(ckpt["e_ema"])
    model.e_ema_p.load_state_dict(ckpt["e_ema_p"])
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

    test_dataset = Dataset_for_test(args.test_path, mode='test', root_txt=args.txt_path, suffix='.jpg', transforms=transform)
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
        for i, (imgs, img_paths, pair) in enumerate(tqdm(loader, mininterval=1)):
            src_img = imgs[0].to(device)
            drv_img = imgs[1].to(device)

            filenames = img_paths[1]
            # print(filenames)

            img_s, img_d, img_r, flow = model([src_img, drv_img])

            for i_b, (ims, imd, imr, sn, f) in enumerate(zip(img_s, img_d, img_r, pair, flow)):
                # print(f'******{sn}')
                f = f.cpu().numpy().transpose([1, 2, 0])
                f_show = flow_to_image(f)

                save_tmp = f"{args.save_image_pair_dir}"
                os.makedirs(save_tmp, exist_ok=True)
                save_image_list(
                    [ims, imd, imr],
                    # f"{args.save_image_pair_dir}/{trg_img_n[i_b]}_{src_img_n[i_b]}.png",   
                    f"{save_tmp}/{sn}.png"                 
                )
                save_image(imr, f"{args.save_image_single_dir}/{sn}.png",)
                im = cv2.imread(f"{save_tmp}/{sn}.png")
                im_flow = np.hstack((im, f_show))
                cv2.imwrite(f"{save_tmp}/{sn}.png", im_flow)