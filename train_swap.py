"""
StyleMapGAN
Copyright (c) 2021-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
import argparse
import yaml
import matplotlib.pyplot as plt
from torch import autograd, nn, optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data
from torchvision import transforms
import torchvision
from tqdm import tqdm

from training import lpips
from training.model import Discriminator, Encoder_return_32, Generator_globalatt_flow
from training.pose import Encoder_Pose
from training.dataset import *
from training.vgg import VGG
from criteria import id_loss
from criteria.cx_style_loss import CXLoss
from utils import common

torch.backends.cudnn.benchmark = True


def log_images(step, logdir, name, im_data, subscript=None, log_latest=False):
    fig = common.vis_faces(im_data)
    if log_latest:
        step = 0
    if subscript:
        path = os.path.join(logdir, name, '{}_{:04d}.jpg'.format(subscript, step))
    else:
        path = os.path.join(logdir, name, '{:04d}.jpg'.format(step))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def parse_and_log_images(step, logdir, x, y, y_hat, title, subscript=None, display_count=2):
    im_data = []
    for i in range(display_count):
        cur_im_data = {
            'input_face': common.log_input_image(x[i]),
            'target_face': common.tensor2im(y[i]),
            'output_face': common.tensor2im(y_hat[i]),
        }
        im_data.append(cur_im_data)
    log_images(step, logdir, title, im_data=im_data, subscript=subscript)


def save_args(path, args):
    args_dict = args.__dict__
    with open(path, 'w') as f:
        yaml.dump(args_dict, f)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def gather_grad(params, world_size):
    for param in params:
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data.div_(world_size)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    with torch.no_grad():
        par1 = dict(model1.named_parameters())
        par2 = dict(model2.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def copy_norm_params(model_tgt, model_src):
    with torch.no_grad():
        src_state_dict = model_src.state_dict()
        tgt_state_dict = model_tgt.state_dict()
        names = [name for name, _ in model_tgt.named_parameters()]

        for n in names:
            del src_state_dict[n]

        tgt_state_dict.update(src_state_dict)
        model_tgt.load_state_dict(tgt_state_dict)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    (grad_real,) = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def make_noise(batch, latent_channel_size, device):
    return torch.randn(batch, latent_channel_size, device=device)


class DDPModel(nn.Module):
    def __init__(self, device, args):
        super(DDPModel, self).__init__()

        self.generator = Generator_globalatt_flow(
            args.size,
            args.latent_channel_size,
            args.latent_spatial_size,
            lr_mul=args.lr_mul,
            channel_multiplier=args.channel_multiplier,
            normalize_mode=args.normalize_mode,
            small_generator=args.small_generator
        )
        self.g_ema = Generator_globalatt_flow(
            args.size,
            args.latent_channel_size,
            args.latent_spatial_size,
            lr_mul=args.lr_mul,
            channel_multiplier=args.channel_multiplier,
            normalize_mode=args.normalize_mode,
            small_generator=args.small_generator
        )

        self.discriminator = Discriminator(
            args.size, channel_multiplier=args.channel_multiplier
        )

        self.encoder = Encoder_return_32(
            args.size,
            args.latent_channel_size,
            args.latent_spatial_size,
            in_ch=3,
            channel_multiplier=args.channel_multiplier,
        )

        self.e_ema = Encoder_return_32(
            args.size,
            args.latent_channel_size,
            args.latent_spatial_size,
            in_ch=3,
            channel_multiplier=args.channel_multiplier,
        )

        self.encoder_p = Encoder_Pose()
        self.e_ema_p = Encoder_Pose()

        self.l1_loss = nn.L1Loss(size_average=True)
        self.mse_loss = nn.MSELoss(size_average=True)
        self.id_loss = id_loss.IDLoss().eval()
        self.percept = lpips.exportPerceptualLoss(
            model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
        )
        # cx loss
        self.vgg = VGG()
        self.vgg.load_state_dict(torch.load('pretrained_models/vgg19_conv.pth'))
        requires_grad(self.vgg, False)
        print('vgg for cx loss ready')

        self.cx_loss = CXLoss(sigma=0.5)

        self.device = device
        self.args = args

    def calc_cx(self, real, fake):
        style_layer = ['r32', 'r42']
        vgg_style = self.vgg(real, style_layer)
        vgg_fake = self.vgg(fake, style_layer)
        cx_style_loss = 0

        for i, val in enumerate(vgg_fake):
            cx_style_loss += self.cx_loss(vgg_style[i], vgg_fake[i])
        return cx_style_loss

    def forward(self, real_img, mode):
        if mode == "D":
            # real img is list
            with torch.no_grad():
                trg = real_img[0]
                src = real_img[1]

                trg_src = torch.cat([trg, src], dim=0)
                w, w_feat = self.encoder(trg_src)

                w_feat_tgt = [torch.chunk(f, 2, dim=0)[0] for f in w_feat][::-1]

                trg_w, src_w = torch.chunk(w, 2, dim=0)

                flow, _ = self.encoder_p(trg)

                fake_img = self.generator([trg_w, src_w, w_feat_tgt, flow])

            real_pred = self.discriminator(trg)
            fake_pred = self.discriminator(fake_img)
            d_loss = d_logistic_loss(real_pred, fake_pred)

            return (
                d_loss,
                real_pred.mean(),
                fake_pred.mean(),
            )

        elif mode == "D_reg":
            # real_img is tensor
            real_img.requires_grad = True
            real_pred = self.discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)
            d_reg_loss = (
                self.args.r1 / 2 * r1_loss * self.args.d_reg_every + 0 * real_pred[0]
            )

            return d_reg_loss, r1_loss

        elif mode == "E_x_rec":
            # real img is list
            trg = real_img[0]
            src = real_img[1]
            same = real_img[2]

            trg_src = torch.cat([trg, src], dim=0)
            w, w_feat = self.encoder(trg_src)
            w_feat_tgt = [torch.chunk(f, 2, dim=0)[0] for f in w_feat][::-1]

            trg_w, src_w = torch.chunk(w, 2, dim=0)

            flow, _ = self.encoder_p(trg)

            fake_img = self.generator([trg_w, src_w, w_feat_tgt, flow])

            # l2 loss for same id
            same = same.unsqueeze(-1).unsqueeze(-1)
            same = same.expand(trg.shape)

            x_rec_loss = self.mse_loss(torch.mul(trg, same), torch.mul(fake_img, same))
            perceptual_loss = self.percept(trg, fake_img).mean()

            # id loss
            id_loss, sim_improvement, id_logs = self.id_loss(fake_img, src, trg)

            # contextual loss
            cx_loss = self.calc_cx(trg, fake_img)

            fake_pred_from_E = self.discriminator(fake_img)
            indomainGAN_E_loss = F.softplus(-fake_pred_from_E).mean()

            return x_rec_loss, perceptual_loss, indomainGAN_E_loss, id_loss, cx_loss, fake_img

        elif mode == "cal_mse_lpips":
            # real img is list
            trg = real_img[0]
            src = real_img[1]
            same = real_img[2]

            trg_src = torch.cat([trg, src], dim=0)
            w, w_feat = self.e_ema(trg_src)
            w_feat_tgt = [torch.chunk(f, 2, dim=0)[0] for f in w_feat][::-1]

            trg_w, src_w = torch.chunk(w, 2, dim=0)

            flow, _ = self.e_ema_p(trg)
            fake_img = self.g_ema([trg_w, src_w, w_feat_tgt, flow])

            same = same.unsqueeze(-1).unsqueeze(-1)
            same = same.expand(trg.shape)

            x_rec_loss = self.mse_loss(torch.mul(trg, same), torch.mul(fake_img, same))
            perceptual_loss = self.percept(trg, fake_img).mean()
            cx_loss = self.calc_cx(trg, fake_img)

            return x_rec_loss, perceptual_loss, cx_loss, fake_img


def run(ddp_fn, world_size, args):
    print("world size", world_size)
    mp.spawn(ddp_fn, args=(world_size, args), nprocs=world_size, join=True)


def ddp_main(rank, world_size, args):
    print(f"Running DDP model on rank {rank}.")
    setup(rank, world_size)
    map_location = f"cuda:{rank}"
    torch.cuda.set_device(map_location)

    if args.ckpt:  # ignore current arguments
        ckpt = torch.load(args.ckpt, map_location=map_location)
        ckpt_p = torch.load('session/reenactment/checkpoints/500000.pt', map_location=map_location)

        train_args = ckpt["train_args"]
        print("load model:", args.ckpt)
        train_args.start_iter = int(args.ckpt.split("/")[-1].replace(".pt", ""))
        print(f"continue training from {train_args.start_iter} iter")
        args.ckpt = True
        args.start_iter = train_args.start_iter
    else:
        args.start_iter = 0

    # create model and move it to GPU with id rank
    model = DDPModel(device=map_location, args=args).to(map_location)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model.train()

    ## let loss model in eval mode
    model.module.id_loss.eval()
    model.module.percept.eval()

    g_module = model.module.generator
    g_ema_module = model.module.g_ema
    g_ema_module.eval()
    accumulate(g_ema_module, g_module, 0)

    e_module = model.module.encoder
    e_ema_module = model.module.e_ema
    e_ema_module.eval()
    accumulate(e_ema_module, e_module, 0)

    e_module_p = model.module.encoder_p
    e_ema_module_p = model.module.e_ema_p
    e_ema_module_p.eval()
    accumulate(e_ema_module_p, e_module_p, 0)

    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    r1_val = 0
    g_optim = optim.Adam(
        g_module.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )

    d_optim = optim.Adam(
        model.module.discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    e_optim = optim.Adam(
        e_module.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )

    e_optim_p = optim.Adam(
        e_module_p.parameters(),
        lr=args.lr * 0.01,
        betas=(0, 0.99),
    )

    accum = 0.999

    if args.ckpt:
        model.module.generator.load_state_dict(ckpt["generator"])
        model.module.discriminator.load_state_dict(ckpt["discriminator"])
        model.module.g_ema.load_state_dict(ckpt["g_ema"])

        model.module.encoder.load_state_dict(ckpt["encoder"])
        model.module.e_ema.load_state_dict(ckpt["e_ema"])

        # load pose encoder
        print('load pose')
        model.module.encoder_p.load_state_dict(ckpt_p["encoder_p"])
        model.module.e_ema_p.load_state_dict(ckpt_p["e_ema_p"])


    transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])

    save_dir = os.path.join("./session", args.checkname)
    os.makedirs(save_dir, 0o777, exist_ok=True)
    os.makedirs(save_dir + "/checkpoints", 0o777, exist_ok=True)
    os.makedirs(save_dir + "/imgs", 0o777, exist_ok=True)
    save_args(os.path.join('session',args.checkname,'args.yaml') , args)

    train_dataset = SwapTrainDataset(args.train_img_path, transform)
    val_dataset = SwapValDataset(args.val_img_path, transform)

    print(f"train_dataset: {len(train_dataset)}, val_dataset: {len(val_dataset)}")

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_per_gpu,
        drop_last=True,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_per_gpu,
        drop_last=True,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    train_loader = sample_data(train_loader)
    pbar = range(args.start_iter, args.iter)
    pbar = tqdm(pbar, initial=args.start_iter, mininterval=1)

    epoch = -1
    gpu_group = dist.new_group(list(range(args.ngpus)))

    for i in pbar:
        if i > args.iter:
            print("Done!")
            break
        elif i % (len(train_dataset) // args.batch) == 0:
            epoch += 1
            val_sampler.set_epoch(epoch)
            train_sampler.set_epoch(epoch)
            print("epoch: ", epoch)

        trg_img, src_img, same = next(train_loader)
        trg_img = trg_img.to(map_location)
        src_img = src_img.to(map_location)
        same = same.to(map_location)

        requires_grad(model.module.discriminator, True)
        # D adv
        d_loss, real_score, fake_score = model([trg_img, src_img], "D")
        d_loss = d_loss.mean()

        d_optim.zero_grad()

        (
            d_loss * args.lambda_d_loss
        ).backward()
        d_optim.step()

        # D reg

        d_regularize = i % args.d_reg_every == 0
        if d_regularize:
            d_reg_loss, r1_loss = model(trg_img, "D_reg") # r1 loss
            d_reg_loss = d_reg_loss.mean()
            d_optim.zero_grad()
            d_reg_loss.backward()
            d_optim.step()
            r1_val = r1_loss.mean().item()

        requires_grad(model.module.discriminator, False)

        # E_x_rec,
        x_rec_loss, perceptual_loss, indomainGAN_E_loss, id_loss, cx_loss, fake_img = model([trg_img, src_img, same], "E_x_rec")

        x_rec_loss = x_rec_loss.mean() * args.lambda_x_rec_loss
        perceptual_loss = perceptual_loss.mean() * args.lambda_perceptual_loss
        indomainGAN_E_loss = indomainGAN_E_loss.mean() * args.lambda_indomainGAN_E_loss
        id_loss = id_loss * args.lambda_id_loss
        cx_loss = cx_loss * args.lambda_cx_loss

        e_optim.zero_grad()
        g_optim.zero_grad()
        e_optim_p.zero_grad()

        encoder_loss = (
            x_rec_loss
            + perceptual_loss
            + indomainGAN_E_loss
            + id_loss
            + cx_loss
        )

        encoder_loss.backward()

        e_optim.step()
        g_optim.step()
        e_optim_p.step()

        pbar.set_description(
            (f"g: {indomainGAN_E_loss.item():.4f}; d: {d_loss.item():.4f}; r1: {r1_val:.4f}; rec: {x_rec_loss.item():.4f}; id: {id_loss.item():.4f}; cx: {cx_loss.item():.4f}; per: {perceptual_loss.item():.4f}")
        )

        # log image
        if rank == 0:
            if i % args.image_interval == 0 or (i < 1000 and i % 25 == 0):
                parse_and_log_images(i, save_dir, trg_img, src_img, fake_img, title='imgs/train/')

        with torch.no_grad():
            accumulate(g_ema_module, g_module, accum)
            accumulate(e_ema_module, e_module, accum)
            accumulate(e_ema_module_p, e_module_p, accum)

            if i % args.save_img_interval == 0:
                copy_norm_params(g_ema_module, g_module)
                copy_norm_params(e_ema_module, e_module)
                copy_norm_params(e_ema_module_p, e_module_p)

                x_rec_loss_avg, perceptual_loss_avg, cx_loss_avg = 0, 0, 0
                iter_num = 0
                for (trg_img, src_img, same) in tqdm(val_loader):
                    trg_img = trg_img.to(map_location)
                    src_img = src_img.to(map_location)
                    same = same.to(map_location)

                    x_rec_loss, perceptual_loss, cx_loss, fake_img = model([trg_img, src_img, same], "cal_mse_lpips")

                    x_rec_loss_avg += x_rec_loss.mean()
                    perceptual_loss_avg += perceptual_loss.mean()
                    cx_loss_avg += cx_loss.mean()
                    iter_num += 1

                    # log images
                    if rank == 0:
                        parse_and_log_images(i, save_dir, trg_img, src_img, fake_img, title='imgs/test/')

                x_rec_loss_avg /= iter_num
                perceptual_loss_avg /= iter_num
                cx_loss_avg /= iter_num

                dist.reduce(
                    x_rec_loss_avg, dst=0, op=dist.ReduceOp.SUM, group=gpu_group
                )
                dist.reduce(
                    perceptual_loss_avg,
                    dst=0,
                    op=dist.ReduceOp.SUM,
                    group=gpu_group,
                )
                if i % args.save_network_interval == 0:
                    if rank == 0:
                        x_rec_loss_avg = x_rec_loss_avg / args.ngpus
                        perceptual_loss_avg = perceptual_loss_avg / args.ngpus
                        x_rec_loss_avg_val = x_rec_loss_avg.item()
                        perceptual_loss_avg_val = perceptual_loss_avg.item()
                        cx_loss_avg_val = cx_loss_avg.item()

                        print(
                            f"x_rec_loss_avg: {x_rec_loss_avg_val}, perceptual_loss_avg: {perceptual_loss_avg_val}, cx_loss_avg: {cx_loss_avg_val}"
                        )
                        torch.save(
                            {
                                "generator": model.module.generator.state_dict(),
                                "discriminator": model.module.discriminator.state_dict(),
                                "encoder": model.module.encoder.state_dict(),
                                "encoder_p": model.module.encoder_p.state_dict(),
                                "g_ema": g_ema_module.state_dict(),
                                "e_ema": e_ema_module.state_dict(),
                                "e_ema_p": e_ema_module_p.state_dict(),
                                "train_args": args,
                            },
                            f"{save_dir}/checkpoints/{str(i).zfill(6)}.pt",
                        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkname", type=str, default='exp-swap-id2.5-cx0.5')
    parser.add_argument("--describ", type=str, default='no')
    parser.add_argument("--train_img_path", type=str, default='data/CelebA-HQ/train/images/')
    parser.add_argument("--val_img_path", type=str, default='data/CelebA-HQ/val/images/')
    parser.add_argument(
        "--dataset",
        type=str,
        default="danbooru",
        choices=[
            "celeba_hq",
            "afhq",
            "ffhq",
            "lsun/church_outdoor",
            "lsun/car",
            "lsun/bedroom",
        ],
    )
    parser.add_argument("--iter", type=int, default=500000)
    parser.add_argument("--save_network_interval", type=int, default=5000)
    parser.add_argument("--save_img_interval", type=int, default=1000)
    parser.add_argument("--small_generator", action="store_true")
    parser.add_argument("--batch", type=int, default=8, help="total batch sizes")
    parser.add_argument("--size", type=int, choices=[128, 256, 512, 1024], default=256)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_mul", type=float, default=1)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--latent_channel_size", type=int, default=512)
    parser.add_argument("--latent_spatial_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_interval", type=int, default=50)
    parser.add_argument(
        "--normalize_mode",
        type=str,
        choices=["LayerNorm", "InstanceNorm2d", "BatchNorm2d", "GroupNorm"],
        default="LayerNorm",
    )
    parser.add_argument("--mapping_layer_num", type=int, default=8)

    parser.add_argument("--lambda_x_rec_loss", type=float, default=1)
    parser.add_argument("--lambda_d_loss", type=float, default=1)
    parser.add_argument("--lambda_id_loss", type=float, default=2.5)
    parser.add_argument("--lambda_cx_loss", type=float, default=0.5)
    parser.add_argument("--lambda_perceptual_loss", type=float, default=1)
    parser.add_argument("--lambda_indomainGAN_D_loss", type=float, default=1)
    parser.add_argument("--lambda_indomainGAN_E_loss", type=float, default=1)

    input_args = parser.parse_args()
    ngpus = 2
    print("{} GPUS!".format(ngpus))

    assert input_args.batch % ngpus == 0
    input_args.batch_per_gpu = input_args.batch // ngpus
    input_args.ngpus = ngpus
    print("{} batch per gpu!".format(input_args.batch_per_gpu))

    run(ddp_main, ngpus, input_args)