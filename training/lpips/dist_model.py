"""
    Refer to https://github.com/rosinality/stylegan2-pytorch/blob/master/lpips/dist_model.py
    Refer to https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/trainer.py
"""

from __future__ import absolute_import
import os
import torch
from collections import OrderedDict
from torch.autograd import Variable
from .base_model import BaseModel
from scipy.ndimage import zoom
from tqdm import tqdm
import numpy as np

from . import networks_basic as networks
from training import lpips as util


class exportModel(torch.nn.Module):
    def name(self):
        return self.model_name

    def initialize(
        self,
        model="net-lin",
        net="vgg",
        colorspace="Lab",
        pnet_rand=False,
        pnet_tune=False,
        model_path=None,
        use_gpu=True,
        printNet=False,
        spatial=False,
        is_train=False,
        lr=0.0001,
        beta1=0.5,
        version="0.1",
    ):

        self.model = model
        self.net = net
        self.is_train = is_train
        self.spatial = spatial
        self.use_gpu = use_gpu
        self.model_name = "%s [%s]" % (model, net)

        assert self.model == "net-lin"  # pretrained net + linear layer
        self.net = networks.PNetLin(
            pnet_rand=pnet_rand,
            pnet_tune=pnet_tune,
            pnet_type=net,
            use_dropout=True,
            spatial=spatial,
            version=version,
            lpips=True,
        )
        kw = {}
        if not use_gpu:
            kw["map_location"] = "cpu"
        if model_path is None:
            import inspect

            model_path = os.path.abspath(
                os.path.join(
                    inspect.getfile(self.initialize),
                    "..",
                    "weights/v%s/%s.pth" % (version, net),
                )
            )

        assert not is_train
        print("Loading model from: %s" % model_path)
        self.net.load_state_dict(torch.load(model_path, **kw), strict=False)
        self.net.eval()

        if printNet:
            print("---------- Networks initialized -------------")
            networks.print_network(self.net)
            print("-----------------------------------------------")

    def forward(self, in0, in1, retPerLayer=False):

        return self.net.forward(in0, in1, retPerLayer=retPerLayer)


class DistModel(BaseModel):
    def name(self):
        return self.model_name

    def initialize(
        self,
        model="net-lin",
        net="alex",
        colorspace="Lab",
        pnet_rand=False,
        pnet_tune=False,
        model_path=None,
        use_gpu=True,
        printNet=False,
        spatial=False,
        is_train=False,
        lr=0.0001,
        beta1=0.5,
        version="0.1",
        gpu_ids=[0],
    ):

        BaseModel.initialize(self, use_gpu=use_gpu, gpu_ids=gpu_ids)

        self.model = model
        self.net = net
        self.is_train = is_train
        self.spatial = spatial
        self.gpu_ids = gpu_ids
        self.model_name = "%s [%s]" % (model, net)

        if self.model == "net-lin":  # pretrained net + linear layer
            self.net = networks.PNetLin(
                pnet_rand=pnet_rand,
                pnet_tune=pnet_tune,
                pnet_type=net,
                use_dropout=True,
                spatial=spatial,
                version=version,
                lpips=True,
            )
            kw = {}
            if not use_gpu:
                kw["map_location"] = "cpu"
            if model_path is None:
                import inspect

                model_path = os.path.abspath(
                    os.path.join(
                        inspect.getfile(self.initialize),
                        "..",
                        "weights/v%s/%s.pth" % (version, net),
                    )
                )

            if not is_train:
                print("Loading model from: %s" % model_path)
                self.net.load_state_dict(torch.load(model_path, **kw), strict=False)

        elif self.model == "net":  # pretrained network
            self.net = networks.PNetLin(pnet_rand=pnet_rand, pnet_type=net, lpips=False)
        elif self.model in ["L2", "l2"]:
            self.net = networks.L2(
                use_gpu=use_gpu, colorspace=colorspace
            )  # not really a network, only for testing
            self.model_name = "L2"
        elif self.model in ["DSSIM", "dssim", "SSIM", "ssim"]:
            self.net = networks.DSSIM(use_gpu=use_gpu, colorspace=colorspace)
            self.model_name = "SSIM"
        else:
            raise ValueError("Model [%s] not recognized." % self.model)

        self.parameters = list(self.net.parameters())

        if self.is_train:  # training mode
            # extra network on top to go from distances (d0,d1) => predicted human judgment (h*)
            self.rankLoss = networks.BCERankingLoss()
            self.parameters += list(self.rankLoss.net.parameters())
            self.lr = lr
            self.old_lr = lr
            self.optimizer_net = torch.optim.Adam(
                self.parameters, lr=lr, betas=(beta1, 0.999)
            )
        else:  # test mode
            self.net.eval()

        if use_gpu:
            self.net.to(gpu_ids[0])
            self.net = torch.nn.DataParallel(self.net, device_ids=gpu_ids)
            if self.is_train:
                self.rankLoss = self.rankLoss.to(
                    device=gpu_ids[0]
                )  # just put this on GPU0

        if printNet:
            print("---------- Networks initialized -------------")
            networks.print_network(self.net)
            print("-----------------------------------------------")

    def forward(self, in0, in1, retPerLayer=False):


        return self.net.forward(in0, in1, retPerLayer=retPerLayer)

    # ***** training FUNCTIONS *****
    def optimize_parameters(self):
        self.forward_train()
        self.optimizer_net.zero_grad()
        self.backward_train()
        self.optimizer_net.step()
        self.clamp_weights()

    def clamp_weights(self):
        for module in self.net.modules():
            if hasattr(module, "weight") and module.kernel_size == (1, 1):
                module.weight.data = torch.clamp(module.weight.data, min=0)

    def set_input(self, data):
        self.input_ref = data["ref"]
        self.input_p0 = data["p0"]
        self.input_p1 = data["p1"]
        self.input_judge = data["judge"]

        if self.use_gpu:
            self.input_ref = self.input_ref.to(device=self.gpu_ids[0])
            self.input_p0 = self.input_p0.to(device=self.gpu_ids[0])
            self.input_p1 = self.input_p1.to(device=self.gpu_ids[0])
            self.input_judge = self.input_judge.to(device=self.gpu_ids[0])

        self.var_ref = Variable(self.input_ref, requires_grad=True)
        self.var_p0 = Variable(self.input_p0, requires_grad=True)
        self.var_p1 = Variable(self.input_p1, requires_grad=True)

    def forward_train(self):  # run forward pass
        # print(self.net.module.scaling_layer.shift)
        # print(torch.norm(self.net.module.net.slice1[0].weight).item(), torch.norm(self.net.module.lin0.model[1].weight).item())

        self.d0 = self.forward(self.var_ref, self.var_p0)
        self.d1 = self.forward(self.var_ref, self.var_p1)
        self.acc_r = self.compute_accuracy(self.d0, self.d1, self.input_judge)

        self.var_judge = Variable(1.0 * self.input_judge).view(self.d0.size())

        self.loss_total = self.rankLoss.forward(
            self.d0, self.d1, self.var_judge * 2.0 - 1.0
        )

        return self.loss_total

    def backward_train(self):
        torch.mean(self.loss_total).backward()

    def compute_accuracy(self, d0, d1, judge):
        """ d0, d1 are Variables, judge is a Tensor """
        d1_lt_d0 = (d1 < d0).cpu().data.numpy().flatten()
        judge_per = judge.cpu().numpy().flatten()
        return d1_lt_d0 * judge_per + (1 - d1_lt_d0) * (1 - judge_per)

    def get_current_errors(self):
        retDict = OrderedDict(
            [("loss_total", self.loss_total.data.cpu().numpy()), ("acc_r", self.acc_r)]
        )

        for key in retDict.keys():
            retDict[key] = np.mean(retDict[key])

        return retDict

    def get_current_visuals(self):
        zoom_factor = 256 / self.var_ref.data.size()[2]

        ref_img = util.tensor2im(self.var_ref.data)
        p0_img = util.tensor2im(self.var_p0.data)
        p1_img = util.tensor2im(self.var_p1.data)

        ref_img_vis = zoom(ref_img, [zoom_factor, zoom_factor, 1], order=0)
        p0_img_vis = zoom(p0_img, [zoom_factor, zoom_factor, 1], order=0)
        p1_img_vis = zoom(p1_img, [zoom_factor, zoom_factor, 1], order=0)

        return OrderedDict(
            [("ref", ref_img_vis), ("p0", p0_img_vis), ("p1", p1_img_vis)]
        )

    def save(self, path, label):
        if self.use_gpu:
            self.save_network(self.net.module, path, "", label)
        else:
            self.save_network(self.net, path, "", label)
        self.save_network(self.rankLoss.net, path, "rank", label)

    def update_learning_rate(self, nepoch_decay):
        lrd = self.lr / nepoch_decay
        lr = self.old_lr - lrd

        for param_group in self.optimizer_net.param_groups:
            param_group["lr"] = lr

        print("update lr [%s] decay: %f -> %f" % (type, self.old_lr, lr))
        self.old_lr = lr


def score_2afc_dataset(data_loader, func, name=""):


    d0s = []
    d1s = []
    gts = []

    for data in tqdm(data_loader.load_data(), desc=name):
        d0s += func(data["ref"], data["p0"]).data.cpu().numpy().flatten().tolist()
        d1s += func(data["ref"], data["p1"]).data.cpu().numpy().flatten().tolist()
        gts += data["judge"].cpu().numpy().flatten().tolist()

    d0s = np.array(d0s)
    d1s = np.array(d1s)
    gts = np.array(gts)
    scores = (d0s < d1s) * (1.0 - gts) + (d1s < d0s) * gts + (d1s == d0s) * 0.5

    return (np.mean(scores), dict(d0s=d0s, d1s=d1s, gts=gts, scores=scores))


def score_jnd_dataset(data_loader, func, name=""):


    ds = []
    gts = []

    for data in tqdm(data_loader.load_data(), desc=name):
        ds += func(data["p0"], data["p1"]).data.cpu().numpy().tolist()
        gts += data["same"].cpu().numpy().flatten().tolist()

    sames = np.array(gts)
    ds = np.array(ds)

    sorted_inds = np.argsort(ds)
    sames_sorted = sames[sorted_inds]

    TPs = np.cumsum(sames_sorted)
    FPs = np.cumsum(1 - sames_sorted)
    FNs = np.sum(sames_sorted) - TPs

    precs = TPs / (TPs + FPs)
    recs = TPs / (TPs + FNs)
    score = util.voc_ap(recs, precs)

    return (score, dict(ds=ds, sames=sames))
