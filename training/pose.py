import torch.nn as nn
from torch.nn import init
import torch


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(
            'Network [%s] was created. Total number of parameters: %.1f million. '
            'To see the architecture, do print(network).' %
            (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1
                                           or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' %
                        init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


def weights_init_xavier(m, gain=0.02):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                 or classname.find('Linear') != -1):
        nn.init.xavier_normal_(m.weight.data, gain=gain)
        if hasattr(m, 'bias'):
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Encoder_Pose(BaseNetwork):
    def __init__(self,
                 norm_layer=nn.InstanceNorm2d,
                 use_dropout=False):
        super(Encoder_Pose, self).__init__()
        input_nc = 3
        output_nc = 2
        num_downs = 8
        ngf = 64
        embedding_dim = 512

        self.num_downs = num_downs
        use_bias = norm_layer == nn.InstanceNorm2d

        # ===   down sample   === #
        self.down0 = nn.Sequential(
            nn.Conv2d(input_nc,
                      ngf * 1,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=use_bias), nn.LeakyReLU(0.2, True), norm_layer(ngf))
        nf_mult = 1
        for i in range(1, num_downs - 1):
            nf_mult_prev = nf_mult
            nf_mult = min(2 * nf_mult_prev, 8)
            layer = nn.Sequential(
                nn.Conv2d(ngf * nf_mult_prev,
                          ngf * nf_mult,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=use_bias), nn.LeakyReLU(0.2, True),
                norm_layer(ngf * nf_mult))
            setattr(self, 'down' + str(i), layer)
        self.down7 = nn.Sequential(
            nn.Conv2d(ngf * nf_mult,
                      embedding_dim,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=use_bias))

        # ===   up sample   === #
        nf_mult = 1
        for i in range(1, num_downs - 1):
            nf_mult_prev = nf_mult
            nf_mult = min(2 * nf_mult_prev, 8)
            layer = nn.Sequential(
                nn.ReLU(True), nn.Upsample(scale_factor=2.0, mode='bilinear'),
                nn.Conv2d(ngf * nf_mult,
                          ngf * nf_mult_prev,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=use_bias), norm_layer(ngf * nf_mult_prev))
            setattr(self, 'up' + str(i), layer)
        self.up7 = nn.Sequential(
            nn.ReLU(True), nn.Upsample(scale_factor=2.0, mode='bilinear'),
            nn.Conv2d(embedding_dim,
                      ngf * nf_mult,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=use_bias), norm_layer(ngf * nf_mult))

        self.out2 = nn.Sequential(
            nn.ReLU(False), nn.Upsample(scale_factor=2.0, mode='bilinear'),
            nn.Conv2d(ngf * 4,
                      output_nc,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=use_bias))
        self.tanh = nn.Tanh()

    def forward(self, x):
        # === down sampling === #
        down0 = self.down0(x)
        down1 = self.down1(down0)
        down2 = self.down2(down1)
        hid = down2
        for i in range(3, self.num_downs):
            hid = getattr(self, 'down' + str(i))(hid)
            pose_code = hid

        # === up sampling === #
        for i in range(3, self.num_downs)[::-1]:
            hid = getattr(self, 'up' + str(i))(hid)
        up_3 = hid

        out_64 = self.out2(up_3)

        return self.tanh(out_64), pose_code.flatten(1)

    def init_weights(self, init_type='normal', gain=0.02):
        self.apply(weights_init_xavier)   


class Encoder_Pose_v2(BaseNetwork):
    # not encoder to vector
    def __init__(self,
                 norm_layer=nn.InstanceNorm2d,
                 use_dropout=False):
        super(Encoder_Pose_v2, self).__init__()
        input_nc = 3
        output_nc = 2
        num_downs = 5
        ngf = 64

        self.num_downs = num_downs
        use_bias = norm_layer == nn.InstanceNorm2d

        # ===   down sample   === #
        self.down0 = nn.Sequential(
            nn.Conv2d(input_nc,
                      ngf * 1,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=use_bias), nn.LeakyReLU(0.2, True), norm_layer(ngf))
        nf_mult = 1
        for i in range(1, num_downs):
            nf_mult_prev = nf_mult
            nf_mult = min(2 * nf_mult_prev, 8)
            layer = nn.Sequential(
                nn.Conv2d(ngf * nf_mult_prev,
                          ngf * nf_mult,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=use_bias), nn.LeakyReLU(0.2, True),
                norm_layer(ngf * nf_mult))
            setattr(self, 'down' + str(i), layer)

        self.trans1 = nn.Sequential(
                nn.Conv2d(ngf * nf_mult,
                          ngf * nf_mult,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=use_bias), nn.LeakyReLU(0.2, True),
                norm_layer(ngf * nf_mult))

        self.trans2 = nn.Sequential(
                nn.Conv2d(ngf * nf_mult,
                          ngf * nf_mult,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=use_bias), nn.LeakyReLU(0.2, True),
                norm_layer(ngf * nf_mult))

        # ===   up sample   === #
        nf_mult = 2
        for i in range(2, num_downs):
            nf_mult_prev = nf_mult
            nf_mult = min(2 * nf_mult_prev, 8)
            layer = nn.Sequential(
                nn.ReLU(True), nn.Upsample(scale_factor=2.0, mode='bilinear'),
                nn.Conv2d(ngf * nf_mult,
                          ngf * nf_mult_prev,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=use_bias), norm_layer(ngf * nf_mult_prev))
            setattr(self, 'up' + str(i), layer)

        self.out2 = nn.Sequential(
            nn.ReLU(False), nn.Upsample(scale_factor=2.0, mode='bilinear'),
            nn.Conv2d(ngf * 4,
                      output_nc,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=use_bias))
        self.tanh = nn.Tanh()

    def forward(self, x):
        # === down sampling === #
        down0 = self.down0(x)
        down1 = self.down1(down0)
        down2 = self.down2(down1)
        hid = down2
        for i in range(3, self.num_downs):
            hid = getattr(self, 'down' + str(i))(hid)
            pose_code = hid

        # trans
        hid = self.trans2(self.trans1(hid))

        # === up sampling === #
        for i in range(3, self.num_downs)[::-1]:
            hid = getattr(self, 'up' + str(i))(hid)
        up_3 = hid

        out_64 = self.out2(up_3)

        return self.tanh(out_64), pose_code.flatten(1)

    def init_weights(self, init_type='normal', gain=0.02):
        self.apply(weights_init_xavier)   


if __name__ == '__main__':
    net = Encoder_Pose_v2()
    img = torch.randn(1, 3, 256, 256)
    out = net(img)
    print(out.shape)
