import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.config import Config
from nets.vgg import vgg as add_vgg
from nets.l2Norm import L2Norm
from nets.priorBox import PriorBox
from nets.detect import Detect

class SSD(nn.Module):
    def __init__(self, phase, base, extras, head, num_classes):
        '''
        forward输出：'test'阶段=检测结果；训练=loc,conf,priors

        :param phase: 阶段
        :param base: 构建vgg网络的基础
        :param extras: 构建额外网络的pytorch
        :param head(torch):
        loc:head[0]  loc_layers
        conf:head[1]  conf_layers
        :param num_classes: 类别总数
        '''
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = Config
        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        # 获得conv4_3的内容
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # 获得fc7的内容
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # 获得后面的内容
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # 添加回归层和分类层
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # 进行resize
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == 'test':
            # loc会resize到batch_size,num_anchors,4
            # conf会resize到batch_size,num_anchors,
            # output = self.detect(
            #     loc.view(loc.size(0), -1, 4),  # loc preds
            #     self.softmax(conf.view(conf.size(0), -1,
            #                            self.num_classes)),  # conf preds
            #     self.priors
            # )
            output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45,
                                       # PyTorch1.5.0 support new-style autograd function
                                       loc.view(loc.size(0), -1, 4),  # loc preds
                                       self.softmax(conf.view(conf.size(0), -1,
                                                              self.num_classes)),  # conf preds
                                       self.priors.type(type(x.data))  # default boxes
                                       )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output







def add_extras(i, batch_norm = False):
    layers = []
    in_channels = i

    # Block6
    # 1024x19x19 -> 512x10x10
    layers += [nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]

    # Block7
    # 512x10x10 -> 256x5x5
    layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

    # Block8
    # 256x5x5 -> 256x3x3
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]

    # Block 9
    # 256x3x3 -> 256x1x1
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]

    return layers

# 先验框个数
mbox = [4, 6, 6, 6, 4, 4]

def get_ssd(phase, num_classes):

    vgg, extra_layers = add_vgg(3), add_extras(1024)

    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        # vgg[21].outchannels, mbox[0]x4
        # vgg[-2].outchannels, mbox[1]x4
        loc_layers += [nn.Conv2d(vgg[v].out_channels, mbox[k] * 4,
                                 kernel_size=3, padding=1)]
        # vgg[21].outchannels, mbox[0]xnum_classes
        # vgg[-2].outchannels, mbox[1]xnum_classes
        conf_layers += [nn.Conv2d(vgg[v].out_channels, mbox[k] * num_classes,
                                  kernel_size=3, padding=1)]

    for k, v in enumerate(extra_layers[1::2], 2):
        # 从1到结束，每隔两个，index=2
        loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4,
                                 kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes,
                                  kernel_size=3, padding=1)]

    SSD_MODEL = SSD(phase, vgg, extra_layers, (loc_layers, conf_layers), num_classes)
    return SSD_MODEL