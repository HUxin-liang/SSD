import torch
import numpy as np
from math import sqrt

class PriorBox(object):
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        # 每层feature map的大小
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, v in enumerate(self.feature_maps):
            # [38, 19, 10, 5, 3, 1]
            # 生成38x38大小的feature map
            x, y = np.meshgrid(np.arange(v), np.arange(v))
            # x = [0-37]
            x = x.reshape(-1)
            y = y.reshape(-1)
            for i, j in zip(y, x):
                # steps : 感受野 [8, 16, 32, 64, 100, 300]
                # 300 / 8 = 37.5 是对应的feature, 目的是为了下面还原坐标，即中心坐标转为小数形式
                f_k = self.image_size / self.steps[k]
                # 计算网格中心 即 某层先验框中心点/该层feature大小=原始图像中，先验框中心在图片的比例
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                # 先验框的短边占原图的比例
                # 38x38 featuremap, min_size=30, 30/300=0.1,
                # 19x19 featuremap, min_size=60, 60/300=0.2,
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]
                # 长边所占原图的比例
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 长方形
                for ar in self.aspect_ratios[k]:
                    # aspect_ratios : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        # 获得所有的先验框
        output = torch.Tensor(mean).view(-1, 4)

        # 梯度剪枝
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

