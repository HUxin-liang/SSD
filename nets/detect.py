import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torch.autograd import Function
from utils.config import Config
from utils.box_utils import decode, nms
import numpy as np
import warnings

warnings.filterwarnings("ignore")

class Detect(Function):
    # def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
    #     self.num_classes = num_classes
    #     self.backgroud_label = bkg_label
    #     self.top_k = top_k
    #     self.nms_thresh = nms_thresh
    #     if nms_thresh <= 0:
    #         raise ValueError('nms_threshold must be non negative')
    #     self.conf_thresh = conf_thresh
    #     self.variance = Config['variance']

    @staticmethod
    def forward(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh, loc_data, conf_data, prior_data):
        self.num_classes = num_classes
        self.backgroud_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative')
        self.conf_thresh = conf_thresh
        self.variance = Config['variance']

        warnings.filterwarnings("ignore")
        loc_data = loc_data.cpu()
        conf_data = conf_data.cpu()
        # batch_size
        num = loc_data.size(0)
        num_priors = prior_data.size(0)
        # 生成空的output
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2,1)
        # 对每一张图片进行处理
        for i in range(num):
            # 对先验框解码获得预测框
            deconded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                # 对每一类进行非极大抑制
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(deconded_boxes)
                boxes = deconded_boxes[l_mask].view(-1,4)
                # 进行非极大值抑制
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output



