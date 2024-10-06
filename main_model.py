import os
import re
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.cswin import CSWin_64_12211_tiny_224, CSWin_64_24322_small_224, CSWin_96_24322_base_384, \
    CSWin_96_24322_base_224
    

from models.backbone.GAE import GAEAttention
from models.head.FCN import FCNHead
from models.neck.msfi import MSFI
from collections import OrderedDict
from util.common import ScaleInOutput

class Conv3Relu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Conv3Relu, self).__init__()
        self.extract = nn.Sequential(nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1),
                                               stride=(stride, stride), bias=False),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.extract(x)
        return x

class ChangeDetection(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.inplanes = int(re.sub(r"\D", "", opt.backbone.split("_")[-1]))  
        self.dl = opt.dual_label
        self.auxiliary_head = False   
        self._create_backbone(opt.backbone)
        self._create_neck(opt.neck)
        self._create_heads(opt.head)
        self.GAE1 = GAEAttention(self.inplanes*1)
        self.GAE2 = GAEAttention(self.inplanes*2)

        self.contrast_loss1 = Conv3Relu(512,2)
        self.contrast_loss2 = Conv3Relu(512,512)

        if opt.pretrain.endswith(".pt"):
            self._init_weight(opt.pretrain)   

    def forward(self, xa, xb, tta=False):
        if not tta: 
            return self.forward_once(xa, xb)
        else:
            return self.forward_tta(xa, xb)

    def forward_once(self, xa, xb):
        _, _, h_input, w_input = xa.shape
        assert xa.shape == xb.shape, "The two images are not the same size, please check it."

        fa1, fa2, fa3, fa4 = self.backbone(xa) 
        fb1, fb2, fb3, fb4 = self.backbone(xb)
        fa1 = self.GAE1(fa1)
        fb1 = self.GAE1(fb1)
        fa2 = self.GAE2(fa2)
        fb2 = self.GAE2(fb2)

        ms_feats = fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4  

        change4, change = self.neck(ms_feats)
        feature_map_4_1 =  self.contrast_loss1(change4) 
        feature_map_4_2 = self.contrast_loss2(change4)  
        out = self.head_forward(ms_feats, change, out_size=(h_input, w_input))

        return feature_map_4_1, feature_map_4_2, out

    def head_forward(self, ms_feats, change, out_size):
        out = F.interpolate(self.head1(change), size=out_size, mode='bilinear', align_corners=True)
        return out

    def _init_weight(self, pretrain=''):  
        for m in self.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):  
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if pretrain.endswith('.pt'):
            pretrained_dict = torch.load(pretrain)
            if isinstance(pretrained_dict, nn.DataParallel):
                pretrained_dict = pretrained_dict.module
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.state_dict().items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(OrderedDict(model_dict), strict=True)
            print("=> ChangeDetection load {}/{} items from: {}".format(len(pretrained_dict),
                                                                        len(model_dict), pretrain))

    def _create_backbone(self, backbone):
        if 'cswin' in backbone:
            if '_t_' in backbone:
                self.backbone = CSWin_64_12211_tiny_224(pretrained=True)
            elif '_s_' in backbone:
                self.backbone = CSWin_64_24322_small_224(pretrained=True)
            elif '_b_' in backbone:
                self.backbone = CSWin_96_24322_base_384(pretrained=True)
            elif '_b448_' in backbone:
                self.backbone = CSWin_96_24322_base_224(pretrained=True)
        else:
            raise Exception('Not Implemented yet: {}'.format(backbone))

    def _create_neck(self, neck):
        if 'fpn' in neck:
            self.neck = MSFI(self.inplanes, neck)
    def _select_head(self, head):
        if head == 'fcn':
            return FCNHead(self.inplanes, 2)

    def _create_heads(self, head):
        self.head1 = self._select_head(head)
        self.head2 = self._select_head(head) if self.dl else None

class EnsembleModel(nn.Module):
    def __init__(self, ckp_paths, device, method="avg2", input_size=512):
        super(EnsembleModel, self).__init__()
        self.method = method
        self.models_list = []
        assert isinstance(ckp_paths, list), "ckp_path must be a list: {}".format(ckp_paths)
        print("-"*50+"\n--Ensamble method: {}".format(method))
        for ckp_path in ckp_paths:
            if os.path.isdir(ckp_path):
                weight_file = os.listdir(ckp_path)
                ckp_path = os.path.join(ckp_path, weight_file[0])
            print("--Load model: {}".format(ckp_path))
            model = torch.load(ckp_path, map_location=device)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel) \
                    or isinstance(model, nn.DataParallel):
                model = model.module
            self.models_list.append(model)
        self.scale = ScaleInOutput(input_size)

    def eval(self):
        for model in self.models_list:
            model.eval()

    def forward(self, xa, xb, tta=False):
        xa, xb = self.scale.scale_input((xa, xb))
        out1, out2 = 0, 0
        cd_pred1 = None

        for i, model in enumerate(self.models_list):
            _, _, outs = model(xa, xb, tta)
            if not isinstance(outs, tuple):  
                outs = (outs, outs)
            outs = self.scale.scale_output(outs)
            if "avg" in self.method:
                if self.method == "avg2":
                    outs = (F.softmax(outs[0], dim=1), F.softmax(outs[1], dim=1))  
                out1 += outs[0]
                out2 += outs[1]
                _, cd_pred1 = torch.max(out1, 1) 
            elif self.method == "vote":  
                _, out1_tmp = torch.max(outs[0], 1) 
                _, out2_tmp = torch.max(outs[1], 1)
                out1 += out1_tmp
                out2 += out2_tmp
                cd_pred1 = out1 / i >= 0.5

        return _, _, cd_pred1