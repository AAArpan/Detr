from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch_intermediate_layer_getter import IntermediateLayerGetter
from typing import Dict, List
from wavemix import Level4Waveblock, Level3Waveblock, Level2Waveblock, Level1Waveblock
import torch.nn as nn
from einops.layers.torch import Rearrange
from util.misc import NestedTensor 
from torchsummary import summary

from .position_encoding import build_position_encoding

class FrozenBatchNorm2d(torch.nn.Module):
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"pool":"0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list):
        xs, mod = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    def __init__(self, transformer_model: nn.Module,
                 return_interm_layers: bool):
        backbone = transformer_model
        num_channels = 512  
        super().__init__(backbone, train_backbone=False, num_channels=2048, return_interm_layers=False)

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    model = WaveMix(
        num_classes=1000,
        depth=16,
        mult=2,
        ff_channel=192,
        final_dim=192,
        dropout=0.5,
        level=3,
        initial_conv='pachify',
        patch_size=4,
        stride=2
    ).cuda()
    backbone = Backbone(model, 2048).cuda()
    state_dict = torch.hub.load_state_dict_from_url('https://huggingface.co/cloudwalker/wavemix/resolve/main/Saved_Models_Weights/ImageNet/imagenet_71.49.pth')
    backbone.load_state_dict(state_dict, strict=False)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

class WaveMix(nn.Module):
    def __init__(
        self,
        *,
        num_classes=1000,
        depth = 16,
        mult = 2,
        ff_channel = 192,
        final_dim = 192,
        dropout = 0.5,
        level = 3,
        initial_conv = 'pachify', # or 'strided'
        patch_size = 4,
        stride = 2,

    ):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth): 
                if level == 4:
                    self.layers.append(Level4Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
                elif level == 3:
                    self.layers.append(Level3Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
                elif level == 2:
                    self.layers.append(Level2Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
                else:
                    self.layers.append(Level1Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
        
        self.pool = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1)
        )

        if initial_conv == 'strided':
            self.conv = nn.Sequential(
            nn.Conv2d(3, int(final_dim/2), 3, stride, 1),
            nn.Conv2d(int(final_dim/2), final_dim, 3, stride, 1)
        )
        else:
            self.conv = nn.Sequential(
            nn.Conv2d(3, int(final_dim/4),3, 1, 1),
            nn.Conv2d(int(final_dim/4), int(final_dim/2), 3, 1, 1),
            nn.Conv2d(int(final_dim/2), final_dim, patch_size, patch_size),
            nn.GELU(),
            nn.BatchNorm2d(final_dim)
            )
        
    def forward(self, img):
        # print(img.shape)
        x = self.conv(img)   
            
        for attn in self.layers:
            x = attn(x) + x
        # print(x.shape, "hi")
        out = self.pool(x)
        # print(out.shape)
        return out

# arr = torch.rand(1,3,224,224).cuda()
# pred = transformer_model.cuda()(arr)
# state_dict = torch.hub.load_state_dict_from_url('LOCAL_PATH')
# backbone.load_state_dict(state_dict)
