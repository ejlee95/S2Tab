# -----------------------------------------------------------------------
# S2Tab official code : model/backbone.py
# -----------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -----------------------------------------------------------------------

import collections
from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from transformers.models.detr.modeling_detr import DetrConfig
from util.misc import is_main_process

class DETRPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, d_model)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        num_channels, d_model = config.num_channels, config.d_model

        self.num_channels = num_channels
        self.d_model = d_model

        name = config.backbone
        ## TODO simple vit 2) without padding like Pix2Struct
        if name == 'vit': # encoder has patch_embedding
            self.backbone = None
            self.projection = None
            self.return_key = None
            self.intermediate_feature = None
        elif 'fpn' in name: # resnet#.fpn
            backbone_name = name.split('.')[0]
            layer_name = config.backbone_layer
            self.projection = resnet_fpn_backbone(backbone_name,
                                                  weights='DEFAULT' if is_main_process() else None,
                                                  trainable_layers=5,
                                                  returned_layers=None,
                                                  )
            return_layer_idx = 0
            temp_block = self.projection.fpn.inner_blocks[return_layer_idx]
            temp_blocks = list(temp_block.named_parameters())
            self.intermediate_feature = temp_blocks[0][1].shape[0] # output channel
            self.return_key = '0'
        else:
            self.backbone = getattr(torchvision.models, name)(
                weights='DEFAULT' if is_main_process() else None,
                norm_layer=FrozenBatchNorm2d if 'resnet' in name else None
            )
            layer_name = config.backbone_layer
            return_layer_idx = -1

            return_layers = {layer_name: '0'}

            # feature_info
            temp_block = self.backbone.get_submodule(layer_name)[return_layer_idx]
            temp_blocks = list(temp_block.named_parameters())
            self.intermediate_feature = temp_blocks[-2][1].shape[0]
            return_layers = {layer_name.replace('features.', ''): '0'}
            if hasattr(self.backbone, 'features'):
                self.projection = IntermediateLayerGetter(self.backbone.features, return_layers=return_layers)
            else:
                self.projection = IntermediateLayerGetter(self.backbone, return_layers=return_layers)
            self.return_key = '0' if ('stages' not in layer_name and name != 'resnet31') else None

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.BoolTensor=None) -> Tuple[torch.Tensor, torch.BoolTensor]:
        batch_size, num_channels, height, width = pixel_values.shape
        if self.projection == None:
            return pixel_values, pixel_mask
        
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if self.return_key is not None:
            embeddings = self.projection(pixel_values)[self.return_key] # (bs, h_dim, h/16, w/16)
        else:
            embeddings = self.projection(pixel_values)
        if pixel_mask is not None:
            # resize
            pixel_mask = F.interpolate(pixel_mask[None].float(), size=embeddings.shape[-2:]).to(torch.bool)[0]

        return embeddings, pixel_mask
    

def build_backbone(args):

    image_max = args.train.image_max if args.exec.eval == False else args.test.image_max
    image_min = args.train.image_min if args.exec.eval == False else args.test.image_min

    cfg = DetrConfig(d_model=args.model.backbone.hidden_dim,
                     is_encoder_decoder=True,
                     activation_function='gelu',
                     dropout=args.model.backbone.dropout,
                     attention_dropout=0,
                     position_embedding_type='sine',
                     backbone=args.model.backbone.backbone_name,
                     backbone_layer=args.model.backbone.backbone_layer,
                     layer_norm_eps=1e-12,
                     patch_size=16,
                     image_size=(image_max, image_min),
                )
    
    return DETRPatchEmbeddings(cfg)