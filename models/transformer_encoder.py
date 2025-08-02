# -----------------------------------------------------------------------
# S2Tab official code : models/transformer_encoder.py
# -----------------------------------------------------------------------
# Modified from huggingface (https://github.com/huggingface/transformers/blob/v4.22.1/src/transformers)
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------

from dataclasses import dataclass
from typing import Union, Optional, Tuple
import math
import torch
from torch import nn

from .coordconv import CoordConv2d

from transformers.models.detr.modeling_detr import DetrPreTrainedModel, DetrConfig, DetrEncoderLayer, DetrAttention, _expand_mask, DetrEncoder
from transformers.models.beit.modeling_beit import BeitPooler
from transformers.modeling_outputs import BaseModelOutputWithPooling

@dataclass
class TableDetrModelOutputWithPooling(BaseModelOutputWithPooling):
    pixel_mask: torch.FloatTensor = None
    pixel_mask_origin: torch.FloatTensor = None
    position_embedings: torch.FloatTensor = None
    feature_size: Tuple = None
    segm_map: torch.FloatTensor = None
    border_class: Tuple = None
    border_mask: Tuple = None

class TablePooler(BeitPooler):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.layernorm is not None:
            # Mean pool the final hidden states of the patch tokens
            patch_tokens = hidden_states
            pooled_output = self.layernorm(patch_tokens.mean(1))
        else:
            # Pool by simply taking the final hidden state of the [CLS] token
            pooled_output = hidden_states[:, 0]

        return pooled_output

class DetrSinePositionEmbedding(nn.Module):
    """
    Modified from huggingface/detr.
    Including 'cls' token embedding
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, embedding_dim=64, temperature=10000, normalize=False, scale=None, include_cls=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.include_cls = include_cls

    def forward(self, pixel_values, pixel_mask):
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")
        # not_mask = ~pixel_mask
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        if self.include_cls:
            y_embed = torch.cat((torch.zeros(y_embed.shape[0], 1, y_embed.shape[2]).to(y_embed.device), y_embed), dim=1) # (batch, h+1, w)
            x_embed = torch.cat((torch.zeros(x_embed.shape[0], x_embed.shape[1], 1).to(x_embed.device), x_embed), dim=2) # (batch, h, w+1)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.embedding_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        if self.include_cls:
            pos = torch.cat((pos_y[:, 1:, :, :], pos_x[:, :, 1:, :]), dim=3).permute(0, 3, 1, 2) # (batch, C, h, w)
            pos_cls = torch.cat((pos_y[:, :1, :1, :], pos_x[:, :1, :1, :]), dim=3).permute(0, 3, 1, 2) # (batch, C, 1, 1)
        else:
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
            pos_cls = None
        return pos, pos_cls

class DETREmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: DetrConfig) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        if config.use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        else:
            self.mask_token = None
        self.d_model = config.d_model

        # Create projection layer
        self.input_projection = nn.Conv2d(config.intermediate_feature, config.d_model, kernel_size=1)

        self.include_cls = not config.use_mean_pooling

        if config.position_embedding_type == 'sine':
            assert config.d_model % 2 == 0
            self.position_embeddings = DetrSinePositionEmbedding(embedding_dim=config.d_model//2, 
                                                                    normalize=False, 
                                                                    scale=None, 
                                                                    include_cls=self.include_cls)
        else:
            self.position_embeddings = None
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, embeddings: torch.Tensor,
                        bool_masked_pos: Optional[torch.BoolTensor] = None,
                        pixel_mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        batch_size, _, h, w = embeddings.size()
        seq_len = h * w

        # add projection
        embeddings = self.input_projection(embeddings)

        # add 2d sine position embedding
        if self.position_embeddings is not None:
            pos_emb, pos_cls_emb = self.position_embeddings(embeddings, pixel_mask)
            pos_emb = pos_emb.flatten(2)
            if pos_cls_emb is not None:
                pos_cls_emb = pos_cls_emb.flatten(2)
                pos_emb = torch.cat((pos_cls_emb, pos_emb), dim=2)
            pos_emb = pos_emb.transpose(1, 2).contiguous()

        if self.include_cls:
            # prepare cls token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # flatten features
        embeddings = embeddings.flatten(2).transpose(1, 2).contiguous()
        pixel_mask = pixel_mask.flatten(1).contiguous()
        
        if bool_masked_pos is not None:
            bool_masked_pos = bool_masked_pos.flatten(1)
            # masking embeddings
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1 - w) + mask_tokens * w

        if self.include_cls:
            # expend cls token
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)
            pixel_mask = torch.cat((torch.ones((batch_size, 1), dtype=torch.bool).to(pixel_mask.device), 
                                        pixel_mask), dim=1)
            if bool_masked_pos is not None:
                bool_masked_pos = torch.cat((torch.zeros((batch_size, 1), dtype=torch.bool).to(bool_masked_pos.device), 
                                        bool_masked_pos), dim=1)
        # add position embedding
        if self.position_embeddings is None:
            pos_emb = None

        embeddings = self.dropout(embeddings)

        return embeddings, pos_emb, pixel_mask, (h, w)
    
class HeatmapRegressor(nn.Module):
    def __init__(self, config, upsample_num):
        super().__init__()

        self.layer_num = upsample_num
        self.self = nn.ModuleDict({
                            'conv': nn.ModuleList([nn.Conv2d(config.d_model, config.d_model, 3, 1, padding='same') for _ in range(upsample_num+1)]),
                            'upsample': nn.Upsample(scale_factor=2, mode='nearest'),
                            'act': nn.ReLU(),
                            'coordconv': CoordConv2d(config.d_model, 1, 1, 1, 'same'),
                            })

    def forward(self, x):
        x = self.self.conv[0](x)
        for i in range(1, self.layer_num+1):
            x = self.self.upsample(x)
            x_res = self.self.conv[i](x)
            x = x + x_res
            x = self.self.act(x)
        x = self.self.coordconv(x)

        return x

class TableEncoder(DetrPreTrainedModel):
    def __init__(self, config: DetrConfig, add_pooling_layer: bool=True) -> None:
        super().__init__(config, add_pooling_layer)

        self.embeddings = DETREmbeddings(config)
        self.encoder = DetrEncoder(config)

        if config.include_heatmap:
            self.heatmap_head = HeatmapRegressor(config, 2)
        
        if add_pooling_layer:
            self.pooler = TablePooler(config)
        else:
            self.pooler = None
        
        if config.add_roi_projection:
            if config.roi_projection_only_phy:
                self.roi_projection = nn.Conv2d(config.d_model, config.d_model//2, (2,2), (2,2), bias=False)
            else:    
                self.roi_projection = nn.Conv2d(config.d_model, config.d_model, (2,2), (2,2), bias=False)
        

        self.post_init()

    def forward(
            self,
            pixel_values: Optional[torch.Tensor]=None,
            pixel_mask: Optional[torch.BoolTensor]=None,
            output_attentions: Optional[bool]=None,
            output_hidden_states: Optional[bool]=None,
            return_dict: Optional[bool]=None,
            **kwargs,
        ) -> Union[tuple, TableDetrModelOutputWithPooling]:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            # Prepare head mask if needed
            # 1.0 in head_mask indicate we keep the head
            # attention_probs has shape bsz x n_heads x N x N
            # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
            # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
            # head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

            embedding_output, pos_emb, pixel_mask, interp_size = self.embeddings(pixel_values, None, pixel_mask) # interp_size = (h, w)
            C = self.embeddings.d_model

            segm_map = None

            encoder_outputs = self.encoder(
                inputs_embeds=embedding_output,
                attention_mask=pixel_mask,
                position_embeddings=pos_emb,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
            sequence_output = encoder_outputs[0]
            row_mask, col_mask = None, None

            last_hidden_state_output = sequence_output
            pixel_mask_output = pixel_mask
            position_embedings_output = pos_emb

            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

            border_class_row, border_class_col = None, None

            if not return_dict:
                head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
                return head_outputs + encoder_outputs[1:]

            return TableDetrModelOutputWithPooling(
                last_hidden_state=last_hidden_state_output,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
                pixel_mask=pixel_mask_output,
                pixel_mask_origin=pixel_mask,
                position_embedings=position_embedings_output, 
                feature_size=interp_size,
                segm_map=segm_map,
                border_class = (border_class_row, border_class_col),
                border_mask = (row_mask, col_mask),
            )

