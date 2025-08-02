# -----------------------------------------------------------------------
# S2Tab official code : models/transformer_decoder_mult_ver2_1_1.py
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

"""PyTorch BERT-multi-task model."""

from typing import List, Optional, Union, Tuple
from dataclasses import dataclass
from copy import deepcopy

import math
import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertLayer, BertAttention, BertSelfAttention, BertSelfOutput, BertIntermediate, BertOutput
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.utils import logging
from transformers.activations import ACT2FN

logger = logging.get_logger(__name__)

class PositionalEmbedding(nn.Module):
    """
    modified from transformer xl in huggingface transformers library
    """
    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.stack((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=2).flatten(1)

        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]

class BertMultiEmbeddings(nn.Module):
    """Construct the embeddings from coord, token type_row, token_type_col."""

    def __init__(self, config):
        super().__init__()
        d_emb = config.hidden_size // 2
        self.coord_embed_aggr_type = 'sum'
        
        self.coord_x_embeddings = nn.Embedding(config.coord_offset + config.coord_bin, d_emb)
        self.coord_y_embeddings = nn.Embedding(config.coord_offset + config.coord_bin, d_emb)

        self.token_type_embeddings_row = nn.Embedding(config.type_vocab_size, config.hidden_size//4)
        self.token_type_embeddings_col = nn.Embedding(config.type_vocab_size, config.hidden_size//4)
        if config.position_embedding_type == 'absolute':
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        elif config.position_embedding_type == 'sine':
            self.position_embeddings = PositionalEmbedding(config.hidden_size)
        elif config.position_embedding_type == 'none':
            """no position embeddings"""
            self.position_embeddings = None
            pass
        else:
            raise ValueError(f"position embedding of {config.position_embedding_type} is not implemented. please set learnable or sine")
        
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "sine")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))       
        self.register_buffer(
            "token_type_ids_row",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "token_type_ids_col",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        input_coord_ids: Optional[torch.FloatTensor] = None,
        input_delta_vars: Optional[torch.FloatTensor] = None,
        token_type_ids_row: Optional[torch.LongTensor] = None,
        token_type_ids_col: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_coord_ids is not None:
            input_shape = input_coord_ids.size()[:-2] # batch, seqlen
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids_row is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids_row = self.token_type_ids_row[:, :seq_length]
                buffered_token_type_ids_row_expanded = buffered_token_type_ids_row.expand(input_shape[0], seq_length)
                token_type_ids_row = buffered_token_type_ids_row_expanded
                buffered_token_type_ids_col = self.token_type_ids_col[:, :seq_length]
                buffered_token_type_ids_col_expanded = buffered_token_type_ids_col.expand(input_shape[0], seq_length)
                token_type_ids_col = buffered_token_type_ids_col_expanded
            else:
                token_type_ids_row = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
                token_type_ids_col = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            ## bilinear interpolation
            coord_embedding_111 = self.coord_x_embeddings(input_coord_ids[:, :, 0, 0]) + self.coord_y_embeddings(input_coord_ids[:, :, 0, 1])
            coord_embedding_112 = self.coord_x_embeddings(input_coord_ids[:, :, 1, 0]) + self.coord_y_embeddings(input_coord_ids[:, :, 1, 1])
            coord_embedding_121 = self.coord_x_embeddings(input_coord_ids[:, :, 2, 0]) + self.coord_y_embeddings(input_coord_ids[:, :, 2, 1])
            coord_embedding_122 = self.coord_x_embeddings(input_coord_ids[:, :, 3, 0]) + self.coord_y_embeddings(input_coord_ids[:, :, 3, 1])
            coord_embedding_211 = self.coord_x_embeddings(input_coord_ids[:, :, 4, 0]) + self.coord_y_embeddings(input_coord_ids[:, :, 4, 1])
            coord_embedding_212 = self.coord_x_embeddings(input_coord_ids[:, :, 5, 0]) + self.coord_y_embeddings(input_coord_ids[:, :, 5, 1])
            coord_embedding_221 = self.coord_x_embeddings(input_coord_ids[:, :, 6, 0]) + self.coord_y_embeddings(input_coord_ids[:, :, 6, 1])
            coord_embedding_222 = self.coord_x_embeddings(input_coord_ids[:, :, 7, 0]) + self.coord_y_embeddings(input_coord_ids[:, :, 7, 1])

            delta_x11 = input_delta_vars[:, :, 0].unsqueeze(-1).repeat(1, 1, coord_embedding_111.shape[-1])
            delta_x12 = input_delta_vars[:, :, 1].unsqueeze(-1).repeat(1, 1, coord_embedding_111.shape[-1])
            delta_y11 = input_delta_vars[:, :, 2].unsqueeze(-1).repeat(1, 1, coord_embedding_111.shape[-1])
            delta_y12 = input_delta_vars[:, :, 3].unsqueeze(-1).repeat(1, 1, coord_embedding_111.shape[-1])
            delta_x21 = input_delta_vars[:, :, 4].unsqueeze(-1).repeat(1, 1, coord_embedding_111.shape[-1])
            delta_x22 = input_delta_vars[:, :, 5].unsqueeze(-1).repeat(1, 1, coord_embedding_111.shape[-1])
            delta_y21 = input_delta_vars[:, :, 6].unsqueeze(-1).repeat(1, 1, coord_embedding_111.shape[-1])
            delta_y22 = input_delta_vars[:, :, 7].unsqueeze(-1).repeat(1, 1, coord_embedding_111.shape[-1])

            inputs_embeds = coord_embedding_111 * delta_x11 * delta_y11 + coord_embedding_112 * delta_x11 * delta_y12 + \
                            coord_embedding_121 * delta_x12 * delta_y11 + coord_embedding_122 * delta_x12 * delta_y12 + \
                            coord_embedding_211 * delta_x21 * delta_y21 + coord_embedding_212 * delta_x21 * delta_y22 + \
                            coord_embedding_221 * delta_x22 * delta_y21 + coord_embedding_222 * delta_x22 * delta_y22

        token_type_embeddings = torch.cat((self.token_type_embeddings_row(token_type_ids_row), 
                                           self.token_type_embeddings_col(token_type_ids_col)),
                                           dim=-1)

        # embeddings = (row, column, coord)
        embeddings = torch.cat((token_type_embeddings, inputs_embeds), dim=-1)

        # add 1D position embedding
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        elif self.position_embedding_type == "sine":
            position_embeddings = self.position_embeddings(position_ids[0], embeddings.shape[0])
            embeddings += position_embeddings
        elif self.position_embedding_type == "none":
            pass
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertDetrSelfAttention(BertSelfAttention):
    def __init__(self, config, position_embedding_type=None, cross_modal=False):
        super().__init__(config, position_embedding_type)
        self.cross_modal = cross_modal
        self.apply_attension_sharpening = config.apply_attension_sharpening

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[torch.Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings
    
    def attention_sharpening(self, tensor: torch.Tensor, dim: int, alpha: float=1.0):
        eps = 1e-12
        tensor = torch.exp(alpha * tensor)
        length = tensor.shape[dim]
        tensor = (tensor - 1) / (torch.sum(tensor, dim=dim).unsqueeze(dim) - length + eps)

        return tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_position_embeddings: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        # add position embeddings to the hidden states before projecting to queries and keys
        hidden_states_original = hidden_states
        hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        # add encoder_position_embeddings to the encoder states
        encoder_hidden_states_original = encoder_hidden_states
        encoder_hidden_states = self.with_pos_embed(encoder_hidden_states, encoder_position_embeddings)

        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None
        if not is_cross_attention or self.cross_modal:
            self.apply_attension_sharpening = False

        if self.cross_modal and past_key_value is not None:
            # bi-directional attention
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states_original))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        elif self.cross_modal:
            # bi-directional attention
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states_original))
        elif is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions - standard cross attention : from encoder output
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            # standard cross attention : from encoder output
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states_original))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states_original))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states_original))
            
        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if position_embeddings is not None:
            assert "relative" not in self.position_embedding_type
        elif "relative" in self.position_embedding_type:
            assert position_embeddings is None

        if "relative" in self.position_embedding_type:
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        if self.apply_attension_sharpening:
            attention_probs = self.attention_sharpening(attention_probs, dim=-1, alpha=12)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class BertMultiSelfOutput(BertSelfOutput):
    def __init__(self, config, split_dim):
        super().__init__(config)
        self.dense = nn.ModuleList([nn.Linear(sd, sd) for sd in split_dim])
        self.LayerNorm = nn.ModuleList([nn.LayerNorm(sd, eps=config.layer_norm_eps) for sd in split_dim])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.split_dim = split_dim

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = list(hidden_states.split(self.split_dim, dim=-1)) # tuple
        input_tensor = list(input_tensor.split(self.split_dim, dim=-1))
        for i in range(len(self.split_dim)):
            hidden_states[i] = self.dense[i](hidden_states[i])
            hidden_states[i] = self.dropout(hidden_states[i])
            hidden_states[i] = self.LayerNorm[i](hidden_states[i] + input_tensor[i])
        hidden_states = torch.cat(hidden_states, dim=-1)

        return hidden_states
    
class BertMultiSelfOutput_dense(BertSelfOutput):
    def __init__(self, config, split_dim):
        super().__init__(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.ModuleList([nn.LayerNorm(sd, eps=config.layer_norm_eps) for sd in split_dim])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.split_dim = split_dim

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = list(input_tensor.split(self.split_dim, dim=-1))
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = list(hidden_states.split(self.split_dim, dim=-1)) # tuple
        for i in range(len(self.split_dim)):
            hidden_states[i] = self.LayerNorm[i](hidden_states[i] + input_tensor[i])
        hidden_states = torch.cat(hidden_states, dim=-1)

        return hidden_states

class BertMultiAttention(BertAttention):
    def __init__(self, config, position_embedding_type=None, cross_modal=False):
        super().__init__(config, position_embedding_type)
        # overloading
        self.self = BertDetrSelfAttention(config, position_embedding_type=position_embedding_type, cross_modal=cross_modal)
        if config.fusion_attention:
            split_dim = [config.hidden_size//4, config.hidden_size//4, config.hidden_size//2]
            self.output = BertMultiSelfOutput(config, split_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_embeddings: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            encoder_position_embeddings: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
        ) -> Tuple[torch.Tensor]:
            self_outputs = self.self(
                hidden_states,
                attention_mask,
                position_embeddings,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                encoder_position_embeddings,
                past_key_value,
                output_attentions,
            )
            attention_output = self.output(self_outputs[0], hidden_states)
            outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
            return outputs


class MultiwayIntermediate(BertIntermediate):
    def __init__(self, config, split_dims):
        super().__init__(config)
        self.split_dim = split_dims[0]
        self.dense = nn.ModuleList([nn.Linear(sd, insz) for sd, insz in zip(*split_dims)])
        # self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = list(hidden_states.split(self.split_dim, dim=-1))
        for i in range(len(self.split_dim)):
            hidden_states[i] = self.dense[i](hidden_states[i])
            hidden_states[i] = self.intermediate_act_fn(hidden_states[i])
        hidden_states = torch.cat(hidden_states, dim=-1)
        return hidden_states
    
class MultiwayOutput(BertOutput):
    def __init__(self, config, split_dims):
        super().__init__(config)
        self.split_inter_dim = split_dims[1]
        self.split_dim = split_dims[0]
        self.dense = nn.ModuleList([nn.Linear(insz, sd) for sd, insz in zip(*split_dims)])
        # self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.ModuleList([nn.LayerNorm(sd, eps=config.layer_norm_eps) for sd in split_dims[0]])
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = list(hidden_states.split(self.split_inter_dim, dim=-1))
        input_tensor = list(input_tensor.split(self.split_dim, dim=-1))
        for i in range(len(self.split_inter_dim)):
            hidden_states[i] = self.dense[i](hidden_states[i])
            hidden_states[i] = self.dropout(hidden_states[i])
            hidden_states[i] = self.LayerNorm[i](hidden_states[i] + input_tensor[i])
        hidden_states = torch.cat(hidden_states, dim=-1)
        return hidden_states


class BertMultiOutput(BertOutput):
    def __init__(self, config, split_dim):
        super().__init__(config)
        self.split_dim = split_dim
        # self.dense = nn.ModuleList([nn.Linear(insz, sd) for sd, insz in zip(*split_dims)])
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.ModuleList([nn.LayerNorm(sd, eps=config.layer_norm_eps) for sd in split_dim])
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = list(input_tensor.split(self.split_dim, dim=-1))
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = list(hidden_states.split(self.split_dim, dim=-1))
        for i in range(len(self.split_dim)):
            hidden_states[i] = self.LayerNorm[i](hidden_states[i] + input_tensor[i])
        hidden_states = torch.cat(hidden_states, dim=-1)
        return hidden_states
    
class BertMultiLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)

        self.attention = BertMultiAttention(config) #, position_embedding_type="relative_key_query")
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertMultiAttention(config, position_embedding_type="absolute")

        if config.fusion_attention:
            config_sub = deepcopy(config)
            config_sub.hidden_size = config.hidden_size // 2
            config_sub.cross_attention_hidden_size = config.hidden_size // 2
            config_sub.fusion_attention = False
            self.log_modal_crossattention = BertMultiAttention(config_sub, position_embedding_type="absolute", cross_modal=True)
            self.phy_modal_crossattention = BertMultiAttention(config_sub, position_embedding_type="absolute", cross_modal=True)
            split_dim = [config.hidden_size//2, config.hidden_size//2]
            split_inter_dim = [config.intermediate_size//2, config.intermediate_size//2]
            self.intermediate = MultiwayIntermediate(config, [split_dim, split_inter_dim])
            self.output = MultiwayOutput(config, [split_dim, split_inter_dim])


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_position_embeddings: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            position_embeddings,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 2,3 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[2:4] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                position_embeddings,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                encoder_position_embeddings,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 2,3 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        if hasattr(self, "log_modal_crossattention"):
            log_attention_output, phy_attention_output = attention_output.split([attention_output.shape[-1]//2, attention_output.shape[-1]//2], dim=-1)

            # log modal cross_attn cached key/values tuple is at positions 4,5 of past_key_value tuple
            log_modal_cross_attn_past_key_value = past_key_value[4:6] if past_key_value is not None else None
            log_modal_cross_attention_outputs = self.log_modal_crossattention(
                log_attention_output,
                attention_mask,
                position_embeddings,
                head_mask,
                phy_attention_output,
                attention_mask,
                position_embeddings,
                log_modal_cross_attn_past_key_value,
                output_attentions,
            )

            # phy modal cross_attn cached key/values tuple is at positions 6,7 of past_key_value tuple
            phy_modal_cross_attn_past_key_value = past_key_value[6:8] if past_key_value is not None else None
            phy_modal_cross_attention_outputs = self.phy_modal_crossattention(
                phy_attention_output,
                attention_mask,
                position_embeddings,
                head_mask,
                log_attention_output,
                attention_mask,
                position_embeddings,
                phy_modal_cross_attn_past_key_value,
                output_attentions,
            )

            modal_attention_output = torch.cat((log_modal_cross_attention_outputs[0], 
                                                phy_modal_cross_attention_outputs[0]), dim=-1)
            attention_output = modal_attention_output
            outputs = outputs + log_modal_cross_attention_outputs[1:-1] + phy_modal_cross_attention_outputs[1:-1] # add cross attentions if we output attention weights

            # add cross-attn cache to positions 4,5 and 6,7 of present_key_value tuple
            log_modal_cross_attn_present_key_value = log_modal_cross_attention_outputs[-1]
            phy_modal_cross_attn_present_key_value = phy_modal_cross_attention_outputs[-1]
            present_key_value = present_key_value + log_modal_cross_attn_present_key_value + phy_modal_cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

class BertMultiEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)

        # overloading
        self.layer = nn.ModuleList([BertMultiLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_position_embeddings: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    position_embeddings,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    encoder_position_embeddings,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    position_embeddings,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    encoder_position_embeddings,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (FFN) """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
    def reset_parameters(self, sigma_min=0.005, var=0, init_type='sigma') -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                if layer.bias is not None:
                    if init_type == 'sigma':
                        bound = math.log(sigma_min)*2 # 0.0005
                        torch.nn.init.constant_(layer.bias, bound)
                    elif init_type == 'class':
                        assert isinstance(var, int)
                        init_bias = torch.zeros_like(layer.bias)
                        init_bias[var] = 1
                        with torch.no_grad():
                            layer.bias.copy_(init_bias)
            else:
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
@dataclass
class CausalMLMOutputWithCrossAttentions(CausalLMOutputWithCrossAttentions):
    continuous_logits: torch.FloatTensor = None
    logits_col: torch.FloatTensor = None

class TableDecoder(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        self.config = config

        self.embeddings = BertMultiEmbeddings(config)
        self.encoder = BertMultiEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None


        # prediction heads
        in_dim_type = config.hidden_size // 4
        in_dim_box = config.hidden_size // 2
        self.token_type_head_row = MLP(in_dim_type, config.hidden_size, config.type_vocab_size, 2)
        self.token_type_head_col = MLP(in_dim_type, config.hidden_size, config.type_vocab_size, 2)
        self.coord_head = MLP(in_dim_box, config.hidden_size, 4, 3)

        # Initialize weights and apply final processing
        self.post_init()
        self.token_type_head_row.reset_parameters(var=config.type_offset + 2, init_type='class')
        self.token_type_head_col.reset_parameters(var=config.type_offset + 2, init_type='class')


    def prepare_inputs_for_generation(self, 
                                      input_ids,
                                      past=None, 
                                      attention_mask=None, 
                                      **model_kwargs):
        input_coord_ids, token_type_ids_row, token_type_ids_col, input_delta_var = input_ids
        input_shape = token_type_ids_row.shape[:2] # (bs, seq_len)
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = token_type_ids_row.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_coord_ids = input_coord_ids[:, -1:, :]
            input_delta_var = input_delta_var[:, -1:, :]
            token_type_ids_row = token_type_ids_row[:, -1:]
            token_type_ids_col = token_type_ids_col[:, -1:]

        return {"input_coord_ids": input_coord_ids, 
                "input_delta_vars": input_delta_var,
                "token_type_ids_row": token_type_ids_row,
                "token_type_ids_col": token_type_ids_col,
                "attention_mask": attention_mask, 
                "past_key_values": past}

    def forward(
        self,
        input_coord_ids: Optional[torch.Tensor] = None,
        input_delta_vars: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids_row: Optional[torch.Tensor] = None,
        token_type_ids_col: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_position_embeddings: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_coord_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_coord_ids and inputs_embeds at the same time")
        elif input_coord_ids is not None:
            input_shape = input_coord_ids.size()[:-2]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_coord_ids.device if input_coord_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0 # CHECK

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids_row is None:
            if hasattr(self.embeddings, "token_type_ids_row"):
                buffered_token_type_ids_row = self.embeddings.buffered_token_type_ids_row[:, :seq_length]
                buffered_token_type_ids_row_expanded = buffered_token_type_ids_row.expand(batch_size, seq_length)
                token_type_ids_row = buffered_token_type_ids_row_expanded
                buffered_token_type_ids_col = self.token_type_ids_col[:, :seq_length]
                buffered_token_type_ids_col_expanded = buffered_token_type_ids_col.expand(input_shape[0], seq_length)
                token_type_ids_col = buffered_token_type_ids_col_expanded
            else:
                token_type_ids_row = torch.zeros(input_shape, dtype=torch.long, device=device)
                token_type_ids_col = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_coord_ids=input_coord_ids,
            input_delta_vars=input_delta_vars,
            position_ids=position_ids,
            token_type_ids_row=token_type_ids_row,
            token_type_ids_col=token_type_ids_col,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        ) ### we you wan't to feed 'position_embeddings' to the encoder, than you output position_embeddings from self.embeddings
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            position_embeddings=None,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            encoder_position_embeddings=encoder_position_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        initial_sequence_output = encoder_outputs[0]

        outputs = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=initial_sequence_output,
            # pooler_output=initial_pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        ) # (entire_hidden_state, pooled_state, past_states)

        sequence_output = outputs[0]
        row_sequence_output, col_sequence_output, box_sequence_output = \
            sequence_output.split([self.config.hidden_size//4, self.config.hidden_size//4, self.config.hidden_size//2], dim=-1)


        prediction_coord = self.coord_head(box_sequence_output)

        prediction_token_type_row = self.token_type_head_row(row_sequence_output)
        prediction_token_type_col = self.token_type_head_col(col_sequence_output)

        return CausalMLMOutputWithCrossAttentions(
            loss=None,
            logits=prediction_token_type_row,
            logits_col=prediction_token_type_col,
            continuous_logits=prediction_coord,
            past_key_values=outputs[1],
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
