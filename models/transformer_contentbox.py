# -----------------------------------------------------------------------
# S2Tab official code : models/transformer_mult_ver2_1_1_empty.py
# -----------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -----------------------------------------------------------------------

from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import VisionEncoderDecoderModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import shift_tokens_right
from transformers import BertConfig, DetrConfig, BeitConfig, ViTConfig

from models.transformer_decoder_contentbox import TableDecoder
import models.generation_utils_contentbox as gen_utils

class TableTransformer(VisionEncoderDecoderModel):
    def __init__(
        self,
        config: Optional[PretrainedConfig]=None,
        encoder: Optional[PreTrainedModel]=None,
        decoder: Optional[PreTrainedModel]=None,
        vision_decoder: Optional[nn.Module]=None,
    ):
        super().__init__(config, encoder, decoder)
        self.config.bos_token_id = decoder.config.bos_token_id
        self.config.pad_token_id = decoder.config.pad_token_id
        self.config.eos_token_id = decoder.config.eos_token_id
        self.config.sep_token_id = decoder.config.sep_token_id
        self.config.eoh_token_id = decoder.config.eoh_token_id
        self.config.decoder_start_token_id_row = decoder.config.bos_token_id
        self.config.decoder_start_token_id_col = decoder.config.bos_token_id
        self.config.max_length = decoder.config.max_length
        self.config.min_length = decoder.config.min_length

        self.encoder_type = encoder.config.model_type

        self.vision_decoder = vision_decoder

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        """
        Implement in subclasses of [`PreTrainedModel`] for custom behavior to prepare inputs in the generate method.
        """
        if past is None and 'past_key_values' in kwargs:
            past = kwargs['past_key_values']
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_coord": decoder_inputs["input_coord_ids"],
            "decoder_input_type_row": decoder_inputs["token_type_ids_row"],
            "decoder_input_type_col": decoder_inputs["token_type_ids_col"],
            "decoder_input_delta": decoder_inputs["input_delta_vars"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
            **kwargs,
        }
        return input_dict

    def forward(
        self,
        pixel_values=None,
        pixel_mask=None,
        decoder_input_coord=None,
        decoder_input_delta=None,
        decoder_input_type_row=None,
        decoder_input_type_col=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ocr_roi_option=None,
        image_sizes=None,
        **kwargs,
    ):
        from models.transformer_encoder import TableDetrModelOutputWithPooling

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}
        kwargs_encoder['pixel_mask'] = pixel_mask

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        ocr_roi_option = ocr_roi_option if ocr_roi_option is not None else 'prediction' # prediction or gt

        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            encoder_outputs = self.encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = TableDetrModelOutputWithPooling(*encoder_outputs)
        

        encoder_hidden_states = encoder_outputs[0]
        encoder_position_embeddings = encoder_outputs.position_embedings

        # optionally project encoder_hidden_states
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # resize encoder attention mask
        encoder_attention_mask = encoder_outputs.pixel_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(torch.float32)

        if (labels is not None) and (decoder_input_coord is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )
        
        ## (Optional) auxiliary center heatmap loss
        if hasattr(self.encoder, 'heatmap_head') or self.vision_decoder is not None:
            bs = encoder_hidden_states.shape[0]
            h, w = encoder_outputs.feature_size
            encoder_hidden_states_patches = encoder_hidden_states.reshape(bs, h, w, -1)
            encoder_hidden_states_patches = encoder_hidden_states_patches.permute(0, 3, 1, 2) # bs, c, h, w
            encoder_center_heatmap = self.encoder.heatmap_head(encoder_hidden_states_patches)
        else:
            encoder_center_heatmap = None

        # Decode
        decoder_outputs = self.decoder(
            input_coord_ids=decoder_input_coord,
            input_delta_vars=decoder_input_delta,
            token_type_ids_row=decoder_input_type_row,
            token_type_ids_col=decoder_input_type_col,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_position_embeddings=encoder_position_embeddings,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # additional OCR
        # option: gt box, pred box
        h_margin, w_margin = 0.02, 0.02
        if ocr_roi_option == 'prediction':
            if image_sizes is not None:
                ### sampling points - original size
                roi_pts = torch.sigmoid(decoder_outputs.continuous_logits.clone().detach()) # normalized
                tgt_sizes = image_sizes
                roi_pts[..., 2:] = roi_pts[..., 2:] + roi_pts[..., :2]
                roi_pts[..., 0] = roi_pts[..., 0] - w_margin
                roi_pts[..., 2] = roi_pts[..., 2] + w_margin
                roi_pts[..., 1] = roi_pts[..., 1] - h_margin
                roi_pts[..., 3] = roi_pts[..., 3] + h_margin
                img_h, img_w = tgt_sizes.unbind(1)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
                roi_pts = roi_pts * scale_fct[:, None, :]

                ### adjust points to the feature size
                padded_sample_size = [img_h.max().item(), img_w.max().item()]
                ratio_h, ratio_w = h / padded_sample_size[0], w / padded_sample_size[1] # h, w ratio
                ratio = torch.as_tensor([ratio_w, ratio_h, ratio_w, ratio_h], device=roi_pts.device) # 1/16 mostly, but it depends on the real input size due to quantization error
                roi_pts = roi_pts * ratio[None, None, :]

                #### to prevent exceeding resized region
                encoder_mask = encoder_attention_mask.reshape(encoder_hidden_states.shape[0], h, w) # bs, h', w'
                clip_size = [x.nonzero() for x in encoder_mask]
                clip_size = [[x[:, 1].max().item(), x[:, 0].max().item()] for x in clip_size] # w, h
                clip_size = torch.as_tensor(clip_size, device=roi_pts.device).tile((1, 2)) # bs, 4 (w,h,w,h)
                roi_pts = roi_pts.clamp(max=clip_size[:,None,:], min=torch.zeros_like(roi_pts))

                roi_inv_masks = (decoder_outputs.logits.argmax(-1) == self.config.pad_token_id) | \
                                (decoder_outputs.logits.argmax(-1) == self.config.sep_token_id) | \
                                (decoder_outputs.logits.argmax(-1) == self.config.eos_token_id) | \
                                (decoder_outputs.logits.argmax(-1) == self.config.eoh_token_id) | \
                                (decoder_outputs.logits_col.argmax(-1) == self.config.pad_token_id) | \
                                (decoder_outputs.logits_col.argmax(-1) == self.config.sep_token_id) | \
                                (decoder_outputs.logits_col.argmax(-1) == self.config.eos_token_id) | \
                                (decoder_outputs.logits_col.argmax(-1) == self.config.eoh_token_id)
                roi_masks = ~roi_inv_masks
        else:
            if kwargs.get('roi_boxes',False):
                ### sampling points - original size
                roi_pts = kwargs['roi_boxes']
                tgt_sizes = image_sizes
                roi_pts[..., 2:] += roi_pts[..., :2]
                roi_pts[..., 0] -= w_margin
                roi_pts[..., 2] += w_margin
                roi_pts[..., 1] -= h_margin
                roi_pts[..., 3] += h_margin
                img_h, img_w = tgt_sizes.unbind(1)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
                roi_pts = roi_pts * scale_fct[:, None, :]

                ### adjust points to the feature size
                padded_sample_size = [img_h.max().item(), img_w.max().item()]
                ratio_h, ratio_w = h / padded_sample_size[0], w / padded_sample_size[1] # h, w ratio
                ratio = torch.as_tensor([ratio_w, ratio_h, ratio_w, ratio_h], device=roi_pts.device) # 1/16 mostly, but it depends on the real input size due to quantization error
                roi_pts = roi_pts * ratio[None, None, :]

                #### to prevent exceeding resized region
                encoder_mask = encoder_attention_mask.reshape(encoder_hidden_states.shape[0], h, w) # bs, h', w'
                clip_size = [x.nonzero() for x in encoder_mask]
                clip_size = [[x[:, 1].max().item(), x[:, 0].max().item()] for x in clip_size] # w, h
                clip_size = torch.as_tensor(clip_size, device=roi_pts.device).tile((1, 2)) # bs, 4 (w,h,w,h)
                roi_pts = roi_pts.clamp(max=clip_size[:,None,:], min=torch.zeros_like(roi_pts))
                
                roi_masks = kwargs['roi_masks']
        if self.vision_decoder is not None:
            text_output = self.vision_decoder(encoder_hidden_states_patches, roi_pts, roi_masks)
        else:
            text_output = {'feature': None, 
                           'logits': None, 
                           'pt_lengths': None, 
                           'attn_scores': None, 
                           'name': 'vision'}

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutputWithPooling(
            logits=decoder_outputs.logits,
            logits_col=decoder_outputs.logits_col,
            continuous_logits=decoder_outputs.continuous_logits,
            empty_logits=decoder_outputs.empty_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            encoder_feature_size=encoder_outputs.feature_size,
            encoder_attention_mask=encoder_attention_mask,
            encoder_attention_mask_origin=encoder_outputs.pixel_mask_origin,
            encoder_center_heatmap=encoder_center_heatmap,
            vision_decoder_logits=text_output['logits'],
        )

    def _get_decoder_start_token_id(self, *args, **kwargs):
        return gen_utils._get_decoder_start_token_id(self, *args, **kwargs)

    def _prepare_decoder_input_ids_for_generation(self, *args, **kwargs):
        return gen_utils._prepare_decoder_input_ids_for_generation(self, *args, **kwargs)

    def generate(self, *args, **kwargs):
        return gen_utils.generate(self, *args, **kwargs)

    def greedy_search(self, *args, **kwargs):
        return gen_utils.greedy_search(self, *args, **kwargs)

    def _get_logits_processor(self, *args, **kwargs):
        return gen_utils._get_logits_processor(self, *args, **kwargs)

def build_transformer(args, vocab=None, intermediate_feature=None):
    # encoder
    encoder_include_heatmap = True if ('loss_aux_heatmap_weight' in args.loss and args.loss.loss_aux_heatmap_weight > 0) else False
    encoder_add_roi_projection = True if ('loss_vis_clip_weight' in args.loss and args.loss.loss_vis_clip_weight > 0) or \
                                         ('loss_vis_clip_softmax_weight' in args.loss and args.loss.loss_vis_clip_softmax_weight > 0) else False
    roi_projection_only_phy = True if ('clip_only_phy' in args.loss and args.loss.clip_only_phy) else False
    if 'type' not in args.model.transformer.encoder or args.model.transformer.encoder.type == 'standard':
        from models.transformer_encoder import TableEncoder
        encoder_cfg = DetrConfig(d_model=args.model.transformer.encoder.hidden_dim,
                                    encoder_attention_heads=args.model.transformer.encoder.nheads,
                                    encoder_layers=args.model.transformer.encoder.enc_layers,
                                    encoder_ffn_dim=args.model.transformer.encoder.dim_feedforward,
                                    is_encoder_decoder=True,
                                    encoder_layerdrop=0, #0.1,
                                    activation_function='gelu',
                                    dropout=args.model.transformer.encoder.dropout,
                                    attention_dropout=0,
                                    position_embedding_type='sine',
                                    use_mask_token=False,
                                    encoder_output='patch',
                                    use_mean_pooling=True,
                                    layer_norm_eps=1e-12,
                                    intermediate_feature=intermediate_feature,
                                    include_heatmap=encoder_include_heatmap,
                                    add_roi_projection=encoder_add_roi_projection,
                                    roi_projection_only_phy=roi_projection_only_phy,
                                )
        encoder = TableEncoder(encoder_cfg, add_pooling_layer=True)
    # elif args.model.transformer.encoder.type == 'vit': # discarded
    #     from models.transformer_encoder_vit import TableEncoder

    #     image_max = args.train.image_max if args.exec.eval == False else args.test.image_max
    #     image_min = args.train.image_min if args.exec.eval == False else args.test.image_min
    #     patch_size = 16

    #     encoder_cfg = ViTConfig(hidden_size=args.model.transformer.encoder.hidden_dim,
    #                             num_hidden_layers=args.model.transformer.encoder.enc_layers,
    #                             num_attention_heads=args.model.transformer.encoder.nheads,
    #                             intermediate_size=args.model.transformer.encoder.dim_feedforward,
    #                             hidden_act='gelu',
    #                             hidden_dropout_prob=args.model.transformer.encoder.dropout,
    #                             attention_probs_dropout_prob=args.model.transformer.encoder.dropout,
    #                             layer_norm_eps=1e-12,
    #                             image_size=image_max,
    #                             patch_size=patch_size,
    #                             num_channels=3,
    #                             encoder_output='patch',
    #                             use_mean_pooling=True,
    #                             )
    #     encoder = TableEncoder(encoder_cfg, add_pooling_layer=True, use_mask_token=False)

    #     if not args.exec.resume:
    #         # load pre-trained model
    #         from transformers import ViTModel
    #         vit_base = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    #         # change positional encoding
    #         from models.transformer_encoder_vit import interpolate_pos_encoding
    #         num_patches = (image_max // patch_size) * (image_max // patch_size) # assume the input image is square!
    #         pos_enc = interpolate_pos_encoding(vit_base.embeddings.position_embeddings, num_patches, patch_size, image_max, image_max)

    #         # exclude pooler if layer_num doesn't match!
    #         if vit_base.config.num_hidden_layers != encoder_cfg.num_hidden_layers:
    #             delattr(vit_base, 'pooler')
    #         vit_base.embeddings.position_embeddings = nn.Parameter(pos_enc)

    #         # update encoder
    #         encoder.load_state_dict(vit_base.state_dict(), strict=False)

    else: # args.encoder.type == 'dit'
        # TODO
        image_max = args.train.image_max if args.exec.eval == False else args.test.image_max
        image_min = args.train.image_min if args.exec.eval == False else args.test.image_min
        encoder_cfg = BeitConfig(attention_probs_dropout_prob=args.model.transformer.encoder.dropout,
                                 hidden_size=args.model.transformer.encoder.hidden_dim,
                                 num_hidden_layers=args.model.transformer.encoder.enc_layers,
                                 num_attention_heads=args.model.transformer.encoder.nheads,
                                 intermediate_size=args.model.transformer.encoder.dim_feedforward,
                                 hidden_act='gelu',
                                 hidden_dropout_prob=args.model.transformer.encoder.dropout,
                                 layer_norm_eps=1e-12,
                                 image_size=image_max,
                                 patch_size=16,
                                 num_channels=3,
                                 use_absolute_position_embeddings=True,
                                 use_mask_token=True,
                                 use_auxiliary_head=False,
                                 use_mean_pooling=True,
                                 semantic_loss_ignore_index=255,
                                 )

    # decoder
    eoh_token_id = vocab['EOH'] if args.model.transformer.decoder.include_header else None

    max_seq_len = args.train.max_seq_len if not args.exec.eval else args.test.max_seq_len

    decoder_cfg = BertConfig(hidden_size=args.model.transformer.decoder.hidden_dim,
                             num_hidden_layers=args.model.transformer.decoder.dec_layers,
                             num_attention_heads=args.model.transformer.decoder.nheads,
                             intermediate_size=args.model.transformer.decoder.dim_feedforward,
                             hidden_act='gelu',
                             hidden_dropout_prob=args.model.transformer.decoder.dropout,
                             attention_probs_dropout_prob=args.model.transformer.decoder.dropout,
                             max_position_embeddings=max_seq_len,
                             max_length=max_seq_len,
                             type_vocab_size=vocab['MAX_CLASS'],
                             layer_norm_eps=1e-5,
                             use_cache=True,
                             position_embedding_type=args.model.transformer.decoder.dec_pos_emb,
                             eos_token_id=vocab['EOS'],
                             bos_token_id=vocab['BOS'],
                             pad_token_id=vocab['PAD'],
                             is_decoder=True,
                             add_cross_attention=True,
                             cross_attention_hidden_size=encoder_cfg.hidden_size,
                             type_offset=vocab['CLASS_OFFSET'],
                             coord_offset=vocab['COORD_SHIFT'],
                             coord_bin=vocab['X_BIN'],
                             sep_token_id=vocab['SEP'],
                             eoh_token_id=eoh_token_id,
                             fusion_attention=True,
                             apply_attension_sharpening=True if (args.exec.eval and args.test.apply_as) else False,
                             )

    decoder = TableDecoder(decoder_cfg, add_pooling_layer=False)

    if 'vocab_file' in args.dataset and args.dataset.vocab_file is not None and 'vision_ocr' in args.model.transformer:
        from .vision_ocr import BaseVision
        vision_decoder = BaseVision(args)
    else:
        vision_decoder = None

    return TableTransformer(encoder=encoder, decoder=decoder, vision_decoder=vision_decoder)

@dataclass
class Seq2SeqLMOutputWithPooling(Seq2SeqLMOutput):
    logits_col: torch.FloatTensor = None
    continuous_logits: torch.FloatTensor = None
    empty_logits: torch.FloatTensor = None    
    encoder_feature_size: Tuple = None
    encoder_attention_mask: torch.FloatTensor = None
    encoder_attention_mask_origin: torch.FloatTensor = None
    encoder_center_heatmap: torch.FloatTensor = None
    vision_decoder_logits: torch.FloatTensor = None

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
    
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias[..., 0], 11.33) # row_num avg
                    torch.nn.init.constant_(layer.bias[..., 1], 4.18) # col_num avg
            else:
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)