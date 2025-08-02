# -----------------------------------------------------------------------
# S2Tab official code : models/network_box_regression_mult_ver2_1_1_empty.py
# -----------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -----------------------------------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
import numpy as np

from util.misc import NestedTensor, is_dist_avail_and_initialized, \
                    nested_tensor_from_tensor_list, get_world_size

from .transformer_contentbox import build_transformer
from .backbone import build_backbone
import models.generation_utils_contentbox as gen_utils
from models.loss_utils import softmax_focal_loss, heatmap_focal_loss


class Tab2seq(nn.Module):
    def __init__(self, backbone, transformer, vocab=None):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer

        hidden_dim = transformer.config.decoder.hidden_size # 512
        self.hidden_dim = hidden_dim
        self.pos_dim = hidden_dim // 4
        self.label_dim = hidden_dim // 4
        
        self.vocab = vocab

        self.num_classes = self.vocab['MAX_CLASS']
        self.max_seq_len = transformer.decoder.config.max_length

    def forward(self, samples: NestedTensor, sources: dict = None, 
                       draw_attention: bool = False, is_train: bool = False):
        """
        Train: teacher-forcing (given targets)
        Inference: auto-regressive decoding
        """
        if is_train:
            return self.train_step(samples, sources)
        else:
            return self.infer(samples, draw_attention, sources)

    def train_step(self, samples: NestedTensor, sources: dict = None):
        """
        - samples.tensor: batched images [N x 3 x H x W]
        - samples.mask: a binary mask of [N x H x W], containing 1 on real pixels
        - sources: dictionary of input sequence
                    'coord': 4 points for top-left and bottom-right corners, total 8 points, (x,y) [N x seq_len x 8 x 2]
                    'delta': bilinear interpolation coefficients [N x seq_len x 8]
                    'token': token class for row and column, decoded in the line 81-98 [N x seq_len]
                    'mask': valid sequence token [N x seq_len]

        Return:
        - 'pred_mu': predicted normalized box coordinates
                    in defined order, representing (x1,y1,w,h). 
        - 'pred_type_logits_row': logits for row classification
        - 'pred_type_logits_col': logits for column classification
        - 'pred_empty_logits: logits for empty-cell classification
        - 'pred_type_sequences_row': row class prediction
        - 'pred_type_sequences_col': column class prediction
        - 'pred_empty_sequences': empty-cell class prediction
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        CLASS_OFFSET = self.vocab['CLASS_OFFSET']
        MAX_COL = self.vocab['MAX_COL'] if 'MAX_COL' in self.vocab else 30

        src_coord = sources['coord'] # input sequence coord vocab - bs,seq_num, 8, 2 (4 * 2 points, x,y each)
        src_delta = sources['delta'] # input sequence delta - bs, seq_num, 8
        src_type = sources['token']
        src_type_row = torch.where(src_type > CLASS_OFFSET, 
                                      (src_type - 1 - CLASS_OFFSET) % MAX_COL + \
                                       1 + CLASS_OFFSET, 
                                      src_type)
        src_type_col = torch.where(src_type > CLASS_OFFSET, 
                                        torch.div(src_type - 1 - CLASS_OFFSET, 
                                                  MAX_COL, rounding_mode='floor') + \
                                        1 + CLASS_OFFSET, 
                                      src_type)

        embeddings, pixel_mask = self.backbone(samples.tensors, samples.mask)

        output_hidden_states = True if 'tgt_roi_boxes' in sources else False

        roi_boxes = sources['tgt_roi_boxes'] if 'tgt_roi_boxes' in sources else None
        image_sizes = sources['tgt_roi_image_sizes'] if 'tgt_roi_image_sizes' in sources else None
        roi_masks = sources['tgt_roi_mask'] if 'tgt_roi_mask' in sources else None
        ocr_roi_option = sources['ocr_roi_option'] if 'ocr_roi_option' in sources else 'ground truth'

        output = self.transformer(embeddings, pixel_mask, \
                                  src_coord, src_delta, src_type_row, src_type_col, \
                                  decoder_attention_mask=sources['mask'], use_cache=True, \
                                  output_hidden_states=output_hidden_states, \
                                  roi_boxes=roi_boxes, \
                                  image_sizes=image_sizes, \
                                  ocr_roi_option=ocr_roi_option, \
                                  roi_masks = roi_masks,
                                )

        mu_logit = output.continuous_logits # (bs, seq_len, 4)
        mu = torch.sigmoid(mu_logit)

        token_type_logits_row = output.logits # (bs, seq_len, token_type_num)
        token_type_logits_col = output.logits_col # (bs, seq_len, token_type_num)
        # empty_logits = torch.sigmoid(output.empty_logits)
        empty_logits = output.empty_logits

        type_sequences_row = token_type_logits_row.argmax(dim=-1)
        type_sequences_col = token_type_logits_col.argmax(dim=-1)
        empty_sequences = torch.greater_equal(empty_logits, 0.5).to(torch.float32) if empty_logits is not None else None

        ## TODO: heatmap from backbone??
        if output.encoder_center_heatmap is not None:
            center_heatmap = output.encoder_center_heatmap
            # center_heatmap = F.sigmoid(center_heatmap)
        else:
            center_heatmap = None

        # visual clip loss
        if 'tgt_roi_boxes' in sources and hasattr(self.transformer.encoder, 'roi_projection'):
            ## roi-aligned feature
            ### sampling points - original size
            pts = sources['tgt_roi_boxes'] # x,y,w,h
            pts[..., 2:] += pts[..., :2]
            tgt_sizes = sources['tgt_roi_image_sizes'] # h,w
            img_h, img_w = tgt_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            pts = pts * scale_fct[:, None, :]
            
            ### sample features
            encoder_hidden_states = output.encoder_last_hidden_state
            encoder_feature_size = output.encoder_feature_size # (h, w)
            reshaped_encoder_hidden_states = encoder_hidden_states.reshape(encoder_hidden_states.shape[0], encoder_feature_size[0], encoder_feature_size[1], -1)
            reshaped_encoder_hidden_states = reshaped_encoder_hidden_states.permute(0, 3, 1, 2)

            ### adjust points to the feature size
            padded_sample_size = list(samples.tensors.shape[-2:])
            ratio_h, ratio_w = encoder_feature_size[0] / padded_sample_size[0], encoder_feature_size[1] / padded_sample_size[1] # h, w ratio
            ratio = torch.as_tensor([ratio_w, ratio_h, ratio_w, ratio_h], device=pts.device) # 1/16 mostly, but it depends on the real input size due to quantization error
            pts = pts * ratio[None, None, :]
            
            #### to prevent exceeding resized region
            encoder_mask = output.encoder_attention_mask.reshape(encoder_hidden_states.shape[0], encoder_feature_size[0], encoder_feature_size[1]) # bs, h', w'
            clip_size = [x.nonzero() for x in encoder_mask]
            clip_size = [[x[:, 1].max().item(), x[:, 0].max().item()] for x in clip_size] # w, h
            clip_size = torch.as_tensor(clip_size, device=pts.device).tile((1, 2)) # bs, 4 (w,h,w,h)
            pts = pts.clamp(max=clip_size[:,None,:])

            rois = torchvision.ops.roi_align(reshaped_encoder_hidden_states, list(pts), (2, 2))
            rois = self.transformer.encoder.roi_projection(rois) # bs * n, c, 2, 2 -> bs * n, c, 1, 1
            rois = rois.view(pts.shape[0], pts.shape[1], -1) # bs, n, c

            ## decoder hidden states
            decoder_hidden_states = output.decoder_hidden_states[-1] # bs, n, c

            if rois.shape[2] == decoder_hidden_states.shape[2] // 2: # only phy
                decoder_hidden_states = decoder_hidden_states[:, :, decoder_hidden_states.shape[2]//2:]
        else:
            rois, decoder_hidden_states = None, None

        # OCR
        if 'vision_decoder_logits' in output and output.vision_decoder_logits is not None:
            ocr_logits = output.vision_decoder_logits
        else:
            ocr_logits = None

        out = {
               'pred_mu': mu,
               'pred_type_logits_row': token_type_logits_row,
               'pred_type_logits_col': token_type_logits_col,
               'pred_empty_logits': empty_logits,
               'pred_type_sequences_row': type_sequences_row,
               'pred_type_sequences_col': type_sequences_col,
               'pred_empty_sequences': empty_sequences,
               'pred_center_heatmap': center_heatmap,
               'tgt_roi_feature': rois,
               'decoder_hidden_states': decoder_hidden_states,
               'pred_ocr_logits': ocr_logits,
               }
        
        return out

    def infer(self, samples: NestedTensor, draw_attention: bool=False, sources: dict=None):
        """
        - samples.tensor: batched images [N x 3 x H x W]
        - samples.mask: a binary mask of [N x H x W], containing 1 on real pixels
        - draw_attention: draw cross attention or not

        Return:
        - 'pred_type_logits_row': logits for row classification
        - 'pred_type_logits_col': logits for column classification
        - 'pred_empty_logits: logits for empty-cell classification
        - 'pred_coord_sequences': predicted normalized box coordinates
                                    in defined order, representing (x1,y1,w,h). 
        - 'pred_type_sequences_row': row class prediction
        - 'pred_type_sequences_col': column class prediction
        - 'cross_attn', 'self_attn', 'decoder_hidden_state': optional values for visualization
        """

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        table_logit_processor = gen_utils.LogitsProcessorListMultiTask(
                                        [gen_utils.TableLogitProcessor(self.transformer.decoder.config.type_vocab_size, 
                                                                       self.transformer.decoder.config.coord_bin,
                                                                       self.transformer.decoder.config.coord_offset,)])
        
        embeddings, pixel_mask = self.backbone(samples.tensors, samples.mask)

        image_sizes = sources['tgt_roi_image_sizes'] if 'tgt_roi_image_sizes' in sources else None
        ocr_roi_option = 'prediction'

        output = self.transformer.generate(embeddings,
                                            pixel_mask=pixel_mask,
                                            use_cache=True,
                                            logits_processor=table_logit_processor,
                                            return_dict_in_generate=True,
                                            output_scores=True,
                                            max_length=self.max_seq_len,
                                            output_attentions=draw_attention,
                                            output_hidden_states=True,
                                            image_sizes=image_sizes,
                                            ocr_roi_option=ocr_roi_option,
                                            )
        

        output_sequences = output.sequences
        scores = output.scores

        # coordinates
        box_logit = torch.stack(scores['coord_logits'], dim=1)
        mu = torch.sigmoid(box_logit)

        # classes
        token_type_row = torch.stack(scores['type_scores_row'], dim=1)
        token_type_col = torch.stack(scores['type_scores_col'], dim=1)
        if 'empty_scores' in scores:
            empty = torch.stack(scores['empty_scores'], dim=1)
        else:
            empty = None
        
        if output.decoder_hidden_states is not None:
            decoder_last_hidden_state = torch.cat([x[-1] for x in output.decoder_hidden_states], dim=1)
        else:
            decoder_last_hidden_state = None

        # OCR
        if 'ocr_logits' in scores and len(scores['ocr_logits']) > 0:
            ocr_logits = torch.stack(scores['ocr_logits'], dim=1) # (B, N, max_token_length, class)
        else:
            ocr_logits = None

        out = {
               'pred_type_logits_row': token_type_row,
               'pred_type_logits_col': token_type_col,
               'pred_empty_logits': empty,
               'pred_coord_sequences': mu,
               'pred_type_sequences_row': output_sequences[1][:, 1:],
               'pred_type_sequences_col': output_sequences[2][:, 1:],
               'cross_attn': (output.encoder_feature_size, output.cross_attentions),
               'self_attn': output.decoder_attentions,
               'decoder_hidden_state': decoder_last_hidden_state,
               'pred_ocr_logits': ocr_logits,
            }

        return out


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


class SeqCriterion(nn.Module):
    def __init__(self, 
                 num_classes, 
                 weight_dict, 
                 losses, 
                 vocab=None,
                 gamma=2,
                 initial_t=None, #10,
                 initial_b=None, #10,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.losses = losses

        self.CLASS_OFFSET = vocab['CLASS_OFFSET']
        self.MAX_COL = vocab['MAX_COL'] if 'MAX_COL' in vocab else 30
        self.ENDROW_OFFSET = 1000 if self.MAX_COL == 30 else 2500 # 30 or 50

        self.vocab = vocab
        self.focal_gamma = gamma

        if initial_t is not None and initial_b is not None:
            self.t = torch.nn.Parameter(torch.log(torch.zeros(1) + initial_t))
            self.b = torch.nn.Parameter(torch.zeros(1) + initial_b)
    
    def loss_labels_focal(self, outputs, targets, **kwargs):
        """ Compute the loss for class labels: focal loss """
        assert 'pred_type_logits_row' in outputs and \
                'pred_type_logits_col' in outputs
        assert 'token_weights' in targets[0]
        src_logits_row = outputs['pred_type_logits_row']
        src_logits_col = outputs['pred_type_logits_col']

        src_logits_row = src_logits_row.permute(0, 2, 1)
        src_logits_col = src_logits_col.permute(0, 2, 1)
        tgt_classes = torch.stack([t['target_seq']['seq']['token'] for t in targets], dim=0).to(torch.int64) # (bs, seq)

        CLASS_OFFSET = self.CLASS_OFFSET
        MAX_COL = self.MAX_COL
        ENDROW_OFFSET = self.ENDROW_OFFSET
        tgt_classes_row = torch.where(tgt_classes > CLASS_OFFSET, 
                                      (tgt_classes - 1 - CLASS_OFFSET) % MAX_COL + 1 + CLASS_OFFSET, 
                                      tgt_classes)
        tgt_classes_row = torch.where(tgt_classes > CLASS_OFFSET+ENDROW_OFFSET, 
                                    (tgt_classes - 1 - CLASS_OFFSET-ENDROW_OFFSET) % MAX_COL + \
                                        1 + CLASS_OFFSET + MAX_COL, 
                                    tgt_classes_row)
        
        tgt_classes_col = torch.where(tgt_classes > CLASS_OFFSET, 
                                        torch.div(tgt_classes - 1 - CLASS_OFFSET, 
                                                  MAX_COL, rounding_mode='floor') + \
                                        1 + CLASS_OFFSET,
                                      tgt_classes)
        tgt_classes_col = torch.where(tgt_classes > CLASS_OFFSET+ENDROW_OFFSET, 
                                        torch.div(tgt_classes - 1 - CLASS_OFFSET - ENDROW_OFFSET, 
                                                  MAX_COL, rounding_mode='floor') + \
                                        1 + CLASS_OFFSET + MAX_COL, 
                                    tgt_classes_col)

        if 'pred_empty_logits' in outputs:
            src_empty_logits = outputs['pred_empty_logits']
            tgt_empty_classes = torch.stack([t['target_seq']['seq']['empty'] for t in targets], dim=0)
        token_weights = torch.stack([t['token_weights'] for t in targets], dim=0) # (bs, seq)

        loss_fce_row = softmax_focal_loss(src_logits_row, tgt_classes_row, gamma=self.focal_gamma, reduction='none', num_classes=self.num_classes)
        loss_fce_col = softmax_focal_loss(src_logits_col, tgt_classes_col, gamma=self.focal_gamma, reduction='none', num_classes=self.num_classes)
        
        loss_fce = loss_fce_row + loss_fce_col

        loss_fce = loss_fce * token_weights

        if 'pred_empty_logits' in outputs:
            # loss_bce_h = torchvision.ops.sigmoid_focal_loss(src_empty_logits, tgt_empty_classes.unsqueeze(2), reduction='none') # alpha 0.25 gamma 2
            loss_bce_h = F.binary_cross_entropy_with_logits(src_empty_logits, tgt_empty_classes.unsqueeze(2), reduction='none')
            loss_bce_h = loss_bce_h * token_weights[...,None]

        losses = {}
        losses['loss_class'] = loss_fce.sum() / (token_weights != 0).sum()

        if 'pred_empty_logits' in outputs:
            losses['loss_empty_class'] = loss_bce_h.sum() / (token_weights != 0).sum()

        return losses, {}

    def loss_boxes(self, outputs, targets, **kwargs):
        """ Compute the localization loss : L1 loss + generalized IoU loss """
        assert 'pred_mu' in outputs
        src_mu = outputs['pred_mu'] # (bs, seq_len, 4)

        boxes = [t['target_seq']['seq']['coord'] for t in targets]
        tgt_boxes = torch.stack([b for b in boxes], dim=0)

        valid = torch.stack([t['target_seq']['mask_box'] for t in targets], dim=0)

        # L1 loss
        # loss = F.smooth_l1_loss(src_mu, tgt_boxes, reduction='none')
        loss = F.l1_loss(src_mu, tgt_boxes, reduction='none')
        loss = loss.sum(2) * valid
        losses = {}
        losses['loss_boxes'] = loss.sum() / valid.sum()

        # complete iou loss - xywh
        src_pred = xywh_to_xyxy(src_mu.clone())
        tgt_boxes = xywh_to_xyxy(tgt_boxes)

        # iou_loss = torchvision.ops.complete_box_iou_loss(src_pred, tgt_boxes, reduction='none')
        iou_loss = torchvision.ops.generalized_box_iou_loss(src_pred, tgt_boxes, reduction='none')
        iou_loss = iou_loss[valid] # to remove invalid values 

        losses['loss_boxes_giou'] = iou_loss.sum() / valid.sum()

        return losses, {}
    
    def loss_aux_heatmap_regression(self, outputs, targets, **kwargs):
        assert 'pred_center_heatmap' in outputs
        assert 'heatmap' in kwargs
        src_heatmap = outputs['pred_center_heatmap']

        tgt_heatmap_nested = kwargs['heatmap']
        tgt_heatmap, valid_mask = tgt_heatmap_nested.tensors, tgt_heatmap_nested.mask

        h, w = src_heatmap.shape[-2:]
        tgt_heatmap = torchvision.transforms.functional.resize(tgt_heatmap, 
                                                               [h, w], 
                                                               torchvision.transforms.InterpolationMode.NEAREST)
        valid_mask = torchvision.transforms.functional.resize(valid_mask.to(tgt_heatmap.dtype), 
                                                               [h, w], 
                                                               torchvision.transforms.InterpolationMode.NEAREST)
        
        cell_nums = torch.stack([x['target_seq']['mask_box'].sum() for x in targets])
        positive_thresholds = torch.ones_like(cell_nums)

        loss = heatmap_focal_loss(src_heatmap, tgt_heatmap, gamma=2, beta=4, reduction='none', positive_thresholds=positive_thresholds)
        loss = loss * valid_mask[:, None, :, :]

        losses = {}
        losses['loss_aux_heatmap'] = (loss.sum(-1).sum(-1).sum(-1) / cell_nums).mean()

        return losses, {'tgt_heatmap': tgt_heatmap, 
                        'tgt_positive_mask': torch.greater_equal(tgt_heatmap, positive_thresholds[:, None, None, None]).to(tgt_heatmap.dtype),
                        }

    def loss_aux_visual_clip_softmax(self, outputs, targets, **kwargs):
        #### Note that this loss function includes empty boxes also!!!
        assert 'tgt_roi_feature' in outputs and outputs['tgt_roi_feature'] is not None
        assert 'decoder_hidden_states' in outputs and outputs['decoder_hidden_states'] is not None
        src_features = outputs['decoder_hidden_states'] # bs, c, n
        tgt_features = outputs['tgt_roi_feature'] # bs, c, n

        valid_mask = torch.stack([t['target_seq']['mask_box'] for t in targets], dim=0)

        # softmax-version
        ## normalize features
        src_features = F.normalize(src_features, p=2, dim=1)
        tgt_features = F.normalize(tgt_features, p=2, dim=1)

        ## calculate similarities (bs, n, C) x (bs, C, n) = (bs, n, n)
        S = torch.matmul(tgt_features, src_features.transpose(-1, -2)) # S[:,j,i] = <tgt[:,j], src[:,i]>
        tau = 0.04
        S = S / tau # temperature scaling
        ### mask
        mask = valid_mask[:,:,None] * valid_mask[:,None, :] # bs, n, n
        S = torch.where(mask, S, torch.ones_like(S) * torch.finfo(S.dtype).min) # fill -max value to invalid elements

        ## target positive/negative label - regardless of validity
        T = torch.arange(0, S.shape[1], dtype=torch.long, device=S.device).tile((S.shape[0], 1)) # bs, n

        ## InfoNCE Loss
        loss = F.cross_entropy(S, T, reduction='none')
        loss = loss * valid_mask

        losses = {}
        losses['loss_vis_clip_softmax'] = loss.sum() / valid_mask.sum()

        return losses, {}

    def loss_aux_visual_clip(self, outputs, targets, **kwargs):
        #### Note that this loss function includes empty boxes also!!!
        assert 'tgt_roi_feature' in outputs and outputs['tgt_roi_feature'] is not None
        assert 'decoder_hidden_states' in outputs and outputs['decoder_hidden_states'] is not None
        src_features = outputs['decoder_hidden_states'] # bs, n, c
        tgt_features = outputs['tgt_roi_feature'] # bs, n, c

        valid_mask = torch.stack([t['target_seq']['mask_box'] for t in targets], dim=0)

        # softmax-version
        ## normalize features
        src_features = F.normalize(src_features, p=2, dim=-1)
        tgt_features = F.normalize(tgt_features, p=2, dim=-1)

        ## calculate similarities (bs, n, C) x (bs, C, n) = (bs, n, n)
        S = torch.matmul(tgt_features, src_features.transpose(-1, -2)) # S[:,j,i] = <tgt[:,j], src[:,i]>
        S = S * torch.exp(self.t) + self.b
        ### mask
        mask = valid_mask[:,:,None] * valid_mask[:,None, :] # bs, n, n
        S = torch.where(mask, S, torch.ones_like(S) * torch.finfo(S.dtype).min) # fill -max value to invalid elements

        ## target positive/negative label - regardless of validity
        T = 2 * torch.eye(S.shape[1], S.shape[1]).tile((S.shape[0], 1, 1)).to(S.device) - torch.ones_like(S) # bs, n, n (1 for positive, -1 for negative)
        # T = torch.arange(0, S.shape[1], dtype=torch.long, device=S.device).tile((S.shape[0], 1)) # bs, n

        ## Sigmoid CLIP Loss - one-directional version
        loss = -F.logsigmoid(T * S)
        loss = loss * mask

        losses = {}
        losses['loss_vis_clip'] = loss.sum() / mask.sum()

        return losses, {}
    
    def loss_ocr_cross_entropy(self, outputs, targets, **kwargs):
        assert 'content' in targets[0] and 'pred_ocr_logits' in outputs
        pad_idx = targets[0]['pad_idx']
        src = outputs['pred_ocr_logits'] # (B, N, T, C) N = max cell num, T = max seq len
        src = src.permute(0, 3, 1, 2) # (B, C, N, T)
        tgt = nested_tensor_from_tensor_list([t['content'] for t in targets], fill=pad_idx).tensors # (B, N_cell, T)
        tgt_len = tgt.shape[1]
        src = src[:, :, :tgt_len, :] # (B, C, N_cell, T)

        valid_mask_elem = torch.stack([t['target_seq']['mask_box'] for t in targets], dim=0) # (B, N)
        valid_mask_elem = valid_mask_elem[:, :tgt_len] # (B, N_cell)
        
        valid_mask = tgt != pad_idx
        valid_mask = valid_mask * valid_mask_elem[..., None]

        loss = F.cross_entropy(src, tgt, reduction='none')
        loss = loss * valid_mask

        losses = {}
        losses['loss_ocr'] = loss.sum() / valid_mask.sum()

        return losses, {}


    def get_loss(self, loss, outputs, targets, **kwargs):
        loss_map = {
            'labels_class': self.loss_labels_focal,
            'boxes': self.loss_boxes,
            'aux_heatmap': self.loss_aux_heatmap_regression,
            'aux_vis_clip': self.loss_aux_visual_clip,
            'aux_vis_clip_softmax': self.loss_aux_visual_clip_softmax,
            'aux_ocr': self.loss_ocr_cross_entropy,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        # Compute the average number of target boxes
        num_boxes = [len(t["labels"]) for t in targets]
        if 'pred_type_logits_row' in outputs:
            pred_seq_len = outputs['pred_type_logits_row'].shape[1] #.tensors.shape[1]
            num_boxes = [min(x, pred_seq_len) for x in num_boxes]
        num_boxes = sum(num_boxes)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs['pred_type_logits_row'].device) # .tensors

        if is_dist_avail_and_initialized():
            dist.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        aux = {}
        for loss in self.losses:
            computed_losses, aux_single = self.get_loss(loss, outputs, targets, **kwargs)
            losses.update(computed_losses)
            aux.update(aux_single)

        return losses, aux

class Postprocessor(nn.Module):
    """ convert sorted cell boxes into table structure """
    def __init__(self, enc_config=None, dec_config=None, vocab=None, tokenizer=None):
        super().__init__()
        if dec_config is not None:
            self.type_vocab_size = dec_config.type_vocab_size
            self.layer_num = dec_config.num_hidden_layers
        else:
            self.type_vocab_size = 8
            self.layer_num = 3
        if enc_config is not None:
            self.encoder_output = enc_config.encoder_output
            self.use_mean_pooling = enc_config.use_mean_pooling
        else:
            self.encoder_output = 'patch'
            self.use_mean_pooling = False
        self.vocab = vocab
        self.tokenizer = tokenizer

    @torch.no_grad()
    def forward(self, outputs, target_sizes, pp=False):
        """
        outputs: refer to output dict from Tab2seq.infer()
        target_sizes: [batch x 2] of (h,w)

        returns:
        res = list of dict, of {'boxes': cell boxes,
                                'scores': classification scores, 
                                'labels': class (row and column), 
                                'structures': [start-col, start-row, end-col, end-row, is-header], 
                                'mask': (for visualization) cell token (True) or special token (False), 
                                'all_boxes': (for visualization) box prediction including special token, 
                                'empty': empty-cell class
                                }
        """
        res = []
        CLASS_OFFSET = self.vocab['CLASS_OFFSET']
        EOS = self.vocab['EOS']
        PAD = self.vocab['PAD']
        SEP = self.vocab['SEP']
        EOH = self.vocab['EOH']
        MAX_COL = self.vocab['MAX_COL'] if 'MAX_COL' in self.vocab else 30
        ENDROW_OFFSET = 1000 if MAX_COL == 30 else 2500 # 50

        # Convert class logits into labels
        logits_row = outputs['pred_type_logits_row']
        logits_col = outputs['pred_type_logits_col']
        labels_row = outputs['pred_type_sequences_row'] # (batch, num_boxes)
        labels_col = outputs['pred_type_sequences_col'] # (batch, num_boxes)
        empty_logits = outputs['pred_empty_logits']
        empty_labels = torch.greater_equal(empty_logits, 0.5).to(torch.float32).squeeze(2)
        scores_row = torch.gather(torch.softmax(logits_row, 2), 2, labels_row[:, :, None]).squeeze(2)
        scores_col = torch.gather(torch.softmax(logits_col, 2), 2, labels_col[:, :, None]).squeeze(2)
        bs = labels_row.shape[0]

        boxes = outputs['pred_coord_sequences'] # (batch, num_boxes, 4) x,y,w,h or cx,cy,w,h
        boxes = box_quantize(boxes, target_sizes) # (batch, num_boxes, 4)
        boxes[..., 2:] += boxes[..., :2]  #xwyh_to_xyxy(boxes)

        logits_ocr = outputs['pred_ocr_logits']
        if logits_ocr is not None:
            labels_ocr = logits_ocr.argmax(-1) # (batch, num_boxes, token_length)
        else:
            labels_ocr = None

        orig_size = target_sizes

        
        valid_boxes = []
        structures = []
        valid_labels = []
        valid_endrow_labels = []
        valid_scores = []
        valid_masks = [] # viaulization
        valid_empty = []
        valid_ocr = []
        all_boxes = [] # visualize attn
        all_indices = []
        valid_ind = []
        for bi in range(bs):
            box = boxes[bi]
            score_row = scores_row[bi]
            score_col = scores_col[bi]
            lab_row = labels_row[bi]
            lab_col = labels_col[bi]
            empty = empty_labels[bi]
            if labels_ocr is not None:
                lab_ocr = labels_ocr[bi]

            sz = orig_size[bi].cpu().tolist() # h,w
            if (sum(lab_row == EOS) == 0 and sum(lab_row == PAD) == 0):
                end_idx_r = lab_row.shape[0]
            elif sum(lab_row == EOS) == 0 and sum(lab_row == PAD) > 0:
                end_idx_r = torch.argmin((lab_row != PAD).to(torch.float32)).item() # PAD가 시작되는 지점
            else:
                end_idx_r = torch.argmin((lab_row != EOS).to(torch.float32)).item() # EOS가 시작되는 지점

            if (sum(lab_col == EOS) == 0 and sum(lab_col == PAD) == 0):
                end_idx_c = lab_col.shape[0]
            elif sum(lab_col == EOS) == 0 and sum(lab_col == PAD) > 0:
                end_idx_c = torch.argmin((lab_col != PAD).to(torch.float32)).item() # PAD가 시작되는 지점
            else:
                end_idx_c = torch.argmin((lab_col != EOS).to(torch.float32)).item() # EOS가 시작되는 지점
            end_idx = min(end_idx_r, end_idx_c)

            box = box[:end_idx] # (seq, 4)
            score_row = score_row[:end_idx] # (seq)
            score_col = score_col[:end_idx] # (seq)
            lab_row = lab_row[:end_idx] # (seq)
            lab_col = lab_col[:end_idx] # (seq)
            empty = empty[:end_idx]
            structure = []
            if labels_ocr is not None:
                lab_ocr = lab_ocr[:end_idx]
            else:
                lab_ocr = None

            keep_ind = torch.arange(end_idx)
            valid_ind.append(keep_ind.tolist()) 
            all_indices.append(torch.arange(end_idx).tolist())
                    
            if pp == 1:
                rowspan = lab_row - CLASS_OFFSET - 1
                rowspan = torch.where(rowspan >= 0, rowspan % MAX_COL, rowspan).tolist()
                colspan = lab_col - CLASS_OFFSET - 1
                colspan = torch.where(colspan >= 0, colspan % MAX_COL, colspan).tolist()
                valid_mask = (np.asarray(rowspan) >= 0) & (np.asarray(colspan) >= 0)
                endrow_lab = []
                structure = []
                assigned = {}
                cur_r, cur_c = 0, 0
                i = 0
                valid_ind = []
                while i < len(rowspan):
                    r = rowspan[i]
                    c = colspan[i]
                    if i + 1 < len(rowspan) and (rowspan[i+1] + CLASS_OFFSET + 1 == EOH or
                                                 colspan[i+1] + CLASS_OFFSET + 1 == EOH):
                        h = 1
                    else:
                        h = 0

                    if i + 1 < len(rowspan) and (rowspan[i+1] + CLASS_OFFSET + 1 == SEP or
                                                 colspan[i+1] + CLASS_OFFSET + 1 == SEP or 
                                                 h == 1):
                        e = 1
                    elif r <= 0 or c <= 0: # invalid cell!
                        i += 1
                        continue
                    else:
                        e = 0
                    cell_structure = [cur_c, cur_r, cur_c + c - 1, cur_r + r - 1, h] # sc, sr, ec, er, header
                    structure.append(cell_structure)
                    for rr in range(cur_r, cur_r + r):
                        assigned[rr] = assigned.get(rr, []) + list(range(cur_c, cur_c + c))
                    endrow_lab.append(e)
                    valid_ind.append(i)

                    # update cur_c
                    cur_c += c
                    if e == 1:
                        cur_r += 1
                        # reset cur_c
                        cur_c = min([0] + assigned.get(cur_r, [0]))
                        i += 1 # skip the next token
                    i += 1
                colnums = -1
                for k,v in assigned.items():
                    if colnums == -1:
                        colnums = max(v)
                        
                bxs = box[valid_ind].cpu().numpy()
                bxs[..., 2:] -= bxs[..., :2]
                endrow_lab = torch.as_tensor(endrow_lab, device=lab_row.device, dtype=lab_row.dtype)
                lab_row = lab_row[valid_ind]
                lab_col = lab_col[valid_ind]
                score_row = score_row[valid_ind]
                score_col = score_col[valid_ind]
                empty = empty[valid_ind]
                if lab_ocr is not None:
                    lab_ocr = lab_ocr[valid_ind]
                    lab_ocr = [self.tokenizer.get_text(o.tolist(), padding=False, trim=True) for o in lab_ocr]
            else:
                box_ind = (lab_row != SEP) * (lab_col != SEP) * (lab_row != EOH) * (lab_col != EOH)
                _endrow_lab = torch.ones_like(box_ind.to(torch.long)) - box_ind.to(torch.long)
                segment_ind = (box_ind == 0).nonzero(as_tuple=True)[0]
                segment_ind = torch.maximum(torch.zeros_like(segment_ind), segment_ind.sub(1))
                endrow_lab = _endrow_lab.clone()
                endrow_lab[segment_ind] = 1
                endrow_lab = endrow_lab[box_ind]

                bxs = box[box_ind].cpu().numpy()
                bxs[..., 2:] -= bxs[..., :2]

                lab_row = lab_row[box_ind]
                lab_col = lab_col[box_ind]
                score_row = score_row[box_ind]
                score_col = score_col[box_ind]
                valid_mask = []
                empty = empty[box_ind]
                if lab_ocr is not None:
                    lab_ocr = lab_ocr[box_ind]
                    lab_ocr = [self.tokenizer.get_text(o.tolist(), padding=False, trim=True) for o in lab_ocr]

            structures.append(structure)
            valid_boxes.append(bxs)
            all_boxes.append(box.cpu().numpy())
            lab_row = torch.maximum(lab_row - CLASS_OFFSET - 1, torch.zeros_like(lab_row))
            lab_col = torch.maximum(lab_col - CLASS_OFFSET - 1, torch.zeros_like(lab_col))
            valid_labels.append(lab_row + lab_col * MAX_COL + 1)
            valid_endrow_labels.append(endrow_lab)
            valid_scores.append(0.5 * (score_row + score_col) )
            valid_masks.append(valid_mask)
            valid_empty.append(empty)
            valid_ocr.append(lab_ocr)

        for box, lab, endrow_lab, score, structure, mask, all_box, empty, ocr in \
            zip(valid_boxes, valid_labels, valid_endrow_labels, valid_scores, structures, valid_masks, all_boxes, valid_empty, valid_ocr):
            lab += endrow_lab * ENDROW_OFFSET
            coco_lab = lab

            res.append({'boxes': box, # (max_seq_len, 4)
                        'labels': coco_lab, # for coco evaluation
                        'scores': score,
                        'structures': structure,
                        'mask': mask,
                        'all_boxes': all_box,
                        'empty': empty,
                        'tokens': ocr,
                        })

        if 'cross_attn' in outputs and outputs['cross_attn'][1] is not None:
            batch_image_size = outputs['input_size'] # h, w
            input_sizes = outputs['input_sizes']
            enc_ft_size, cross_attn = outputs['cross_attn']
            enc_ft_h, enc_ft_w = enc_ft_size
            ratio = max(batch_image_size) / max(enc_ft_size)
            cross_attn = [x[self.layer_num-1] for x in cross_attn] # 0] for x in cross_attn] # 
            cross_attn = [x.reshape(bs, -1, enc_ft_h, enc_ft_w) for x in cross_attn]
            
            cross_attn = [[x[:, :2].mean(1), x[:, 2:4].mean(1), x[:, 4:].mean(1)] for x in cross_attn]
            cross_attn = torch.stack([torch.stack(x, 1) for x in cross_attn], 1) # bs, num_seq, 3, h, w
            cross_attn = [x[all_indices[i]] for i, x in enumerate(cross_attn)]

            for i in range(len(cross_attn)):
                cross_attn_i = cross_attn[i].clone()
                cross_attn_i = torchvision.transforms.functional.resize(cross_attn_i, batch_image_size)
                cross_attn_i = cross_attn_i[:, :, :input_sizes[i][0], :input_sizes[i][1]]
                res[i]['cross_attn'] = cross_attn_i.cpu()

        if 'self_attn' in outputs and outputs['self_attn'] is not None:
            self_attn = outputs['self_attn'] # tuple of each decoder layer's attn map (bs, n_head, 1, num_key)
            self_attn = [x[self.layer_num-1] for x in self_attn] # 0] for x in self_attn] #
            self_attn = [x.squeeze(2) for x in self_attn]
            self_attn = [[x[:, :2].mean(1), x[:, 2:4].mean(1), x[:, 4:].mean(1)] for x in self_attn]
            self_attn = [torch.stack(x, 2) for x in self_attn]
            
            for i in range(len(self_attn[0])):
                self_attn_i = [x[i].cpu() for x in self_attn]
                res[i]['self_attn'] = self_attn_i

        return res

def build(args, vocab=None):
    backbone = build_backbone(args)

    transformer = build_transformer(args, vocab, backbone.intermediate_feature)

    model = Tab2seq(backbone, transformer, vocab=vocab) #, num_classes)

    loss_weight = {
                   'loss_class': args.loss.loss_class_weight,
                   'loss_empty_class': args.loss.loss_class_weight,
                    'loss_boxes': args.loss.loss_box_weight,
                    'loss_boxes_giou': args.loss.loss_box_giou_weight,
                    'loss_aux_heatmap': args.loss.loss_aux_heatmap_weight if 'loss_aux_heatmap_weight' in args.loss else 0.0,
                    'loss_vis_clip': args.loss.loss_vis_clip_weight if 'loss_vis_clip_weight' in args.loss else 0.0,
                    'loss_vis_clip_softmax': args.loss.loss_vis_clip_softmax_weight if 'loss_vis_clip_softmax_weight' in args.loss else 0.0,
                    'loss_ocr': args.loss.loss_ocr_weight if 'loss_ocr_weight' in args.loss else 0.0,
                    }
    loss_list = ['labels_class', 'boxes']

    initial_t, initial_b = None, None
    if 'loss_aux_heatmap_weight' in args.loss and args.loss.loss_aux_heatmap_weight > 0:
        loss_list += ['aux_heatmap']
    if 'loss_vis_clip_weight' in args.loss and args.loss.loss_vis_clip_weight > 0:
        loss_list += ['aux_vis_clip']
        initial_t = args.loss.sigmoid_clip_initial_t
        initial_b = args.loss.sigmoid_clip_initial_t
    if 'loss_vis_clip_softmax_weight' in args.loss and args.loss.loss_vis_clip_softmax_weight > 0:
        loss_list += ['aux_vis_clip_softmax']
    if 'loss_ocr_weight' in args.loss and args.loss.loss_ocr_weight > 0:
        loss_list += ['aux_ocr']

    criterion = SeqCriterion(model.num_classes, loss_weight, loss_list, 
                             vocab, args.loss.focal_gamma, 
                             initial_t, initial_b)

    if hasattr(model.transformer, 'vision_decoder') and model.transformer.vision_decoder is not None:
        tokenizer = model.transformer.vision_decoder.charset
    else:
        tokenizer = None
    postprocessor = Postprocessor(transformer.encoder.config, transformer.decoder.config, vocab, tokenizer)

    return model, criterion, postprocessor


def box_quantize(box, orig_size):
    """
    box: (batch, sequence_len, 4) [x,y,w,h] #[x1,x2,y1,y2]
    orig_size: (batch, 2) # (h,w)

    return quantized box of x \in [0, w), y \in [0, h)
    """
    img_h, img_w = orig_size.unbind(1)
    scale_fct = torch.stack((img_w, img_h, img_w, img_h), dim=-1).to(box.device)
    if len(box.shape) == 3:
        scale_fct = scale_fct[:, None, :]
    quantized_box = box * scale_fct

    for qb, sz in zip(quantized_box, orig_size):
        h, w = sz
        qb[..., 0::2].clamp_(max = w - 1)
        qb[..., 1::2].clamp_(max = h - 1)
    
    return quantized_box

def xywh_to_xyxy(boxes: torch.Tensor=None):
    _boxes = boxes.clone()
    _boxes = torch.cat((_boxes[..., :2], _boxes[..., 2:]), dim=-1)
    _boxes[..., 2:] += _boxes[..., :2]
    return _boxes

def xwyh_to_xyxy(boxes: torch.Tensor=None):
    _boxes = boxes.clone()
    _boxes = torch.cat((_boxes[..., 0::2], _boxes[..., 1::2]), dim=-1)
    _boxes[..., 2:] += _boxes[..., :2]
    return _boxes
