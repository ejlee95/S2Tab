
# -----------------------------------------------------------------------
# S2Tab official code : datasets/sequence_box_regression_quant_empty.py
# -----------------------------------------------------------------------
# Modified from Pix2Seq (https://github.com/google-research/pix2seq.git)
# Copyright 2022 The Pix2Seq Authors.
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

import torch

def expand_delta(_delta, delta, indices, last_idx=None):
    for i, idx in enumerate(indices):
        delta = torch.cat((delta, _delta[idx]), dim=0)
        delta = torch.cat((delta, torch.as_tensor([0])), dim=0) # [0] for <sep>
    if last_idx is not None:
        delta = torch.cat((delta, _delta[last_idx]), dim=0)
    else:
        delta = delta[:-1] # no <sep> at the end
    delta = delta = torch.cat((delta, torch.as_tensor([0])), dim=0) # [0] for <eos>

    return delta

def expand_quant_point(_quant_point, quant_point, indices, last_idx=None, header_end_idx=None, COORD_SHIFT=None, SEP=None, EOH=None, EOS=None):
    for i, idx in enumerate(indices):
        quant_point = torch.cat((quant_point, _quant_point[idx] + COORD_SHIFT), dim=0)
        if header_end_idx is not None and i == header_end_idx:
            quant_point = torch.cat((quant_point, torch.as_tensor([[EOH, EOH]])), dim=0) # <end-of-header>
        else:
            quant_point = torch.cat((quant_point, torch.as_tensor([[SEP, SEP]])), dim=0) # <sep>
    if last_idx is not None:
        quant_point = torch.cat((quant_point, _quant_point[last_idx] + COORD_SHIFT), dim=0)
    else:
        quant_point = quant_point[:-1] # no <sep> at the end
    quant_point = torch.cat((quant_point, torch.as_tensor([[EOS, EOS]])), dim=0) # <eos>

    return quant_point

def reconstruction_seq_from_boxes(target, max_seq_len, 
                                    loss_eos_token_weight=0.1, 
                                    vocab=None,
                                    enable_multi_tables=False):
    """
    Make source/target sequences from annotations
    """
    X_BIN, Y_BIN = vocab['X_BIN'], vocab['Y_BIN']
    BOS, EOS, FAKE, SEP, PAD, EOH = vocab['BOS'], vocab['EOS'], vocab['FAKE'], vocab['SEP'], vocab['PAD'], vocab['EOH']
    CLASS_OFFSET, MAX_CLASS, COORD_SHIFT = vocab['CLASS_OFFSET'], vocab['MAX_CLASS'], vocab['COORD_SHIFT']

    # indexing '<sep>'
    endrow = target['endrow'].clone()
    ori_seqlen = len(endrow) # ignore <bos> and <eos>
    sep_idx = endrow.nonzero(as_tuple=False).squeeze(1) + 1
    sep_idx = torch.cat((torch.as_tensor([0]), sep_idx), dim=0)
    sep_idx = sep_idx[1:] - sep_idx[:-1]
    last_endrow = sep_idx.sum().item() # in case ori_seqlen != last_endrow
    indices = torch.arange(last_endrow).split(tuple(sep_idx.tolist())) # tuple of cells, grouped by same start-row

    header_end_idx = None
    if 'header_labels' in target:
        header = target['header_labels'].clone()
        header_end = header.nonzero()
        if header_end.numel() == 0:
            header_end = -1
            header_end_idx = -1
        else: 
            header_end = header_end.max().item()
            for i, ind in enumerate(indices):
                if ind[-1] == header_end:
                    header_end_idx = i; break
            if header_end_idx == None:
                print('error')

    if last_endrow < ori_seqlen:
        last_idx = torch.arange(last_endrow, ori_seqlen)
    else:
        last_idx = None

    # Prepare coord-values
    boxes = target['boxes'].clone() # [0, 1] x1,y1,x2,y2
    num_boxes = boxes.shape[0]

    quant_boxes = boxes * (X_BIN - 1) # [0, X_BIN)

    ## src - quantized class
    _quant_pts_111 = torch.floor(quant_boxes[:, 0:2]).clamp_(min=0)
    _quant_pts_122 = torch.ceil(quant_boxes[:, 0:2]).clamp_(max=X_BIN-1)
    _quant_pts_112 = torch.stack((_quant_pts_111[:, 0], _quant_pts_122[:, 1]), dim=1)
    _quant_pts_121 = torch.stack((_quant_pts_122[:, 0], _quant_pts_111[:, 1]), dim=1)
    _quant_pts_211 = torch.floor(quant_boxes[:, 2:4]).clamp_(min=0)
    _quant_pts_222 = torch.ceil(quant_boxes[:, 2:4]).clamp_(max=X_BIN-1)
    _quant_pts_212 = torch.stack((_quant_pts_211[:, 0], _quant_pts_222[:, 1]), dim=1)
    _quant_pts_221 = torch.stack((_quant_pts_222[:, 0], _quant_pts_211[:, 1]), dim=1)

    _delta_x11 = quant_boxes[:, 0] - _quant_pts_111[:, 0]
    delta_x11 = torch.as_tensor([0.]) # [0] for <bos>
    delta_x11 = expand_delta(_delta_x11, delta_x11, indices, last_idx)
    delta_x12 = torch.ones_like(delta_x11) - delta_x11

    _delta_y11 = quant_boxes[:, 1] - _quant_pts_111[:, 1]
    delta_y11 = torch.as_tensor([0.]) # [0] for <bos>
    delta_y11 = expand_delta(_delta_y11, delta_y11, indices, last_idx)
    delta_y12 = torch.ones_like(delta_y11) - delta_y11

    _delta_x21 = quant_boxes[:, 2] - _quant_pts_211[:, 0]
    delta_x21 = torch.as_tensor([0.]) # [0] for <bos>
    delta_x21 = expand_delta(_delta_x21, delta_x21, indices, last_idx)
    delta_x22 = torch.ones_like(delta_x21) - delta_x21

    _delta_y21 = quant_boxes[:, 3] - _quant_pts_211[:, 1]
    delta_y21 = torch.as_tensor([0.]) # [0] for <bos>
    delta_y21 = expand_delta(_delta_y21, delta_y21, indices, last_idx)
    delta_y22 = torch.ones_like(delta_y21) - delta_y21

    ### add offset
    quant_pts_111 = expand_quant_point(_quant_pts_111.to(torch.long), torch.as_tensor([[BOS, BOS]]), 
                                       indices, last_idx, header_end_idx, COORD_SHIFT, SEP, EOH, EOS)
    quant_pts_112 = expand_quant_point(_quant_pts_112.to(torch.long), torch.as_tensor([[BOS, BOS]]), 
                                       indices, last_idx, header_end_idx, COORD_SHIFT, SEP, EOH, EOS)
    quant_pts_121 = expand_quant_point(_quant_pts_121.to(torch.long), torch.as_tensor([[BOS, BOS]]), 
                                       indices, last_idx, header_end_idx, COORD_SHIFT, SEP, EOH, EOS)
    quant_pts_122 = expand_quant_point(_quant_pts_122.to(torch.long), torch.as_tensor([[BOS, BOS]]), 
                                       indices, last_idx, header_end_idx, COORD_SHIFT, SEP, EOH, EOS)
    quant_pts_211 = expand_quant_point(_quant_pts_211.to(torch.long), torch.as_tensor([[BOS, BOS]]), 
                                       indices, last_idx, header_end_idx, COORD_SHIFT, SEP, EOH, EOS)
    quant_pts_212 = expand_quant_point(_quant_pts_212.to(torch.long), torch.as_tensor([[BOS, BOS]]), 
                                       indices, last_idx, header_end_idx, COORD_SHIFT, SEP, EOH, EOS)
    quant_pts_221 = expand_quant_point(_quant_pts_221.to(torch.long), torch.as_tensor([[BOS, BOS]]), 
                                       indices, last_idx, header_end_idx, COORD_SHIFT, SEP, EOH, EOS)
    quant_pts_222 = expand_quant_point(_quant_pts_222.to(torch.long), torch.as_tensor([[BOS, BOS]]), 
                                       indices, last_idx, header_end_idx, COORD_SHIFT, SEP, EOH, EOS)

    ## tgt - normalized box
    tgt_boxes = torch.empty((0, 4), dtype=boxes.dtype)
    boxes_xywh = boxes.clone()
    boxes_xywh[..., 2:] -= boxes_xywh[..., :2]
    for i, idx in enumerate(indices):
        tgt_boxes = torch.cat((tgt_boxes, boxes_xywh[idx]), dim=0)
        tgt_boxes = torch.cat((tgt_boxes, torch.as_tensor([[0,0,1,1]])), dim=0) # dummy tgt box(prevent NaN in cIoU loss) for <sep> and <eos> (only at the end)
    if last_idx is not None:
        tgt_boxes = torch.cat((tgt_boxes, boxes_xywh[last_idx]), dim=0)
        tgt_boxes = torch.cat((tgt_boxes, torch.as_tensor([[0,0,1,1]])), dim=0) # dummy tgt box for <sep> and <eos> (only at the end)

    # Prepare class
    _labels = target['labels'].clone()

    labels = torch.as_tensor([BOS]) # <bos>
    for i, idx in enumerate(indices):
        labels = torch.cat((labels, _labels[idx]), dim=0)
        if header_end_idx is not None and i == header_end_idx:
            labels = torch.cat((labels, torch.as_tensor([EOH])), dim=0) # <end-of-header>
        else:
            labels = torch.cat((labels, torch.as_tensor([SEP])), dim=0) # <sep>
    if last_idx is not None:
        labels = torch.cat((labels, _labels[last_idx]), dim=0)
    else:
        labels = labels[:-1]
    labels = torch.cat((labels, torch.as_tensor([EOS])), dim=0) # <eos>
    src_labels = labels #[:-1] to prevent to ignore eos
    tgt_labels = labels[1:]

    seq_len = len(src_labels)

    # Prepare empty-cell class
    _is_empty_seq = target['is_empty'].clone() # only target!!!
    is_empty_seq = torch.as_tensor([0.])
    for i, idx in enumerate(indices):
        is_empty_seq = torch.cat((is_empty_seq, _is_empty_seq[idx]), dim=0)
        is_empty_seq = torch.cat((is_empty_seq, torch.as_tensor([0.])), dim=0) # <end-of-header> or <sep>
    if last_idx is not None:
        is_empty_seq = torch.cat((is_empty_seq, _is_empty_seq[last_idx]), dim=0)
    else:
        is_empty_seq = is_empty_seq[:-1]
    is_empty_seq = torch.cat((is_empty_seq, torch.as_tensor([0.])), dim=0) # <eos>    
    tgt_is_empty_seq = is_empty_seq[1:]
       
    # Pad token - coord: PAD, class: PAD token
    if seq_len <= max_seq_len:
        num_pad = max_seq_len - seq_len
        src_labels = torch.cat((src_labels, torch.as_tensor([PAD] * num_pad).to(torch.int64)), dim=0)
        tgt_labels = torch.cat((tgt_labels, torch.as_tensor([PAD] * (num_pad + 1)).to(torch.int64)), dim=0)
        tgt_is_empty_seq = torch.cat((tgt_is_empty_seq, torch.as_tensor([0.] * (num_pad + 1))), dim=0)

        quant_pts_111 = torch.cat((quant_pts_111, torch.as_tensor([[PAD, PAD]] * num_pad).to(torch.int64)), dim=0)
        quant_pts_112 = torch.cat((quant_pts_112, torch.as_tensor([[PAD, PAD]] * num_pad).to(torch.int64)), dim=0)
        quant_pts_121 = torch.cat((quant_pts_121, torch.as_tensor([[PAD, PAD]] * num_pad).to(torch.int64)), dim=0)
        quant_pts_122 = torch.cat((quant_pts_122, torch.as_tensor([[PAD, PAD]] * num_pad).to(torch.int64)), dim=0)
        quant_pts_211 = torch.cat((quant_pts_211, torch.as_tensor([[PAD, PAD]] * num_pad).to(torch.int64)), dim=0)
        quant_pts_212 = torch.cat((quant_pts_212, torch.as_tensor([[PAD, PAD]] * num_pad).to(torch.int64)), dim=0)
        quant_pts_221 = torch.cat((quant_pts_221, torch.as_tensor([[PAD, PAD]] * num_pad).to(torch.int64)), dim=0)
        quant_pts_222 = torch.cat((quant_pts_222, torch.as_tensor([[PAD, PAD]] * num_pad).to(torch.int64)), dim=0)

        delta_x11 = torch.cat((delta_x11, torch.as_tensor([0.] * num_pad)), dim=0)
        delta_x12 = torch.cat((delta_x12, torch.as_tensor([0.] * num_pad)), dim=0)
        delta_y11 = torch.cat((delta_y11, torch.as_tensor([0.] * num_pad)), dim=0)
        delta_y12 = torch.cat((delta_y12, torch.as_tensor([0.] * num_pad)), dim=0)
        delta_x21 = torch.cat((delta_x21, torch.as_tensor([0.] * num_pad)), dim=0)
        delta_x22 = torch.cat((delta_x22, torch.as_tensor([0.] * num_pad)), dim=0)
        delta_y21 = torch.cat((delta_y21, torch.as_tensor([0.] * num_pad)), dim=0)
        delta_y22 = torch.cat((delta_y22, torch.as_tensor([0.] * num_pad)), dim=0)

        input_seq = {
                     'coord': torch.stack((quant_pts_111, quant_pts_112, quant_pts_121, quant_pts_122,
                                           quant_pts_211, quant_pts_212, quant_pts_221, quant_pts_222), 
                                           dim=1),
                     'delta': torch.stack((delta_x11, delta_x12, delta_y11, delta_y12,
                                           delta_x21, delta_x22, delta_y21, delta_y22), 
                                           dim=1),
                     'token': src_labels,
        }

        tgt_boxes = torch.cat((tgt_boxes, torch.as_tensor([[0,0,1,1]] * (num_pad + 1))), dim=0) # dummy tgt box pad

        target_seq = {
                     'coord': tgt_boxes,
                     'token': tgt_labels,
                     'empty': tgt_is_empty_seq,
        }

        # token weight (to calculate loss in training)
        token_weights = torch.where(tgt_labels == PAD,
                                    torch.zeros_like(tgt_labels, dtype=torch.float32),
                                    torch.ones_like(tgt_labels, dtype=torch.float32))
        token_weights = torch.where(tgt_labels == EOS,
                                        torch.zeros_like(token_weights, dtype=torch.float32) + loss_eos_token_weight,
                                        token_weights)
    else:
        src_labels = src_labels[:max_seq_len]
        input_seq = {
                     'coord': torch.stack((quant_pts_111[:max_seq_len], quant_pts_112[:max_seq_len], 
                                           quant_pts_121[:max_seq_len], quant_pts_122[:max_seq_len],
                                           quant_pts_211[:max_seq_len], quant_pts_212[:max_seq_len], 
                                           quant_pts_221[:max_seq_len], quant_pts_222[:max_seq_len]), 
                                           dim=1),
                     'delta': torch.stack((delta_x11[:max_seq_len], delta_x12[:max_seq_len], 
                                           delta_y11[:max_seq_len], delta_y12[:max_seq_len],
                                           delta_x21[:max_seq_len], delta_x22[:max_seq_len], 
                                           delta_y21[:max_seq_len], delta_y22[:max_seq_len]), 
                                           dim=1),
                     'token': src_labels,
        }
        tgt_boxes = tgt_boxes[:max_seq_len]
        tgt_labels = tgt_labels[:max_seq_len]
        target_seq = {
                     'coord': tgt_boxes,
                     'token': tgt_labels,
                     'empty': tgt_is_empty_seq[:max_seq_len],
        }

        # token weight (to calculate loss in training)
        token_weights = torch.where(tgt_labels == PAD,
                                    torch.zeros_like(tgt_labels, dtype=torch.float32),
                                    torch.ones_like(tgt_labels, dtype=torch.float32))
        token_weights = torch.where(tgt_labels == EOS,
                                        torch.zeros_like(token_weights, dtype=torch.float32) + loss_eos_token_weight,
                                        token_weights)
        
    # mask
    input_mask = torch.where(src_labels == PAD,
                             torch.zeros_like(src_labels, dtype=torch.bool),
                             torch.ones_like(src_labels, dtype=torch.bool))
    input_mask = torch.where(src_labels == EOS,
                             torch.zeros_like(src_labels, dtype=torch.bool),
                             input_mask)
    # including last token (target EOS class) - for decoder attention
    
    target_mask = torch.where(tgt_labels == PAD,
                                    torch.zeros_like(tgt_labels, dtype=torch.bool),
                                    torch.ones_like(tgt_labels, dtype=torch.bool))
    # including (real) EOS token - because when calculating vertex loss, it doesn't care prediction in 'EOS' class
    target_box_mask = torch.where(tgt_labels == PAD,
                                    torch.zeros_like(tgt_labels, dtype=torch.bool),
                                    torch.ones_like(tgt_labels, dtype=torch.bool))
    target_box_mask = torch.where(tgt_labels == EOS,
                                    torch.zeros_like(target_box_mask, dtype=torch.bool),
                                    target_box_mask)
    target_box_mask = torch.where(tgt_labels == SEP,
                                    torch.zeros_like(target_box_mask, dtype=torch.bool),
                                    target_box_mask)
    target_box_mask = torch.where(tgt_labels == EOH,
                                    torch.zeros_like(target_box_mask, dtype=torch.bool),
                                    target_box_mask)

    target['input_seq'] = {'seq': input_seq,
                           'mask': input_mask}
    target['target_seq'] = {'seq': target_seq,
                            'mask': target_mask,
                            'mask_box': target_box_mask,}
    
    # Add token weights (only need in training)
    target['token_weights'] = token_weights

    # modify structure indices
    if 'structure' in target:
        structure = target['structure'].clone()
        temp = [len(x) for x in indices]
        if last_idx is not None:
            temp += [len(last_idx)]
        structure = structure.split(temp, dim=0)
        unit = torch.zeros((1, 4), dtype=structure[0].dtype) - 1 # dummy
        dummy = [unit] * len(structure)
        structure = [torch.cat((x, y), dim=0) for x, y in zip(structure, dummy)]
        structure = torch.cat(structure, dim=0)
        target['structure'] = structure[:max_seq_len]
    
    # modify OCR labels
    if 'content' in target:
        content = target['content'].clone()
        temp = [len(x) for x in indices]
        if last_idx is not None:
            temp += [len(last_idx)]
        content = content.split(temp, dim=0)
        # ocr token for separator (invalid..) is [sep][eos][pad][pad][pad]...
        unit = torch.zeros((1, content[0].shape[1]), dtype=content[0].dtype) + 3 # dummy 3: [pad]
        unit[0,0] = 4 # 4: [sep]
        unit[0,1] = 1 # 1: [eos]
        dummy = [unit] * len(content)
        content = [torch.cat((x, y), dim=0) for x, y in zip(content, dummy)]
        content = torch.cat(content, dim=0)
        content = content[:max_seq_len]
        target['content'] = content

        # for AR-OCR models (OCR embeddings are also input to decoder input)
        unit = torch.zeros((1, content.shape[1]), dtype=content.dtype) + 3 # dummy 3: [pad]
        unit[0,0] = 0 # 0: [bos]
        unit[0,1] = 1 # 1: [eos]
        source_content = torch.cat((unit, content[:-1]), dim=0)
        if source_content.shape[0] <= max_seq_len:
            pad_length = max_seq_len - source_content.shape[0] - 1
            pad_content = torch.zeros((pad_length, content.shape[1]), dtype=content.dtype) + 3
            source_content = torch.cat((source_content, content[-1:], pad_content), dim=0) # max_seq_len, channel
        else:
            source_content = source_content[:max_seq_len]
        target['input_seq']['seq']['content'] = source_content

    return target
