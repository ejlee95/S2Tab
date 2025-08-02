
# -----------------------------------------------------------------------
# S2Tab official code : datasets/sequence_box_regression_quant.py
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

class Expand(object):
    def __init__(self, 
                 enable_multi_tables=False,
                 vocab=None):
        super().__init__()
        self.enable_multi_tables = enable_multi_tables
        self.sep = vocab['SEP']
        self.eoh = vocab['EOH']
        self.eos = vocab['EOS']
        self.eot = vocab['EOT']
        self.coord_shift = vocab['COORD_SHIFT']

    def expand_delta(self, _delta, delta, indices, last_idx=None, table_idx=None):
        table_num = 0
        for i, idx in enumerate(indices):
            delta = torch.cat((delta, _delta[idx]), dim=0)
            delta = torch.cat((delta, torch.as_tensor([0])), dim=0) # [0] for <sep>
            if self.enable_multi_tables and len(table_idx):
                if table_num < len(table_idx) and idx[-1] == table_idx[table_num]:
                    delta = torch.cat((delta, torch.as_tensor([0])), dim=0) # [0] for <eot>
                    table_num += 1
        if last_idx is not None:
            delta = torch.cat((delta, _delta[last_idx]), dim=0)
        else:
            delta = delta[:-1] # no <sep> at the end
        if self.enable_multi_tables and len(table_idx) == 0:
            delta = torch.cat((delta, torch.as_tensor([0])), dim=0) # [0] for <eot>
        delta = torch.cat((delta, torch.as_tensor([0])), dim=0) # [0] for <eos>

        return delta

    def expand_quant_point(self,
                           _quant_point, 
                           quant_point, 
                           indices, 
                           last_idx=None, 
                           header_end_idx=None, 
                           table_idx=None):
        table_num = 0
        for i, idx in enumerate(indices):
            quant_point = torch.cat((quant_point, _quant_point[idx] + self.coord_shift), dim=0)
            if header_end_idx is not None and i == header_end_idx:
                quant_point = torch.cat((quant_point, torch.as_tensor([[self.eoh, self.eoh]])), dim=0) # <end-of-header>
            else:
                quant_point = torch.cat((quant_point, torch.as_tensor([[self.sep, self.sep]])), dim=0) # <sep>
            if self.enable_multi_tables and len(table_idx):
                if table_num < len(table_idx) and idx[-1] == table_idx[table_num]:
                    quant_point = torch.cat((quant_point, torch.as_tensor([[self.eot, self.eot]])), dim=0) # <eot>
                    table_num += 1
        if last_idx is not None:
            quant_point = torch.cat((quant_point, _quant_point[last_idx] + self.coord_shift), dim=0)
        else:
            quant_point = quant_point[:-1] # no <sep> at the end
        
        if self.enable_multi_tables and len(table_idx) == 0:
            quant_point = torch.cat((quant_point, torch.as_tensor([[self.eot, self.eot]])), dim=0) # <eot>
        quant_point = torch.cat((quant_point, torch.as_tensor([[self.eos, self.eos]])), dim=0) # <eos>

        return quant_point

def reconstruction_seq_from_boxes(target, max_seq_len, 
                                    loss_eos_token_weight=0.1, 
                                    vocab=None,
                                    enable_multi_tables=False):
    """
    Make source/target sequences from annotations
    """
    X_BIN, Y_BIN = vocab['X_BIN'], vocab['Y_BIN']
    BOS, EOS, FAKE, SEP, PAD, EOH, EOT = vocab['BOS'], vocab['EOS'], vocab['FAKE'], vocab['SEP'], vocab['PAD'], vocab['EOH'], vocab['EOT']
    CLASS_OFFSET, MAX_CLASS, COORD_SHIFT = vocab['CLASS_OFFSET'], vocab['MAX_CLASS'], vocab['COORD_SHIFT']

    expand_class = Expand(enable_multi_tables, vocab)
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
    
    # if multi-tables case
    tab_idx = torch.as_tensor([])
    if enable_multi_tables and 'table_ids' in target:
        table_ids = target['table_ids']
        tab_idx = (table_ids[1:] - table_ids[:-1]).nonzero(as_tuple=True)[0] # right next to tab_idx = new table

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
    delta_x11 = expand_class.expand_delta(_delta_x11, delta_x11, indices, last_idx, tab_idx)
    delta_x12 = torch.ones_like(delta_x11) - delta_x11

    _delta_y11 = quant_boxes[:, 1] - _quant_pts_111[:, 1]
    delta_y11 = torch.as_tensor([0.]) # [0] for <bos>
    delta_y11 = expand_class.expand_delta(_delta_y11, delta_y11, indices, last_idx, tab_idx)
    delta_y12 = torch.ones_like(delta_y11) - delta_y11

    _delta_x21 = quant_boxes[:, 2] - _quant_pts_211[:, 0]
    delta_x21 = torch.as_tensor([0.]) # [0] for <bos>
    delta_x21 = expand_class.expand_delta(_delta_x21, delta_x21, indices, last_idx, tab_idx)
    delta_x22 = torch.ones_like(delta_x21) - delta_x21

    _delta_y21 = quant_boxes[:, 3] - _quant_pts_211[:, 1]
    delta_y21 = torch.as_tensor([0.]) # [0] for <bos>
    delta_y21 = expand_class.expand_delta(_delta_y21, delta_y21, indices, last_idx, tab_idx)
    delta_y22 = torch.ones_like(delta_y21) - delta_y21

    ### add offset
    quant_pts_111 = expand_class.expand_quant_point(_quant_pts_111.to(torch.long), torch.as_tensor([[BOS, BOS]]), 
                                       indices, last_idx, header_end_idx, tab_idx)
    quant_pts_112 = expand_class.expand_quant_point(_quant_pts_112.to(torch.long), torch.as_tensor([[BOS, BOS]]), 
                                       indices, last_idx, header_end_idx, tab_idx)
    quant_pts_121 = expand_class.expand_quant_point(_quant_pts_121.to(torch.long), torch.as_tensor([[BOS, BOS]]), 
                                       indices, last_idx, header_end_idx, tab_idx)
    quant_pts_122 = expand_class.expand_quant_point(_quant_pts_122.to(torch.long), torch.as_tensor([[BOS, BOS]]), 
                                       indices, last_idx, header_end_idx, tab_idx)
    quant_pts_211 = expand_class.expand_quant_point(_quant_pts_211.to(torch.long), torch.as_tensor([[BOS, BOS]]), 
                                       indices, last_idx, header_end_idx, tab_idx)
    quant_pts_212 = expand_class.expand_quant_point(_quant_pts_212.to(torch.long), torch.as_tensor([[BOS, BOS]]), 
                                       indices, last_idx, header_end_idx, tab_idx)
    quant_pts_221 = expand_class.expand_quant_point(_quant_pts_221.to(torch.long), torch.as_tensor([[BOS, BOS]]), 
                                       indices, last_idx, header_end_idx, tab_idx)
    quant_pts_222 = expand_class.expand_quant_point(_quant_pts_222.to(torch.long), torch.as_tensor([[BOS, BOS]]), 
                                       indices, last_idx, header_end_idx, tab_idx)

    ## tgt - normalized box
    tgt_boxes = torch.empty((0, 4), dtype=boxes.dtype)

    table_num = 0
    boxes_xywh = boxes.clone()
    boxes_xywh[..., 2:] -= boxes_xywh[..., :2]
    for i, idx in enumerate(indices):
        tgt_boxes = torch.cat((tgt_boxes, boxes_xywh[idx]), dim=0)
        tgt_boxes = torch.cat((tgt_boxes, torch.as_tensor([[0,0,1,1]])), dim=0) # dummy tgt box(prevent NaN in cIoU loss) for <sep> and <eos> (only at the end)
        if enable_multi_tables and len(tab_idx) > 0:
            if table_num < len(tab_idx) and idx[-1] == tab_idx[table_num]:
                tgt_boxes = torch.cat((tgt_boxes, torch.as_tensor([[0,0,1,1]])), dim=0) # dummy tgt box for <eot>
                table_num += 1
    if last_idx is not None:
        tgt_boxes = torch.cat((tgt_boxes, boxes_xywh[last_idx]), dim=0)
        tgt_boxes = torch.cat((tgt_boxes, torch.as_tensor([[0,0,1,1]])), dim=0) # dummy tgt box for <sep> and <eos> (only at the end)
    if enable_multi_tables and len(tab_idx) == 0:
        tgt_boxes = torch.cat((tgt_boxes, torch.as_tensor([[0,0,1,1]])), dim=0) # dummy tgt box for <eot>

    # Prepare class
    _labels = target['labels'].clone()

    labels = torch.as_tensor([BOS]) # <bos>
    table_num = 0
    for i, idx in enumerate(indices):
        labels = torch.cat((labels, _labels[idx]), dim=0)
        if header_end_idx is not None and i == header_end_idx:
            labels = torch.cat((labels, torch.as_tensor([EOH])), dim=0) # <end-of-header>
        else:
            labels = torch.cat((labels, torch.as_tensor([SEP])), dim=0) # <sep>
        if enable_multi_tables and len(tab_idx) > 0:
            if table_num < len(tab_idx) and idx[-1] == tab_idx[table_num]:
                labels = torch.cat((labels, torch.as_tensor([EOT])), dim=0) # <eot>
                table_num += 1
    if last_idx is not None:
        labels = torch.cat((labels, _labels[last_idx]), dim=0)
    else:
        labels = labels[:-1]
    if enable_multi_tables and len(tab_idx) == 0:
        labels = torch.cat((labels, torch.as_tensor([EOT])), dim=0) # <eot>
    labels = torch.cat((labels, torch.as_tensor([EOS])), dim=0) # <eos>
    src_labels = labels #[:-1] to prevent to ignore eos
    tgt_labels = labels[1:]

    seq_len = len(src_labels)
       
    # Pad token - coord: PAD, class: PAD token
    if seq_len <= max_seq_len:
        num_pad = max_seq_len - seq_len
        src_labels = torch.cat((src_labels, torch.as_tensor([PAD] * num_pad).to(torch.int64)), dim=0)
        tgt_labels = torch.cat((tgt_labels, torch.as_tensor([PAD] * (num_pad + 1)).to(torch.int64)), dim=0)

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
        }

        # token weight (to calculate loss in training)
        token_weights = torch.where(tgt_labels == PAD,
                                    torch.zeros_like(tgt_labels, dtype=torch.float32),
                                    torch.ones_like(tgt_labels, dtype=torch.float32))
        token_weights = torch.where(tgt_labels == EOS,
                                        torch.zeros_like(token_weights, dtype=torch.float32) + loss_eos_token_weight,
                                        token_weights)
        if tgt_boxes.shape[0] != tgt_labels.shape[0]:
            print('?')

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
    if 'structure' in target and len(target['structure']) > 1:
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

    return target
