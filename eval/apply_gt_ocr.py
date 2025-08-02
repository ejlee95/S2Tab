# -----------------------------------------------------------------------
# BITS official code : eval/apply_gt_ocr.py
# -----------------------------------------------------------------------

import numpy as np


def ious(cell_box, chunk_box):
    """
    cell_box: x1,y1,x2,y2 (N, 4)
    chunk_box: x1,y1,x2,y2 (M, 4)
    output: (N,M)
    """
    lt = np.maximum(cell_box[:,None,:2], chunk_box[None,:,:2]) #T.max(boxes1[:, None, :2], boxes2[:, :2]) # [N,M,2]
    rb = np.minimum(cell_box[:,None,2:], chunk_box[None,:,2:]) #T.min(boxes1[:, None, 2:], boxes2[:, 2:]) # [N,M,2]

    wh = (rb-lt).clip(0) # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    cell_area = (cell_box[:,None,2] - cell_box[:,None,0]) * (cell_box[:,None,3] - cell_box[:,None,1])
    chunk_area = (chunk_box[None,:,2] - chunk_box[None,:,0]) * (chunk_box[None,:,3] - chunk_box[None,:,1])

    union = cell_area + chunk_area - inter

    iou = inter / union

    return iou


def content_match(pred_cbox, gt_struct, gt_mode='xyxy', mode='pubtabnet', empty=None):
    # box format: x,y,x,y
    if mode == 'pubtables':
        gt_tbox = [x['bbox'] for x in gt_struct if 'bbox' in x] # text box
        gt_tind = [i for i, x in enumerate(gt_struct) if 'bbox' in x]

        # pred_cbox = np.asarray(pred_cbox)
        gt_tbox = np.asarray(gt_tbox)
    elif mode == 'scitsr':
        gt_tbox = [x['pos'] for x in gt_struct['cells'] if 'pos' in x] # text box
        gt_tind = [i for i, x in enumerate(gt_struct['cells']) if 'pos' in x]

        # pred_cbox = np.asarray(pred_cbox)
        gt_tbox = np.asarray(gt_tbox)
        if gt_mode == 'xxyy':
            gt_tbox = gt_tbox[:, [0,2,1,3]]
    else: # 'pubtables'
        gt_tbox = [x['bbox'] for x in gt_struct['cells'] if 'bbox' in x] # text box
        gt_tind = [i for i, x in enumerate(gt_struct['cells']) if 'bbox' in x]

        # pred_cbox = np.asarray(pred_cbox)
        gt_tbox = np.asarray(gt_tbox)
        if gt_mode == 'xywh':
            gt_tbox[:, 2:] += gt_tbox[:, :2]

    iou = ious(pred_cbox, gt_tbox) # iou[i,j] = iou(pred_cbox[i], gt_tbox[j])
    gt_ids = np.argmax(iou, axis=1)
    if empty is None:
        empty = [0 for _ in range(len(pred_cbox))]
    
    tokens = []
    for i, j in zip(np.arange(len(pred_cbox)), gt_ids):
        if iou[i,j] == 0 or empty[i] == 1:
            content = ''
        else:
            if mode == 'scitsr':
                content = ' '.join(gt_struct['cells'][gt_tind[j]]['content'])
            elif mode == 'pubtables':
                content = gt_struct[gt_tind[j]]['text']
            else:
                content = gt_struct['cells'][gt_tind[j]]['tokens']
                
        tokens.append(content)
    
    return tokens
