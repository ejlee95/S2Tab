# -----------------------------------------------------------------------
# S2Tab official code : datasets/coco.py
# -----------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -----------------------------------------------------------------------
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
from PIL import Image
import orjson
import copy
from typing import Any, List

import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms.functional as F

import datasets.transforms as T
from datasets import coco_base

class CocoDetection(coco_base.CocoDetection):
    def __init__(self, 
                 img_folder, 
                 ann_file, 
                 transforms, 
                 max_seq_len=100, 
                 version='1.0',
                 vocab=None,
                 include_heatmap=False,
                 include_empty_cells=True,
                ):
        self.no_eval = False
        if not (isinstance(ann_file, list) or ann_file.exists()):
            self.make_dummy_ann_file(img_folder, ann_file)
            self.no_eval = True
        
        super(CocoDetection, self).__init__(img_folder, ann_file)
        # override (to contain row_annotations, col_annotations)
        self.coco = COCO(ann_file)
        if not self.no_eval:
            self.no_eval = self.coco.no_eval
        self._transforms = transforms
        ENDROW_OFFSET=1000 if 'MAX_COL' not in vocab else 1000 if vocab['MAX_COL'] == 30 else 2500
        self.prepare = ConvertCocoPolysToMask(max_seq_len,
                                              vocab['CLASS_OFFSET'],
                                              ENDROW_OFFSET,
                                              include_heatmap,
                                              include_empty_cells,
                                              )
        self.seq_version = version
    
    def _load_row_target(self, id: int) -> List[Any]:
        return self.coco.loadRowAnns(id)

    def _load_col_target(self, id: int) -> List[Any]:
        return self.coco.loadColAnns(id)

    def __getitem__(self, idx):
        _id = self.ids[idx]
        img, _cls = self._load_image(_id)
        target = self._load_target(_id)
        if self.coco.rowcol:
            row_target = self._load_row_target(_id)[0]
            col_target = self._load_col_target(_id)[0]
        else:
            row_target, col_target = [], []
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target, 'row_annotations': row_target, 'col_annotations': col_target, 'cls': _cls}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        # self.check_annotation(img, target['boxes'], target['labels'], target['endrow'], image_id, 7) #CLASS_OFFSET)

        return img, target

    def make_dummy_ann_file(self, img_folder, ann_file):
        """
        Make dummy ann_file(path) to run the code
        """
        dic = {}
        dic["type"] = "instances"
        dic["categories"] = [{'id': 1, 'name': 'cell', 'supercategory': 'cell'}]
        dic['info'] = {'description': 'table Dataset - dummy', 'url': 'http://mscoco.org', \
                        'version': 1.0, 'year': 2021, 'contributor': 'IBM', 'data_created': '2021/10/24'}
        dic['licenses'] = {'id': 0, 'name': 'Attribution-NonCommercial-ShareAlike License', \
                        'url': 'http://images.cocodataset.org/val2014/COCO_val2014_000000391895.jpg'}
        images = []
        annotations = []

        obj_id = 1
        img_post = 'png'
        if len(list(img_folder.glob(f'*.{img_post}'))) == 0:
            img_post = 'jpg'

        for img_id, img_path in enumerate(img_folder.glob(f'*.{img_post}')):
            file_name = img_path.name
            img_load = Image.open(img_path)
            img = {"file_name": file_name, \
                    "height": img_load.height, "id": img_id, "width": img_load.width, \
                    "license": '0', 'url': 'http://images.cocodataset.org/val2014/COCO_val2014_000000391895.jpg',\
                    'data_captured': '2021/10/24'}
            images.append(img)
            for i in range(2):
                anno = {"area": 0., "bbox": [0,0,1,1], "category_id": 1, "id": obj_id, "text_bbox": [0,0,1,1], \
                        "image_id": img_id, "iscrowd": 0, "start_row": 0, "start_col": 0, "end_row": 0, "end_col": 0, "tokens": []}
                
                annotations.append(anno)

                obj_id += 1
        
        dic['images'] = images
        dic['annotations'] = annotations

        ann_file.parent.mkdir(parents=True, exist_ok=True)
        dic = orjson.dumps(dic)
        with open(ann_file, 'wb') as f:
            f.write(dic)
            # json.dump(dic, f, indent=2)


    def delete_dummy_file(self, path):
        path.unlink(missing_ok=True)
        print(f'dummy file {path} deleted')

    def check_annotation(self, img, bbox, classes, endrow, image_id, CLASS_OFFSET):
        color_map = {f'{x},{y},{z}': (int(255 / 30 * x), int(255 / 30 * y), int(255 / 2 * z),) for x in range(1,31) for y in range(1,31) for z in range(0,2)}
        # color_map = {1+CLASS_OFFSET: 'blue', 2+CLASS_OFFSET: 'green', 3+CLASS_OFFSET: 'red', 4+CLASS_OFFSET: 'black',}
        # #{1: 'blue', 2: 'green', 3: 'red', 4: 'black', 5: 'orange', 6: 'cyan', 7: 'purple', 8: 'pink'}
        # label_map = {1+CLASS_OFFSET: 'simple', 2+CLASS_OFFSET: 'simpleP', 
        #             3+CLASS_OFFSET:'span', 4+CLASS_OFFSET: 'spanP', 
        #             # 5: 'simpleE', 6: 'simplePE', 
        #             #     7: 'spanE', 8: 'spanPE'
        #             }
        if not isinstance(img, torch.Tensor):
            img = F.to_tensor(img)
        else:
            mean, std = [0.909, 0.919, 0.915], [0.203, 0.152 , 0.187]
            mean, std = torch.as_tensor(mean).to(img.device), torch.as_tensor(std).to(img.device)
            img = (img * std[:, None, None]) + mean[:, None, None]
        
        bbox[:, 0::2] *= img.shape[2]
        bbox[:, 1::2] *= img.shape[1]
        init_color = (255, 0, 0)
        fin_color = (0, 0, 255)
        if len(bbox) > 0:
            step = 255 / len(bbox)
        else:
            step = 0
        order_color = [(int(init_color[0] - step*i), init_color[1], int(init_color[2] + step*i)) for i in range(len(bbox))]
        row_class = (classes - 1 - CLASS_OFFSET) % 30
        col_class = (classes - 1 - CLASS_OFFSET) // 30
        
        draw_img = torchvision.utils.draw_bounding_boxes((img * 255).to(torch.uint8), bbox, 
                                                            # [label_map[(x.item()-1-CLASS_OFFSET)%4+CLASS_OFFSET+1] for x in classes],
                                                            # [f'{x},{y},{int(z)}' for x, y, z in zip(col_class, row_class, endrow)],
                                                            fill=[True] * len(row_class),
                                                            colors=[color_map[f'{x},{y},{int(z)}'] for x, y, z in zip(col_class, row_class, endrow)], width=2)
                                                            # colors=['green' if z else 'black' for z in endrow], width=2)
                                                            # colors=order_color, width=2)
                                                            #[color_map[x.item()] for x in classes], width=2)
        draw_img = F.to_pil_image(draw_img)
        name = self.coco.loadImgs(image_id)[0]['file_name']
        draw_img.save(f'./check/{image_id}_{name}')

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class COCO(coco_base.COCO):
    """
    Almost copy-paste from pycocotools/coco.py/COCO
    Add row_annotations, col_annotations to the attrib
    """
    def __init__(self, annotation_file=None):
        super(COCO, self).__init__(annotation_file)
        if 'row_annotations' in self.dataset:
            self.createIndex_rowcol()
            self.rowcol = True
        else:
            self.rowcol = False

    def loadRowAnns(self, ids=[]):
        """
        Load row anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.rowanns[id] for id in ids]
        elif type(ids) == int:
            return [self.rowanns[ids]]

    def loadColAnns(self, ids=[]):
        """
        Load col anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.colanns[id] for id in ids]
        elif type(ids) == int:
            return [self.colanns[ids]]
    
    def createIndex_rowcol(self):
        # create index for 'row_annotations' and 'col_annotations'
        print('creating index for row/col annotations...')
        row_anns, col_anns = {}, {}

        if 'row_annotations' in self.dataset:
            for ann in self.dataset['row_annotations']:
                row_anns[ann['image_id']] = ann['rows'] # [[start-y,end-y]]

        if 'col_annotations' in self.dataset:
            for ann in self.dataset['col_annotations']:
                col_anns[ann['image_id']] = ann['cols'] # [[start-x,end-x]]

        self.rowanns = row_anns
        self.colanns = col_anns

class ConvertCocoPolysToMask(object):
    def __init__(self, max_seq_len=100, 
                       CLASS_OFFSET=5,
                       ENDROW_OFFSET=1000,
                       include_heatmap=False,
                       include_empty_cells=True,):
        self.max_seq_len = max_seq_len
        self.CLASS_OFFSET = CLASS_OFFSET
        self.ENDROW_OFFSET = ENDROW_OFFSET
        self.include_heatmap = include_heatmap
        self.include_empty_cells = include_empty_cells

    def __call__(self, image, _target):
        w, h = image.size

        image_id = _target["image_id"]
        image_id = torch.tensor([image_id])

        anno = _target["annotations"]
        row_target = _target["row_annotations"]
        col_target = _target["col_annotations"]

        if 'start_col' in anno[0]:
            structure = [torch.as_tensor([obj["start_col"], obj["end_col"]+1, \
                                            obj["start_row"], obj["end_row"]+1]) for obj in anno] # x1,x2,y1,y2
            structure = torch.stack(structure, dim=0) # (bs, 4)

        else:
            structure = torch.zeros((1,4), dtype=torch.long)

        boxes = [obj['bbox']* (len(obj['bbox']) ==4) + [0,0,1,1] * (len(obj['bbox']) < 4) for obj in anno] # x1,y1,w,h
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if 'text_bbox' in anno[0]:
            text_boxes = [(len(obj['text_bbox']) > 0) * obj['text_bbox'] + 
                            (len(obj['text_bbox']) == 0) * [0,0,0,0] for obj in anno]
            text_boxes = torch.stack([torch.as_tensor(obj) for obj in text_boxes], 
                                    dim=0)
            text_boxes[:, 2:] += text_boxes[:, :2]

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
       
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        if 'text_bbox' in anno[0]:
            text_boxes = text_boxes[keep]
        classes = classes[keep]
        endrow = torch.div((classes - 1), self.ENDROW_OFFSET, rounding_mode='floor')

        classes = ((classes - 1) % self.ENDROW_OFFSET) + 1 + self.CLASS_OFFSET
        # # torch.maximum((classes - 1 - CLASS_OFFSET) // 30, zeros_like(classes)) : colspan
        # # (classes - 1 - CLASS_OFFSET) % 30 : rowspan
        structure = structure[keep]

        # header class (header / body)
        if 'is_header' in anno[0]:
            header_classes = [obj['is_header'] for obj in anno]
            header_classes = torch.tensor(header_classes, dtype=torch.int64)
            header_classes = header_classes[keep]
        else:
            header_classes = None

        # table boxes
        tx1, tx2 = torch.min(boxes[:, 0]), torch.max(boxes[:, 2])
        ty1, ty2 = torch.min(boxes[:, 1]), torch.max(boxes[:, 3])

        # is empty
        if 'tokens' in anno[0]:
            content = np.asarray([''.join(obj['tokens']) for obj in anno])
            content = content[keep]
            is_empty = torch.as_tensor([(len(x) > 0) * False + (len(x) == 0) * True for x in content])
            is_empty = is_empty[keep]
        else:
            is_empty = torch.as_tensor([0 for _ in range(classes.shape[0])])

        if row_target != []:
            row_border_boxes = torch.stack([torch.as_tensor(obj) for obj in row_target], dim=0)
            col_boxes = torch.stack([torch.as_tensor(obj) for obj in col_target], dim=0)
        
            th = torch.stack([col_boxes[structure[:, 0]], row_border_boxes[structure[:, 2]], 
                        col_boxes[structure[:, 1]], row_border_boxes[structure[:, 3]]], dim=1) # th for x1,y1,x2,y2 (bs, 4, 2)
            th = th - boxes[..., None]
            
            row_num = len(row_target) # row border number
            col_num = len(col_target)
        
        if 'table_id' in anno[0]: # multi-tables
            table_ids = torch.as_tensor([obj['table_id'] for obj in anno])
            table_ids = table_ids[keep]
        else:
            table_ids = None
            
        # boxes = boxes[:self.max_seq_len-1, :]
        # if 'text_bbox' in anno[0]:
        #     text_boxes = text_boxes[:self.max_seq_len-1, :]
        # classes = classes[:self.max_seq_len-1]
        # endrow = endrow[:self.max_seq_len-1]
        # structure = structure[:self.max_seq_len-1, :]
        # is_empty = is_empty[:self.max_seq_len-1]
        # if row_target != []:
        #     th = th[:self.max_seq_len-1]
        # if header_classes is not None:
        #     header_classes = header_classes[:self.max_seq_len-1]
        # if table_ids is not None:
        #     table_ids = table_ids[:self.max_seq_len-1]
        
        if row_target != []:
            rowcol_num = max(row_num, col_num)

            structure_by_rc = torch.ones((rowcol_num, rowcol_num, 4), dtype=torch.long) * (-1)
            final_rowcol_num = rowcol_num

            for i in range(rowcol_num):
                structure_i = (structure == i).nonzero(as_tuple=True)
                num_i = len(structure_i[0])
                if num_i == 0:
                    final_rowcol_num = i
                    break

                structure_index_mask = torch.zeros((num_i, 4), dtype=torch.long) # temp
                structure_index_mask[torch.arange(0, num_i), structure_i[1]] = 1
                structure_index = structure_index_mask.cumsum(0) * structure_index_mask - 1

                structure_index_mask_nonzero = structure_index_mask.nonzero()

                num_i = structure_index.max().item() + 1
                assert num_i <= rowcol_num, f"Error: {i}-th row/col number {num_i} exceeds total number {rowcol_num} in sample id {_target['image_id']}."
                structure_by_rc_i = torch.ones((num_i, 4), dtype=torch.long) * (-1)
                structure_by_rc_i[structure_index[structure_index_mask_nonzero[:,0], 
                                                structure_index_mask_nonzero[:,1]], 
                                    structure_i[1]] = structure_i[0]
                structure_by_rc[i, :num_i] = structure_by_rc_i
            structure_by_rc = structure_by_rc[:final_rowcol_num]

        if 'cls' in _target:
            _cls = 1. if _target['cls'] == 'complex' else 0.
            _cls = torch.as_tensor([_cls])

        ## (Optional) auxiliary heatmap loss
        if self.include_heatmap:
            # following CRAFT https://github.com/clovaai/CRAFT-pytorch/issues/3
            heatmap = torch.zeros((h, w), dtype=torch.float32)
            sigma = 10
            spread = 3
            extent = int(spread * sigma)
            y, x = torch.meshgrid(torch.arange(2*extent), torch.arange(2*extent))
            gaussian_template = 1/2/torch.pi/(sigma ** 2) * torch.exp(-1/2 * ((x - spread * sigma - 0.5) ** 2 + 
                                                                              (y - spread * sigma - 0.5) ** 2) / 
                                                                             (sigma ** 2))
            gaussian_template /= gaussian_template.max()
            gaussian_template = gaussian_template.unsqueeze(0)

            src = [[0, 0], 
                   [2*extent-1, 0], 
                   [2*extent-1, 2*extent-1], 
                   [0, 2*extent-1]] # top-left, top-right, bottom-right, bottom-left
            for e, b in zip(is_empty, boxes.clone()):
                if not self.include_empty_cells and e: # empty
                    continue
                b = list(map(int, b))
                b_h, b_w = b[3] - b[1], b[2] - b[0]
                dst = [[0, 0],
                       [b[2]-b[0], 0],
                       [b[2]-b[0], b[3]-b[1]], 
                       [0, b[3]-b[1]]]
                
                if b_h > 2*extent or b_w > 2*extent:
                    src_image = torch.zeros((1, max(b_h, 2*extent), max(b_w, 2*extent)), dtype=gaussian_template.dtype)
                    src_image[:, :2*extent, :2*extent] = gaussian_template.clone()
                else:
                    src_image = gaussian_template.clone()
                try:
                    b_gaussian = F.perspective(src_image,
                                            startpoints=src,
                                            endpoints=dst)
                    b_gaussian = b_gaussian[0, :b[3]-b[1], :b[2]-b[0]]
                    heatmap[b[1]:b[3], b[0]:b[2]] += b_gaussian
                except: # for tiny boxes..
                    heatmap[b[1]+(b[3]-b[1])//2, b[0]+(b[2]-b[0])//2] = 1

        target = {}
        target["boxes"] = boxes
        if 'text_bbox' in anno[0]:
            target["text_boxes"] = text_boxes
        target["original_boxes"] = copy.deepcopy(boxes)
        target["labels"] = classes
        target["endrow"] = endrow
        target["image_id"] = image_id
        target["table_box"] = torch.as_tensor([[tx1,ty1,tx2,ty2]], dtype=torch.float32)
        target["structure"] = structure
        target["is_empty"] = torch.as_tensor(is_empty)
        if row_target != []:
            target["structure_rc"] = structure_by_rc # (rowcol_num, cell_max_num, 4) # cell_max_num may have dummy
            target["rownum"] = torch.as_tensor(row_num - 1, dtype=torch.float32)
            target["colnum"] = torch.as_tensor(col_num - 1, dtype=torch.float32)
            row_boxes = row_border_boxes.clone().to(torch.float32)
            row_boxes = torch.stack((row_boxes[:-1,1], row_boxes[1:, 0]), dim=-1)
            txs = torch.as_tensor([tx1, tx2] * len(row_boxes)).view(-1, 2)
            row_boxes = torch.stack((txs, row_boxes), dim=-1).view(-1, 4) # x,y,x,y
            target["row_boxes"] = row_boxes
            if header_classes is not None:
                target["header_labels"] = header_classes
                if len(header_classes.nonzero()) > 0:
                    header_idx = header_classes.nonzero()[-1][0].item()
                    header_row_idx = structure[header_idx, 2] # start-row
                    header_box = row_boxes[:header_row_idx+1]
                    header_box = torch.as_tensor([[header_box[:, 0].min(), header_box[:, 1].min(), header_box[:, 2].max(), header_box[:, 3].max()]]) # (1,4)
                    target["header_box"] = header_box
        if 'cls' in _target:
            target['cls'] = _cls
        if table_ids is not None:
            target['table_ids'] = table_ids

        if self.include_heatmap:
            target['heatmap'] = heatmap
            
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, max_seq_len=100, 
                         loss_eos_token_weight=0.1,
                         ignore_aspect_ratio=False, seq_version=None, 
                         vocab=None, 
                         image_max=480, image_min=480,
                         interpolation_mode='bilinear',
                         transforms_list=[],
                         enable_multi_tables=False,
                         normalize_dist=[[0.909, 0.919, 0.915], [0.203, 0.152 , 0.187]],
                         include_heatmap=False,
                         include_empty_cell=True,):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(normalize_dist[0], normalize_dist[1]) # PTN, STN, FTN # mean, std
    ])

    def round_to_even(num: float):
        return round(num) + round(num) % 2

    img_size = image_max
    if 'resize' in transforms_list:
        scales = [round_to_even(image_min * x / 10) for x in range(6, 11)]
    else:
        scales = [image_min]

    if image_set == 'train':
        aug_prob = 0.3
        transforms = []
        if 'crop' not in transforms_list: # only resize
            transforms.append(T.RandomResize(scales, max_size=img_size, ignore_aspect_ratio=ignore_aspect_ratio, interpolation_mode=interpolation_mode))
        else:
            if 'resize' in transforms_list:
                before_crop_scales = [round_to_even(image_min * x / 10) for x in range(8, 11)]
            else:
                before_crop_scales = [round_to_even(image_min * 0.8)]
            crop_size = (round_to_even(image_min * 0.6), round_to_even(image_min * 0.8))
            transforms.append(T.RandomSelect(T.RandomResize(scales, 
                                                            max_size=img_size, 
                                                            ignore_aspect_ratio=ignore_aspect_ratio, 
                                                            interpolation_mode=interpolation_mode),
                                                T.Compose([
                                                    T.RandomResize(before_crop_scales,
                                                                ignore_aspect_ratio=ignore_aspect_ratio, 
                                                                interpolation_mode=interpolation_mode),
                                                    T.RandomSelect(T.RandomCrop(crop_size),
                                                                   T.RandomCrop((crop_size[1], crop_size[0]))
                                                    ),
                                                    T.RandomResize(scales, 
                                                                max_size=image_max, 
                                                                ignore_aspect_ratio=ignore_aspect_ratio, 
                                                                interpolation_mode=interpolation_mode)
                                                ]
                                                ),
                                                p=1-aug_prob)
                            )
        if 'rotate' in transforms_list:
            transforms.append(T.RandomRotate(p=aug_prob))
        if 'hflip' in transforms_list:
            transforms.append(T.RandomHorizontalFlip(p=aug_prob))
        if 'transpose' in transforms_list:
            transforms.append(T.Transpose(p=aug_prob))

        transforms.append(T.MaxSeqCrop(max_seq_len))
        if include_heatmap:
            transforms.append(T.RefineHeatmap(include_empty_cells=include_empty_cell, box_key='boxes'))

        transforms += [normalize,
                       T.ConstructSeq(max_seq_len, loss_eos_token_weight, seq_version, vocab, enable_multi_tables)]
        return T.Compose(transforms)
    if image_set == 'val' or image_set == 'test':
        return T.Compose([
            T.RandomResize(scales[-1:], max_size=img_size, ignore_aspect_ratio=ignore_aspect_ratio, interpolation_mode=interpolation_mode),
            T.MaxSeqCrop(max_seq_len),
            normalize,
            T.ConstructSeq(max_seq_len, 0., seq_version, vocab, enable_multi_tables),
        ])
    
    raise ValueError(f'unknown {image_set}')

def build(image_set, args, vocab):
    if ',' in args.dataset.coco_path:
        root = args.dataset.coco_path.split(',')[0]
        root = Path(root)
    else:
        root = Path(args.dataset.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist!'
    if 'image_path' in args.dataset and args.dataset.image_path is not None:
        image_root = Path(args.dataset.image_path)
        assert image_root.exists(), f'provided Image path {image_root} does not exist!'
    else:
        image_root = root
    mode = 'instances'
    postfix = 'json'
    PATHS = {
        "train": (image_root / "train2017", root / "annotations" / f'{mode}_train2017.{postfix}'),
        "val": (image_root / "val2017", root / "annotations" / f'{mode}_val2017.{postfix}'),
        "test": (image_root / "test2017", root / "annotations" / f'{mode}_test2017.{postfix}'),
    }

    max_seq_len = args.train.max_seq_len if image_set == 'train' else args.test.max_seq_len

    interpolation_mode = args.train.interpolation if image_set == 'train' else args.test.interpolation
    
    image_max = args.train.image_max if image_set == 'train' else args.test.image_max
    image_min = args.train.image_min if image_set == 'train' else args.test.image_min

    transforms_list = args.dataset.transform.split(',') if 'transform' in args.dataset and args.dataset.transform is not None else []

    enable_multi_tables = args.dataset.enable_multi_tables if 'enable_multi_tables' in args.dataset else False
    
    include_heatmap = True if ('loss_aux_heatmap_weight' in args.loss and args.loss.loss_aux_heatmap_weight > 0) else False

    mean = list(map(float, args.dataset.mean.split(','))) if 'mean' in args.dataset else [0.909, 0.919, 0.915]
    std = list(map(float, args.dataset.std.split(','))) if 'std' in args.dataset else [0.203, 0.152 , 0.187]

    img_folder, ann_file = PATHS[image_set]

    dataset = CocoDetection(img_folder, 
                            ann_file, 
                            transforms=make_coco_transforms(image_set, max_seq_len, 
                                                            args.dataset.loss_eos_token_weight, 
                                                            ignore_aspect_ratio=False,
                                                            seq_version=args.dataset.seq_version, 
                                                            vocab=vocab,
                                                            image_max=image_max, 
                                                            image_min=image_min,
                                                            interpolation_mode=interpolation_mode,
                                                            transforms_list=transforms_list,
                                                            enable_multi_tables=enable_multi_tables,
                                                            normalize_dist=[mean, std],
                                                            include_heatmap=include_heatmap,
                                                            include_empty_cell=True,
                                                            ), 
                            max_seq_len=max_seq_len,
                            version=args.dataset.seq_version,
                            vocab=vocab,
                            include_heatmap=include_heatmap,
                            include_empty_cells=True,)
    print('build dataset')
    return dataset
    