# -----------------------------------------------------------------------
# S2Tab official code : datasets/transforms.py
# -----------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -----------------------------------------------------------------------

import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import math
import numpy as np

def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()

    if "structure" in target:
        # change the order of the cells
        structure = target["structure"]
        max_col = structure[:, 1].max()
        structure[:, :2] = max_col - structure[:, :2].flip(1)
    else:
        structure = None

    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        # change the order of the cells
        if structure is not None:
            idx = np.lexsort((structure.numpy()[:, 0], structure.numpy()[:, 2]))
            boxes = boxes[idx]
        else:
            idx = np.lexsort((boxes.numpy()[:, 0], boxes.numpy()[:, 2]))
            boxes = boxes[idx]
        target["boxes"] = boxes
    
    if "structure" in target:
        structure = structure[idx]
        target["structure"] = structure

    if "labels" in target:
        # change the order of the cells
        labels = target["labels"]
        labels = labels[idx]
        target["labels"] = labels

    if "endrow" in target:
        ori_endrow = target["endrow"]
        # change endrow labels
        endrow = structure[:, 2][1:] - structure[:, 2][:-1]
        # if ori_endrow[-1] != 1:
        #     print('cropped sequence')
        endrow = torch.cat((endrow, ori_endrow[-1:]), dim=0)
        target["endrow"] = endrow

    if "is_empty" in target:
        # change the order of the cells
        is_empty = target['is_empty']
        is_empty = is_empty[idx]
        target["is_empty"] = is_empty

    if "row_boxes" in target:
        boxes = target["row_boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["row_boxes"] = boxes
    
    if "header_box" in target:
        boxes = target["header_box"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["header_box"] = boxes

    return flipped_image, target

def crop_seq(target, max_seq_len):
    if target is None:
        return target
    
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:max_seq_len - 1]
        target["boxes"] = boxes

    if "text_boxes" in target:
        boxes = target["text_boxes"]
        boxes = boxes[:max_seq_len - 1]
        target["text_boxes"] = boxes

    if "labels" in target:
        labels = target["labels"]
        labels = labels[:max_seq_len - 1]
        target["labels"] = labels
    
    if "endrow" in target:
        endrow = target["endrow"]
        endrow = endrow[:max_seq_len - 1]
        target["endrow"] = endrow
    
    if "structure" in target:
        structure = target["structure"]
        structure = structure[:max_seq_len - 1]
        target["structure"] = structure
    
    if "is_empty" in target:
        is_empty = target["is_empty"]
        is_empty = is_empty[:max_seq_len - 1]
        target["is_empty"] = is_empty
    
    return target


class MaxSeqCrop(object):
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len
    def __call__(self, img, target):
        target = crop_seq(target, self.max_seq_len)
        return img, target

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target

def crop(image, target, region):
    # TODO: row_boxes and header_box are not modified yet!!
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region # top, left, height, width

    # 'structure' is shifted (in the cropped image), but it doesn't affect the training performance and evaluation.
    fields = ["labels", "header_labels", "is_empty", "structure"] # "endrow", 

    if "boxes" in target:
        boxes = target["boxes"] # left, top, right, bottom
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        
        # for validity check
        keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1) # filter out-of-image box
        if keep.nonzero().shape[0] == 0: # if cropped image doesn't contain any object
            return image, target # then don't crop
        
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    # image size
    target["size"] = torch.tensor([h, w])

    if "boxes" in target:
        cropped_boxes = target['boxes'].reshape(-1, 2, 2)
        keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1) # filter out-of-image box
        
        for field in fields:
            if field in target:
                target[field] = target[field][keep]
        
        field = "endrow"
        removed_eor_idx = torch.logical_and(keep == 0, target[field] == 1).nonzero() # will-be-removed & endrow = 1 components
        target[field][torch.maximum(removed_eor_idx - 1, torch.zeros_like(removed_eor_idx))] = 1 # set eor among valid components by left-shifting
        target[field] = target[field][keep]

    return cropped_image, target

def resize(image, target, size, max_size=None, mode='bilinear'):
    # size can be min_size (scalar) or (w, h) tuple
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        # size = min_size
        w, h = image_size
        if max_size is not None:
            min_ori_size = float(min((w, h)))
            max_ori_size = float(max((w, h)))
            if max_ori_size / min_ori_size * size > max_size: 
                # if mode == 'coco':
                #     size = int(round(max_size * min_ori_size / max_ori_size)) 
                # else: # coco2
                size = int(math.floor(max_size * min_ori_size / max_ori_size))
                
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)
        
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            ow = int(size * w / h)
            oh = size
        
        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)
    
    size = get_size(image.size, size, max_size)
    interp_mode = F.InterpolationMode.BICUBIC if mode == 'bicubic' else \
                  F.InterpolationMode.NEAREST if mode == 'nearest' else \
                  F.InterpolationMode.LANCZOS if mode == 'lanczos' else \
                  F.InterpolationMode.HAMMING if mode == 'hamming' else \
                  F.InterpolationMode.BOX if mode == 'box' else \
                  F.InterpolationMode.BILINEAR # if mode == 'bilinear'
    rescaled_image = F.resize(image, size, interpolation=interp_mode)

    if target is None:
        return rescaled_image, None
    
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes
    
    if "text_boxes" in target:
        boxes = target["text_boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["text_boxes"] = scaled_boxes

    if "row_boxes" in target:
        boxes = target["row_boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["row_boxes"] = scaled_boxes
    
    if "header_box" in target:
        boxes = target["header_box"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["header_box"] = scaled_boxes
    
    if "area" in target:
        area = target["area"]
        scaled_area = area * ratio_width * ratio_height
        target["area"] = scaled_area

    if "table_box" in target:
        boxes = target["table_box"] # x,y,x,y
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["table_box"] = scaled_boxes
    
    if "margin" in target:
        margins = target["margin"] # (num, 4, 2) left(-), left(+), up(-), up(+), right(-), right(+), down(-), down(+)
        scaled_margins = margins * torch.as_tensor([[ratio_width, ratio_width], [ratio_height, ratio_height], 
                                                    [ratio_width, ratio_width], [ratio_height, ratio_height]])
        target["margin"] = scaled_margins
    
    if "heatmap" in target:
        heatmap = target["heatmap"].unsqueeze(0)
        heatmap = F.resize(heatmap, size, interpolation=F.InterpolationMode.NEAREST)
        target["heatmap"] = heatmap[0]
    
    h, w = size
    target["size"] = torch.tensor([h, w])

    if "line_segm" in target:
        lines = target["line_segm"] # (2, pos), [0,:] y, [1,:] x
        lines = lines * torch.as_tensor([ratio_height, ratio_width])[:, None]
        target["line_segm"] = lines
    
    if "line_border_row" in target:
        lines = target["line_border_row"] # (N, 2) y1-y2
        lines = lines * torch.as_tensor([ratio_height, ratio_height])[None]
        target["line_border_row"] = lines
    
        lines = target["line_border_col"] # (N, 2) x1-x2
        lines = lines * torch.as_tensor([ratio_width, ratio_width])[None]
        target["line_border_col"] = lines

    return rescaled_image, target


def pad(image, target, padding):
    # pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    target["size"] = torch.tensor(padded_image.size[::-1]) # h,w
    target["padding"] = torch.tensor([0, 0, padding[0], padding[1]]) # left, top, right, bottom

    return padded_image, target

class Pad(object):
    def __init__(self, max_size):
        self.max_size = (max_size, max_size) # x,y
    
    def __call__(self, image, target):
        w, h = image.size
        pad_x = self.max_size[0] - w
        pad_y = self.max_size[1] - h
        return pad(image, target, (pad_x, pad_y))

def rotate270(image, target):
    rotated_image = F.rotate(image, 270, F.InterpolationMode.BILINEAR, expand=True)

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"] # x,y,x,y
        # rotated_boxes = torch.empty_like(boxes)
        rotated_boxes = boxes[:, [1, 0, 3, 2]]
        target["boxes"] = rotated_boxes
    
    if "row_boxes" in target:
        boxes = target["row_boxes"] # x,y,x,y
        # rotated_boxes = torch.empty_like(boxes)
        rotated_boxes = boxes[:, [1, 0, 3, 2]]
        target["row_boxes"] = rotated_boxes
    
    if "header_box" in target:
        boxes = target["header_box"] # x,y,x,y
        # rotated_boxes = torch.empty_like(boxes)
        rotated_boxes = boxes[:, [1, 0, 3, 2]]
        target["header_box"] = rotated_boxes
    
    target['size'] = torch.as_tensor([target['size'][1], target['size'][0]])
    
    return rotated_image, target

def rotate(image, target, angle):
    rotated_image = F.rotate(image, angle, F.InterpolationMode.BILINEAR) #, fill=[255,255,255])

    if target is None:
        return rotated_image, target
    
    if "boxes" in target:
        degree = math.pi * angle / 180.0
        sin, cos = math.sin(degree), math.cos(degree)
        matrix = torch.as_tensor([[cos, -sin], [sin, cos]])

        h, w = image.height, image.width

        pts = target["boxes"]
        bs = pts.shape[0]
        lt = pts[:, :2]
        rb = pts[:, 2:]
        rt = torch.stack((rb[:, 0], lt[:, 1]), dim=1) # N, 2
        lb = torch.stack((lt[:, 0], rb[:, 1]), dim=1) # N, 2
        pts = torch.stack((lt, rb, rt, lb), dim=1) # N, 4, 2
        pts = pts.reshape(bs * 4, 2) # N*4, 2 (lt, rb, rt, lb)
        # adjust (0,0) as the center of the image
        pts[..., 0] -= w/2
        pts[..., 1] -= h/2 

        pts = pts @ matrix
        pts = pts.reshape((bs, 4, 2))
        pts[..., 0] += w / 2
        pts[..., 1] += h / 2
        boxes = torch.stack((pts[..., 0].min(dim=1)[0], 
                           pts[..., 1].min(dim=1)[0],
                           pts[..., 0].max(dim=1)[0],
                           pts[..., 1].max(dim=1)[0]),
                          dim=1)
        boxes[:, 0::2].clamp_(min=0, max=image.width)
        boxes[:, 1::2].clamp_(min=0, max=image.height)
        target["boxes"] = boxes

    return rotated_image, target

class Transpose(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image, target = rotate270(image, target)
            image, target = hflip(image, target)
            return image, target
        else:
            return image, target

class RandomCrop(object):
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image, target):
        region = T.RandomCrop.get_params(image, self.size)
        return crop(image, target, region)

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image_width, image_height = image.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(image, target, (crop_top, crop_left, crop_height, crop_width))

class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad
    
    def __call__(self, image, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(image, target, (pad_x, pad_y))

class RandomResize(object):
    def __init__(self, sizes, max_size=None, ignore_aspect_ratio=False, interpolation_mode='coco'):
        assert isinstance(sizes, (list, tuple, int))
        self.sizes = sizes
        self.max_size = max_size
        self.ignore_aspect_ratio = ignore_aspect_ratio
        self.interpolation_mode = interpolation_mode

    def __call__(self, image, target=None):
        size = random.choice(self.sizes)
        if self.ignore_aspect_ratio:
            return resize(image, target, (size, size), self.interpolation_mode)
        else:
            return resize(image, target, size, self.max_size, self.interpolation_mode)

class RandomRotate(object):
    def __init__(self, p=0.5):
        self.p = p
        self.degree = 10 # there are already-rotated images.. but don't have annotations..
    
    def __call__(self, image, target):
        if random.random() < self.p:
            angle = random.uniform(-self.degree, self.degree)
            return rotate(image, target, angle)
        else:
            return image, target

class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)

class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, self.mean, self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        if "text_boxes" in target:
            boxes = target["text_boxes"]
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["text_boxes"] = boxes
        if "row_boxes" in target:
            boxes = target["row_boxes"]
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["row_boxes"] = boxes
        if "header_box" in target:
            boxes = target["header_box"]
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["header_box"] = boxes
        if "perturb_boxes" in target:
            boxes = target["perturb_boxes"]
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["perturb_boxes"] = boxes
        if "table_box" in target:
            boxes = target["table_box"]
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            # box = box / torch.tensor([w, w, h, h], dtype=torch.float32)
            target["table_box"] = boxes
        return image, target
    
def adjust_heatmap(image, target, include_empty=True, box_key='boxes'):
    if target is None:
        return image, target
    
    H, W = target['size']
    h, w = H//4, W//4

    heatmap = target['heatmap'].clone()
    heatmap = F.resize(heatmap.unsqueeze(0), 
                       [h, w], 
                       T.InterpolationMode.NEAREST)[0]
    
    is_not_empty = (1-target["is_empty"].clone()).to(torch.bool)
    boxes = target[box_key].clone()
    boxes = boxes / 4
    if not include_empty:
        boxes = boxes[is_not_empty]

    centers = boxes[:,:2] + (boxes[:,2:] - boxes[:,:2]) / 2 # x,y
    centers = torch.round(centers) * ((centers % 0.5) != 0) + torch.round(centers+0.1) * ((centers % 0.5) == 0)
    centers = centers.to(torch.long)
    centers = centers.clamp(max=torch.as_tensor([[w-1,h-1]]))
    heatmap[centers[:,1], centers[:,0]] = 1

    target['heatmap'] = heatmap

    return image, target

class RefineHeatmap(object):
    def __init__(self, include_empty_cells, box_key):
        self.include_empty_cells = include_empty_cells
        self.box_key = box_key

    def __call__(self, image, target):
        return adjust_heatmap(image, target, self.include_empty_cells, self.box_key)
        

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        
        return image, target
    
    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n      {0}".format(t)
        format_string += "\n"
        return format_string

class ConstructSeq(object):
    def __init__(self, max_seq_len=100, loss_eos_token_weight=0.1, 
                        seq_version=None, vocab=None, enable_multi_tables=False):
        self.max_seq_len = max_seq_len
        self.loss_eos_token_weight = loss_eos_token_weight
        if seq_version == 'cell':
            from .sequence_cell import reconstruction_seq_from_boxes
        elif seq_version == 'content':
            from .sequence_content import reconstruction_seq_from_boxes
        else:
            raise NotImplementedError(f'seq_version {seq_version} is not implemented.')

        self.ftn = reconstruction_seq_from_boxes
        self.vocab = vocab
        self.enable_multi_tables = enable_multi_tables
    
    def __call__(self, image, target):
        assert "boxes" in target
        assert target["boxes"].dtype == torch.float32
        target = self.ftn(target, self.max_seq_len, 
                          self.loss_eos_token_weight, 
                          self.vocab, 
                          self.enable_multi_tables,
                          )
        return image, target
