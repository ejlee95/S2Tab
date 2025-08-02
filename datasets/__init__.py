# -----------------------------------------------------------------------
# S2Tab official code : datasets/__init__.py
# -----------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -----------------------------------------------------------------------
import torch.utils.data
import torchvision
from datasets import coco_base


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, coco_base.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    
    if isinstance(dataset, torchvision.datasets.CocoDetection) or \
        isinstance(dataset, coco_base.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args, vocab=None, model=None):
    if args.dataset.dataset_file == 'coco_cell':
        from .coco_cell import build as build_coco
        return build_coco(image_set, args, vocab)
    elif args.dataset.dataset_file == 'coco_content':
        from .coco_content import build as build_coco
        return build_coco(image_set, args, vocab, model)
    elif args.dataset.dataset_file == 'instance':
        from .coco_instance import build as build_inst
        return build_inst(image_set, args, vocab, model)

    raise ValueError(f'dataset {args.dataset.dataset_file} not supported')