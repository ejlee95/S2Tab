# -----------------------------------------------------------------------
# S2Tab official code : datasets/coco_instance.py
# -----------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -----------------------------------------------------------------------

from pathlib import Path
from typing import Any, Callable, Optional, Tuple, List, Union
from PIL import Image
import pickle, orjson

import torchvision
import torch
from torchvision.transforms import functional as F



class CocoDetectionInst(torchvision.datasets.VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images and annotations are downloaded to.
        annPost (string): annotation (one file for single image) postfix json or pickle.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: Union[str, List],
        annPost: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        max_seq_len: int = 100,
        version: str='content',
        vocab: dict=None,
        include_heatmap=False,
        include_empty_cells=True,
        charset=None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        if isinstance(root, List):
            roots = root
            self.image_root = []
            self.ann_root = []
            self.filenames = []
            for rt in roots:
                # rt_path = Path(rt)
                image_root = rt / "Images"
                ann_root = rt / "Annotations"
                filenames = sorted(list(image_root.glob('*')))
                postfix = {x.stem: x.suffix for x in filenames}
                filenames = sorted(list(ann_root.glob('*')))
                filenames = [x.stem for x in filenames]
                filenames = [f'{x}{postfix[x]}' for x in filenames]
                self.image_root += [image_root] * len(filenames)
                self.ann_root += [ann_root] * len(filenames)
                self.filenames += filenames
        else:
            rt = Path(root)
            image_root = rt / "Images"
            ann_root = rt / "Annotations"
            filenames = sorted(list(image_root.glob('*')))
            postfix = {x.stem: x.suffix for x in filenames}
            filenames = sorted(list(ann_root.glob('*')))
            filenames = [x.stem for x in filenames]
            filenames = [f'{x}{postfix[x]}' for x in filenames]
            self.image_root = [image_root] * len(filenames)
            self.ann_root = [ann_root] * len(filenames)
            self.filenames = filenames

        if isinstance(annPost, List):
            self.post = annPost[0]
        else:
            self.post = annPost
        self._transforms = transforms
        if version == 'cell' or version == 'both':
            from .coco_cell import ConvertCocoPolysToMask
        else: # 'content'
            from .coco_content import ConvertCocoPolysToMask
        ENDROW_OFFSET=1000 if 'MAX_COL' not in vocab else 1000 if vocab['MAX_COL'] == 30 else 2500
        self.prepare = ConvertCocoPolysToMask(max_seq_len, 
                                              vocab['CLASS_OFFSET'], 
                                              ENDROW_OFFSET, 
                                              include_heatmap, 
                                              include_empty_cells,
                                              charset,
                                            #   vocab_file,
                                            #   vocab
                                            )
        self.seq_version = version

        self.no_eval = True

    def check_annotation(self, img, bbox, classes, image_id):
        if not isinstance(img, torch.Tensor):
            img = F.to_tensor(img)
        else:
            mean, std = [0.929, 0.935, 0.939], [0.027, 0.026, 0.027]
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

        draw_img = torchvision.utils.draw_bounding_boxes((img * 255).to(torch.uint8), bbox, 
                                                            ['cell' for _ in classes],
                                                            #[label_map[x.item()] for x in classes],
                                                            colors=order_color, width=2)
                                                            #[color_map[x.item()] for x in classes], width=2)
        draw_img = F.to_pil_image(draw_img)
        name = self.coco.loadImgs(image_id)[0]['file_name']
        draw_img.save(f'./check/{name}')

    def _load_image(self, path: str, idx: int):
        img = Image.open(self.image_root[idx] / path).convert("RGB")
        ann_path = path.replace('.png', f'.{self.post}').replace('.jpg', f'.{self.post}')
        if self.post == 'json':
            with open(self.ann_root[idx] / ann_path, 'r') as f:
                _cls = orjson.loads(f.read())['type']
        elif self.post == 'pickle':
            with open(self.ann_root[idx] / ann_path, 'rb') as f:
                _cls = pickle.load(f)['type']
        else:
            raise ValueError(f"postfixwhile loading target... postfix {self.post} is not supported.")

        return img, _cls

    def _load_target(self, path: str, idx: int) -> List[Any]:
        ann_path = path.replace('.png', f'.{self.post}').replace('.jpg', f'.{self.post}')
        if self.post == 'json':
            with open(self.ann_root[idx] / ann_path, 'r') as f:
                anns = orjson.loads(f.read())['annotations']
        elif self.post == 'pickle':
            with open(self.ann_root[idx] / ann_path, 'rb') as f:
                anns = pickle.load(f)['annotations']
        else:
            raise ValueError(f"postfixwhile loading target... postfix {self.post} is not supported.")
        return anns

    def _load_row_target(self, path: str, idx: int) -> List[Any]:
        ann_path = path.replace('.png', f'.{self.post}').replace('.jpg', f'.{self.post}')
        if self.post == 'json':
            with open(self.ann_root[idx] / ann_path, 'r') as f:
                anns = orjson.loads(f.read())
        elif self.post == 'pickle':
            with open(self.ann_root[idx] / ann_path, 'rb') as f:
                anns = pickle.load(f)
        else:
            raise ValueError(f"while loading row_target... postfix {self.post} is not supported.")
        if 'row_annotations' in anns:
            anns = anns['row_annotations']
        else:
            anns = []
        return anns


    def _load_col_target(self, path: str, idx: int) -> List[Any]:
        ann_path = path.replace('.png', f'.{self.post}').replace('.jpg', f'.{self.post}')
        if self.post == 'json':
            with open(self.ann_root[idx] / ann_path, 'r') as f:
                anns = orjson.loads(f.read())
        elif self.post == 'pickle':
            with open(self.ann_root[idx] / ann_path, 'rb') as f:
                anns = pickle.load(f)
        else:
            raise ValueError(f"while loading col_target... postfix {self.post} is not supported.")
        if 'col_annotations' in anns:
            anns = anns['col_annotations']
        else:
            anns = []
        return anns

    def _load_id(self, path: str, idx: int) -> int:
        ann_path = path.replace('.png', f'.{self.post}').replace('.jpg', f'.{self.post}')
        if self.post == 'json':
            with open(self.ann_root[idx] / ann_path, 'r') as f:
                _id = orjson.loads(f.read())['images'][0]['id']
        elif self.post == 'pickle':
            with open(self.ann_root[idx] / ann_path, 'rb') as f:
                _id = pickle.load(f)['images'][0]['id']
        else:
            raise ValueError(f"while loading id... postfix {self.post} is not supported.")
        
        return _id

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img, _cls = self._load_image(fname, idx)
        target = self._load_target(fname, idx)
        row_target = self._load_row_target(fname, idx)
        col_target = self._load_col_target(fname, idx)

        # img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self._load_id(fname, idx)
        target = {'image_id': image_id, 'annotations': target, 'row_annotations': row_target, 'col_annotations': col_target, 'cls': _cls}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        # self.check_annotation(img, target['boxes'], target['labels'], image_id)
        return img, target

    def __len__(self) -> int:
        return len(self.filenames)


def build(image_set, args, vocab=None, model=None):
    if ',' in args.dataset.coco_path:
        roots = args.dataset.coco_path.split(',')
        postfix = 'json'
        PATHS = {"train": [[], []], "val": [[], []], "test": [[], []]}
        for root in roots:
            rt = Path(root)
            assert rt.exists(), f'provided COCO path {root} does not exist!'
            PATHS["train"][0] += [rt / "train2017"]
            PATHS["train"][1] += [postfix]
            PATHS["val"][0] += [rt / "val2017"]
            PATHS["val"][1] += [postfix]
            PATHS["test"][0] += [rt / "test2017"]
            PATHS["test"][1] += [postfix]
    else:
        root = Path(args.dataset.coco_path)
        assert root.exists(), f'provided COCO path {root} does not exist!'
        postfix = 'json'
        PATHS = {
            "train": (root / "train2017", postfix),
            "val": (root / "val2017", postfix),
            "test": (root / "test2017", postfix),
        }

    if args.dataset.seq_version == 'cell':
        from .coco_cell import make_coco_transforms
    else: # 'content'
        from .coco_content import make_coco_transforms
    
    max_seq_len = args.train.max_seq_len if image_set == 'train' else args.test.max_seq_len

    interpolation_mode = args.train.interpolation if image_set == 'train' else args.test.interpolation

    image_max = args.train.image_max if image_set == 'train' else args.test.image_max
    image_min = args.train.image_min if image_set == 'train' else args.test.image_min

    transforms_list = args.dataset.transform.split(',') if 'transform' in args.dataset else []

    include_heatmap = True if ('loss_aux_heatmap_weight' in args.loss and args.loss.loss_aux_heatmap_weight > 0) else False

    vocab_file = args.dataset.vocab_file if 'vocab_file' in args.dataset else None
    
    if hasattr(model.transformer, 'vision_decoder') and model.transformer.vision_decoder is not None:
        charset = model.transformer.vision_decoder.charset
    else:
        charset = None

    img_folder, ann_postfix = PATHS[image_set]
    dataset = CocoDetectionInst(img_folder, 
                                ann_postfix, 
                                transforms=make_coco_transforms(image_set, max_seq_len, 
                                                                args.dataset.loss_eos_token_weight, 
                                                                ignore_aspect_ratio=False,
                                                                seq_version=args.dataset.seq_version, 
                                                                vocab=vocab,
                                                                image_max=image_max, 
                                                                image_min=image_min,
                                                                interpolation_mode=interpolation_mode,
                                                                transforms_list=transforms_list,
                                                                include_heatmap=include_heatmap,
                                                                include_empty_cell=True,
                                                                ), 
                                max_seq_len=max_seq_len,
                                version=args.dataset.seq_version,
                                vocab=vocab,
                                include_heatmap=include_heatmap,
                                include_empty_cells=True,
                                charset = charset,
                                )
    print('build dataset')
    return dataset
    