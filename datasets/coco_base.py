# -----------------------------------------------------------------------
# S2Tab official code : datasets/coco_base.py
# -----------------------------------------------------------------------
# Modified from pycocotools and torchvision
# -----------------------------------------------------------------------
import os.path
from typing import Any, Callable, Optional, Tuple, List
from collections import defaultdict
import time
import pickle, orjson
from PIL import Image

import torchvision
from pycocotools.coco import COCO as py_COCO

class CocoDetection(torchvision.datasets.VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int) -> Image.Image:
        obj = self.coco.loadImgs(id)[0]
        path = obj["file_name"]
        if "type" in obj and obj["type"] != "instances":
            cls = obj['type']
        else:
            cls = None
        return Image.open(os.path.join(self.root, path)).convert("RGB"), cls

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image, cls = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)


class COCO(py_COCO):
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            if annotation_file.suffix == '.json':
                dataset = orjson.loads(open(annotation_file, 'r').read())
            elif annotation_file.suffix == '.pickle':
                dataset = pickle.load(open(annotation_file, 'rb'))
            else:
                raise ValueError(f"Invalid annotation file: {annotation_file}")
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            if dataset['info']['description'] == 'table Dataset - dummy':
                self.no_eval = True
            else:
                self.no_eval = False
            self.createIndex()