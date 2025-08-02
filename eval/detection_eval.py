import json
import argparse
from pathlib import Path
import torch
import numpy as np
import cv2
import multiprocessing as mp
import sys
import tqdm
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from datasets import build_dataset, get_coco_api_from_dataset
from datasets.coco_eval import CocoEvaluator
from main import get_config

def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mp.pool.Pool._get_tasks(func, iterable, chunksize)
    result = mp.pool.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mp.pool.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)

def evaluation_whole(filenames, filenames_id):
    # for file, file_id in tqdm.tqdm(zip(filenames, filenames_id)):
    for file, file_id in zip(filenames, filenames_id):
        with open(file, 'r') as f:
            try:
                pred = json.load(f) # 'boxes', 'scores', 'labels'
            except Exception as e:
                print(file, e)
                continue
        basename = file.stem
        pred['labels'] = [1 for _ in range(len(pred['labels']))]
        res = {file_id: {k: np.asarray(v) for k, v in pred.items() if k != 'id'}}
        coco_evaluator.update(res)

def evaluation(file, file_id, exclude_empty):
    try:
        with open(file, 'r') as f:
            pred = json.load(f) # 'boxes', 'scores', 'labels'
    except Exception as e:
        print(file, e)
        return {}
    basename = file.stem
    if exclude_empty and 'empty' in pred:
        ### exclude empty cells
        empty = np.asarray(pred['empty'])
        mask = (1 - empty).astype(bool)
        scores = np.ones_like(np.asarray(pred['scores'])[mask]).tolist()
        labels = np.asarray(pred['labels'])[mask].tolist()
        boxes = np.asarray(pred['boxes'])[mask].tolist()
        pred['scores'] = scores
        pred['labels'] = labels
        pred['boxes'] = boxes
        ####
    pred['labels'] = [1 for _ in range(len(pred['labels']))]

    res = {file_id: {k: np.asarray(v) for k, v in pred.items() if k != 'id'}}
    # coco_evaluator.update(res)

    return res

def _main(args):
    # gt
    dataset = build_dataset('test', args, args.vocab)
    base_ds = get_coco_api_from_dataset(dataset)
    # global coco_evaluator
    coco_evaluator = CocoEvaluator(base_ds, tuple(['bbox']), True)
    # pred
    pred_dir = args.pred_path #.split(',')
    cnt = 0
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True)
    # map fnames to id
    imgs = base_ds.imgs
    map_fname_to_id = {v['file_name']: k for k, v in imgs.items()}

    mp.pool.Pool.istarmap = istarmap
    pool = mp.Pool(processes=4)

    fnames = list(map_fname_to_id.keys())
    fnames_id = [map_fname_to_id[k] for k in fnames]
    exclude_empty = [args.exclude_empty for _ in range(len(fnames))]

    # for pred_dir in pred_dirs:
    pred_dir_path = Path(pred_dir) / 'detection'
    fnames = [pred_dir_path / x.replace('.jpg','.json').replace('.png','.json') for x in fnames]

    for res in tqdm.tqdm(pool.istarmap(evaluation, zip(fnames, fnames_id, exclude_empty)), total=len(fnames)):
        coco_evaluator.update(res)
        
    # for file in tqdm.tqdm(zip(fnames, fnames_id)):
    #     evaluation(fnames, fnames_id)
    
    pool.close()
    pool.join()

    sys.stdout = open(output_path / "test_stats_detection.txt", "w")
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    sys.stdout.close()

    with open(output_path / "macro_eval.json", "w") as f:
        json.dump(coco_evaluator.per_image_eval, f, indent=2)

    torch.save(coco_evaluator, output_path / "eval_detection.pth")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', default='coco_content', choices=('coco_content', 'coco_cell'))
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--pred_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--exclude_empty', action='store_true')

    parser.add_argument('--config', type=str, help="config yaml file path.")

    args = parser.parse_args()

    cfg = get_config(args)

    _main(cfg)