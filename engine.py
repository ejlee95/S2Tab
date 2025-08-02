# -----------------------------------------------------------------------
# S2Tab official code : engine.py
# -----------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -----------------------------------------------------------------------
import sys
from typing import Iterable
import math
import torch
import torchvision
import json, orjson
from pathlib import Path
from copy import deepcopy
from PIL import Image
import numpy as np

import util.misc as utils
from util.visualization import draw_cell
from datasets.coco_eval import CocoEvaluator
from eval import TEDS, format_html, grits_eval
from eval.apply_gt_ocr import content_match
from eval.grits_util.table_datasets import PDFTablesDataset, ToTensor, Compose

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, 
                    log_writer: utils.TensorboardLogger=None, max_norm: float=0,
                    scaler: torch.cuda.amp.GradScaler = None, use_amp: bool=False,
                    lr_scheduler: torch.optim.lr_scheduler = None, lr_scheduler_scheme: list = [0, -1]):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    step = 0
    lr_schedule_by_step, lr_schedule_by_epoch = lr_scheduler_scheme

    for samples, target in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k,v in t.items()} for t in target]

        input_coord_seq = torch.stack([t['input_seq']['seq']['coord'] for t in targets], dim=0)
        input_delta_seq = torch.stack([t['input_seq']['seq']['delta'] for t in targets], dim=0)
        input_token_seq = torch.stack([t['input_seq']['seq']['token'] for t in targets], dim=0)

        input_mask = torch.stack([t['input_seq']['mask'] for t in targets], dim=0)
        input_sequences = {'coord': input_coord_seq,
                        'delta': input_delta_seq,
                        'token': input_token_seq,
                        'mask': input_mask,}
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if 'heatmap' in targets[0]:
            tgt_heatmap = [t['heatmap'].unsqueeze(0) for t in targets]
            tgt_heatmap = utils.nested_tensor_from_tensor_list(tgt_heatmap, defined_size=[1]+[samples.tensors.shape[-2]//4, samples.tensors.shape[-1]//4])
        
        if 'aux_vis_clip' in criterion.losses or \
            'aux_vis_clip_softmax' in criterion.losses or \
            ('aux_ocr' in criterion.losses and criterion.weight_dict['loss_ocr'] > 0):
            tgt_boxes = torch.stack([t['target_seq']['seq']['coord'] for t in targets], dim=0) # x,y,w,h in normalized coordinate
            tgt_sizes = torch.stack([t["size"] for t in targets], dim=0) # (h,w)
            input_sequences['tgt_roi_boxes'] = tgt_boxes
            input_sequences['tgt_roi_image_sizes'] = tgt_sizes
        
        if 'aux_ocr' in criterion.losses and criterion.weight_dict['loss_ocr'] > 0:
           gt_box_mask = torch.stack([t['target_seq']['mask_box'] for t in targets], dim=0)
           input_sequences['tgt_roi_mask'] = gt_box_mask
           input_sequences['ocr_roi_option'] = 'ground truth' if epoch <= 50 else 'prediction' # optional, total epochs 60
        
        if 'content' in targets[0]['input_seq']['seq']:
            input_sequences['content'] = torch.stack([t['input_seq']['seq']['content'] for t in targets], dim=0)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            outputs = model(samples, input_sequences, is_train=True)
            if 'heatmap' in targets[0]:
                loss_dict, aux = criterion(outputs, targets, heatmap=tgt_heatmap)
            else:
                loss_dict, aux = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # reduce losses over all GPUs for logging
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stop training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                torch.nn.utils.clip_grad_norm_(criterion.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                torch.nn.utils.clip_grad_norm_(criterion.parameters(), max_norm)
            optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if step < lr_schedule_by_step:
            lr_scheduler.step()

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_class=loss_dict['loss_class'], head="loss")
            if 'loss_boxes' in loss_dict and weight_dict['loss_boxes'] > 0:
                log_writer.update(loss_boxes=loss_dict['loss_boxes'], head="loss")
            if 'loss_boxes_giou' in loss_dict and weight_dict['loss_boxes_giou'] > 0:
                log_writer.update(loss_boxes_giou=loss_dict['loss_boxes_giou'], head="loss")
            if 'loss_aux_heatmap' in loss_dict and weight_dict['loss_aux_heatmap'] > 0:
                log_writer.update(loss_aux_heatmap=loss_dict['loss_aux_heatmap'], head="loss")
                if step % print_freq == 0:
                    pred_center_heatmap = outputs['pred_center_heatmap']
                    pred_center_heatmap = torch.nn.functional.sigmoid(pred_center_heatmap)
                    log_writer.add_image(torchvision.utils.make_grid(pred_center_heatmap, pad_value=1.0), tag='center_heatmap/pred')
                    tgt_center_heatmap = aux['tgt_heatmap']
                    log_writer.add_image(torchvision.utils.make_grid(tgt_center_heatmap, pad_value=1.0), tag='center_heatmap/gt')
                    tgt_center_positive_mask = aux['tgt_positive_mask']
                    log_writer.add_image(torchvision.utils.make_grid(tgt_center_positive_mask, pad_value=1.0), tag='center_heatmap/gt_positive')
                    # log_writer.add_image(torchvision.utils.make_grid(samples.tensors, pad_value=1.0), tag='center_heatmap/input')
            if 'loss_vis_clip' in loss_dict and weight_dict['loss_vis_clip'] > 0:
                log_writer.update(loss_vis_clip=loss_dict['loss_vis_clip'], head="loss")
                log_writer.update(t=criterion.t.item(), head="param/sigmoid_clip_loss")
                log_writer.update(b=criterion.b.item(), head="param/sigmoid_clip_loss")
            if 'loss_vis_clip_softmax' in loss_dict and weight_dict['loss_vis_clip_softmax'] > 0:
                log_writer.update(loss_vis_clip_softmax=loss_dict['loss_vis_clip_softmax'], head="loss")
            if 'loss_ocr' in loss_dict and weight_dict['loss_ocr'] > 0:
                log_writer.update(loss_ocr=loss_dict['loss_ocr'], head="loss")
            log_writer.update(lr=optimizer.param_groups[0]["lr"], head="opt")
            if len(optimizer.param_groups) > 1:
                log_writer.update(lr_backbone=optimizer.param_groups[1]["lr"], head="opt")
            log_writer.set_step()

        step += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, postprocessors, data_loader, base_ds, device, 
            output_dir=None, draw=False, dataset_mode='fintabnet',
            pp=0, draw_attention=False, gtpath=None, apply_ocr=None, content=False):
    if output_dir is not None:
        (output_dir / "PredRes").mkdir(parents=True, exist_ok=True)
        (output_dir / "detection").mkdir(parents=True, exist_ok=True)        
        if dataset_mode == 'icdar2019':
            (output_dir / 'str').mkdir(parents=True, exist_ok=True)
        if dataset_mode == 'pubtables' and pp == 1 and gtpath is not None:
            grits_dataset = PDFTablesDataset(Path(gtpath, 'test'),
                                             transforms=Compose([ToTensor()]),
                                             max_size=None,
                                             do_crop=False,
                                             make_coco=False, #True,
                                             include_eval=True,
                                             image_extension='.jpg',
                                             xml_fileset='test_filelist.txt',
                                             class_map={
                                                'table': 0,
                                                'table column': 1,
                                                'table row': 2,
                                                'table column header': 3,
                                                'table projected row header': 4,
                                                'table spanning cell': 5,
                                                'no object': 6
                                                }
                                             )

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = tuple(['bbox'])
    if base_ds == None:
        print('not COCO dataset. Run evalation without coco evaluation.')
        coco_evaluator = None
    else:    
        coco_evaluator = CocoEvaluator(base_ds, iou_types)
        print('COCO eval init')

    gt_all = {}
    # Quantitative Evaluation only
    if gtpath is not None and output_dir is not None and pp==1 and \
        (output_dir / 'pred.json').exists():
        print('prediction already done. Only evaluate the performance.')
        with open(output_dir / 'pred.json', 'r') as f:
            pred = orjson.loads(f.read())
        coco_evaluator = None
    else:
        if apply_ocr == 'gt' and gtpath is not None and \
            str(gtpath).endswith('.jsonl') and \
            (dataset_mode == 'fintabnet' or dataset_mode == 'pubtabnet') and pp==1:
            print('gt ocr will be matched with the predictions')
            with open(gtpath, 'r') as f:
                gt_lines = f.readlines()
            for line in gt_lines:
                obj = json.loads(line)
                if obj['split'] == 'train': continue
                html = format_html(obj['html'], wofunc=True if dataset_mode=='fintabnet' else False)
                gt_all[obj['filename']] = {'html': html, 'anno_html': obj['html']}

        pred = {}
        for samples, targets in metric_logger.log_every(data_loader, 10, header):
            samples = samples.to(device)
            for t in targets:
                t['start_ind'] = torch.LongTensor([0])
            targets = [{k: to_device(v, device) for k,v in t.items()} for t in targets]
            
            input_sequences = {} # for ocr..
            if hasattr(model.transformer, 'vision_decoder') and model.transformer.vision_decoder is not None:
                tgt_sizes = torch.stack([t["size"] for t in targets], dim=0) # (h,w)
                input_sequences['tgt_roi_image_sizes'] = tgt_sizes

            outputs = model(samples, draw_attention=draw_attention, sources=input_sequences)

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0) # (h,w)
            # for attention visualization
            outputs['input_size'] = samples.tensors.shape[2:]
            outputs['input_sizes'] = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors(outputs, orig_target_sizes, pp)
            results = {target['image_id'].item(): res for target, res in zip(targets, results)}
            if coco_evaluator is not None:
                coco_evaluator.update(results)

            if output_dir is not None:
                for i, (k, v) in enumerate(results.items()):
                    if base_ds == None:
                        try:
                            v['filename'] = data_loader.dataset.filenames[k]
                        except:
                            v['filename'] = data_loader.dataset.filenames[k-data_loader.dataset.id_offset]
                    else:
                        v['filename'] = base_ds.loadImgs(k)[0]['file_name']
                    v['size'] = targets[i]['size'].cpu().numpy()
                    
                    if content: # filter empty cells!!
                        if 'tokens' not in v or ('tokens' in v and v['tokens'] == None):
                            is_empty = v['empty']
                            ocr_results = []
                            for x in is_empty:
                                if x == 1: ocr_results.append('')
                                else: ocr_results.append('cell')
                            v['tokens'] = ocr_results

                    if len(gt_all) > 0:
                        # apply gt ocr
                        pred_cell_boxes = deepcopy(v['boxes'])
                        pred_cell_boxes[:, 2:] += pred_cell_boxes[:, :2]
                        if 'path' in gt_all and len(gt_all) == 1: # pubtables
                            # load gt word json
                            basename = v['filename'].replace('.jpg', '')
                            idx = grits_dataset.page_ids_to_idx[basename]
                            _, target = grits_dataset[idx]
                            img_filepath = target["img_path"]
                            img_filename = img_filepath.split("/")[-1]
                            img_words_filepath = gt_all['path'] / img_filename.replace(".jpg", "_words.json")
                            target["img_words_path"] = img_words_filepath
                            with open(img_words_filepath, 'r') as f:
                                true_page_tokens = json.load(f)
                            v['tokens'] = content_match(pred_cell_boxes, 
                                                        true_page_tokens, 
                                                        gt_mode='xyxy', 
                                                        mode=dataset_mode, 
                                                        empty=v['empty'] if 'empty' in v else None)
                        elif dataset_mode == 'scitsr':
                            v['tokens'] = content_match(pred_cell_boxes, 
                                                        gt_all[v['filename']], 
                                                        gt_mode='xxyy', 
                                                        mode=dataset_mode, 
                                                        empty=v['empty'] if 'empty' in v else None)
                        else: # pubtabnet, fintabnet
                            v['tokens'] = content_match(pred_cell_boxes, 
                                                        gt_all[v['filename']]['anno_html'], 
                                                        gt_mode='xyxy' if dataset_mode == 'pubtabnet' else 'xywh', 
                                                        mode=dataset_mode, 
                                                        empty=v['empty'] if 'empty' in v else None)
                    
                if draw:
                    if base_ds is not None:
                        draw_cell(results, data_loader.dataset.root, output_dir / "PredRes")
                    else:
                        draw_cell(results, data_loader.dataset.image_root, output_dir / "PredRes")
                    draw = False

            if output_dir is not None: # save the table structure
                for k, v in results.items():
                    output_dict = {'scores': v['scores'].cpu().tolist(),
                                    'labels': v['labels'].cpu().tolist(),
                                    'boxes': v['boxes'].tolist(),
                                    'id': k}
                    if 'empty' in v:
                        output_dict['empty'] = v['empty'].cpu().tolist()
                    with open(output_dir / "detection" / (v['filename'].replace('.png','.json').replace('.jpg','.json')), 'w') as f:
                        json.dump(output_dict, f, indent=2)

                dataset_mode = dataset_mode.lower()
                if dataset_mode == 'scitsr' or dataset_mode == 'wtw-dar' or dataset_mode == 'pubtables':
                    for k, v in results.items():
                        cells = utils.format_scitsr(v)
                        pred[v['filename']] = cells
                elif dataset_mode == 'pubtabnet' or dataset_mode == 'fintabnet' or dataset_mode == 'wtw-teds':
                    for k, v in results.items():
                        cell_structure = utils.format_ptn(v, dataset_mode)

                        pred_html = format_html(cell_structure, wofunc=True if dataset_mode=='fintabnet' or dataset_mode=='wtw-teds' else False)
                        if dataset_mode == 'pubtabnet':
                            pred_html = utils.deal_bb(pred_html)
                        pred[v['filename']] = pred_html
                    # with open(output_dir / 'pred.json', 'w') as f:
                    #     json.dump(pred, f, indent=2)
                elif dataset_mode == 'icdar2019':
                    assert gtpath is not None
                    for k, v in results.items():
                        doc_name = '_'.join(v['filename'].split('_')[:-1])
                        table_id = int(v['filename'].split('_')[-1].rstrip('.jpg'))
                        gt_xml_path = Path(gtpath) / f'{doc_name}.xml'
                        table_tree = utils.format_icdar2019(v, gt_xml_path, table_id)
                        pred[doc_name] = pred.get(doc_name, {})
                        pred[doc_name][table_id] = table_tree

        if len(pred) > 0 and dataset_mode != 'icdar2019':
            with open(output_dir / 'pred.json', 'w') as f:
                json.dump(pred, f, indent=2)

    # Quantitative Evaluation
    if (dataset_mode == 'fintabnet' or dataset_mode == 'pubtabnet' or dataset_mode == 'wtw-teds') and \
        output_dir is not None and pp != 0 and gtpath != None:
        filenames = list(pred.keys())
        if len(gt_all) == 0:
            with open(gtpath, 'r') as f:
                gt_lines = f.readlines()

            gt = {}
            for line in gt_lines:
                obj = json.loads(line)
                if obj['filename'] in filenames:
                    html = format_html(obj['html'], wofunc=True if dataset_mode=='fintabnet' else False)
                    gt[obj['filename']] = {'html': html}
        else:
            gt = {}
            for fname in filenames:
                gt[fname] = {'html': gt_all[fname]['html']}

        with open(output_dir / 'gt.json', 'w') as f:
            json.dump(gt, f, indent=2)
        
        teds = TEDS(structure_only=True, n_jobs=4) #, ignore_nodes=['b', 'i', 'sup', 'sub'])
        scores = teds.batch_evaluate(pred, gt)
        utils.summarize_teds_score(scores, filenames, output_dir, 'struc', gt)
        if apply_ocr is not None:
            teds = TEDS(structure_only=False, n_jobs=16)
            scores = teds.batch_evaluate(pred, gt)
            utils.summarize_teds_score(scores, filenames, output_dir, 'full', gt)

    elif (dataset_mode == 'scitsr' or dataset_mode == 'wtw-dar') and output_dir is not None and gtpath != None:
        dar_score = {
                    #  'dar_recall_con_original': {'mean': 0, 'all': {}},
                    #  'dar_precision_con_original': {'mean': 0, 'all': {}},
                    #  'dar_con_original': {'mean': 0, 'all': {}},
                    #  'dar_recall_con': {'mean': 0, 'all': {}},
                    #  'dar_precision_con': {'mean': 0, 'all': {}},
                    #  'dar_con': {'mean': 0, 'all': {}},
                     'dar_recall_loc': {'mean': 0, 'all': {}}, # macro
                     'dar_precision_loc': {'mean': 0, 'all': {}}, # macro
                     'dar_loc': {'mean': 0, 'all': {}}, # macro
                     'true_positive': {'mean': 0, 'all': {}},
                     'true_adjacency': {'mean': 0, 'all': {}},
                     'pred_adjacency': {'mean': 0, 'all': {}},
                     }
        gt = {}
        check_gt_less_than_pred = False
        for fname in pred.keys():
            try:
                with open(Path(gtpath) / fname.replace('.png', '.json').replace('.jpg', '.json'), 'r') as f:
                    target = json.load(f)
                    target.update({'filename': fname})
                gt[fname] = target
            except:
                check_gt_less_than_pred = True
                continue
        if check_gt_less_than_pred:
            print(f'Number of gt tables ({len(gt)}) are less than pred tables ({len(pred)}). Calculate DAR only overlapping files.')
            keys = list(gt.keys())
            pred = {k: pred[k] for k in keys}

        dar_score = grits_eval.eval_tsr_mp(gt, pred, dar_score)
            
        for kk, vv in dar_score.items():
            vv['mean'] = vv['mean'] / len(vv['all'])
        
        dar_precision_loc = dar_score['true_positive']['mean'] / dar_score['pred_adjacency']['mean']
        dar_recall_loc = dar_score['true_positive']['mean'] / dar_score['true_adjacency']['mean']
        dar_f1_loc = 2 * dar_precision_loc * dar_recall_loc / (dar_precision_loc + dar_recall_loc)

        dar_score['micro_average'] = {'precision': dar_precision_loc,
                                      'recall': dar_recall_loc,
                                      'f1': dar_f1_loc,}
        
        with open(output_dir / 'dar.json', 'w') as f:
            json.dump(dar_score, f, indent=2)
        
        print(f'DAR-Loc without empty cells. Macro Precision:{dar_score["dar_precision_loc"]["mean"]}, Recall:{dar_score["dar_recall_loc"]["mean"]}, F1: {dar_score["dar_loc"]["mean"]}')
        print(f'DAR-Loc without empty cells. Micro Precision:{dar_score["micro_average"]["precision"]}, Recall:{dar_score["micro_average"]["recall"]}, F1: {dar_score["micro_average"]["f1"]}')
        with open(output_dir / 'log.txt', 'a') as f:
            f.write(f'\nDAR-Loc without empty cells. Macro Precision:{dar_score["dar_precision_loc"]["mean"]}, Recall:{dar_score["dar_recall_loc"]["mean"]}, F1: {dar_score["dar_loc"]["mean"]}')
            f.write(f'\nDAR-Loc without empty cells. Micro Precision:{dar_score["micro_average"]["precision"]}, Recall:{dar_score["micro_average"]["recall"]}, F1: {dar_score["micro_average"]["f1"]}')

    elif dataset_mode == 'pubtables' and output_dir is not None and gtpath != None:
        # grits
        step = 1000
        total_pred_keys = list(pred.keys())
        num_preds = len(total_pred_keys)
        check_gt_less_than_pred = False
        for i in range(0, num_preds, step):
            _gt = {}
            _pred = {}
            keys = total_pred_keys[i:i+step]
            for k in keys:
                _pred[k] = pred[k]
            for fname in _pred.keys():
                basename = fname.replace('.jpg', '')
                idx = grits_dataset.page_ids_to_idx[basename]
                _, target = grits_dataset[idx]
                img_filepath = target["img_path"]
                img_filename = img_filepath.split("/")[-1]
                img_words_filepath = Path(gtpath) / "words" / img_filename.replace(".jpg", "_words.json")
                target["img_words_path"] = img_words_filepath

                _gt[fname] = target
            print(f'...Evaluate {i} to {i+step} files')

            grits_score = grits_eval.eval_tsr_grits_mp(_gt, _pred, 'grits-wocon' if apply_ocr==None else 'grits')

            grits_summary = grits_eval.compute_metrics_summary(grits_score, mode='grits-wocon' if apply_ocr==None else 'grits')
            grits_eval.print_metrics_summary(grits_summary)

            grits_score += [grits_summary]
            with open(output_dir / f'grits_{i}.json', 'w') as f:
                json.dump(grits_score, f, indent=2)
        
    elif (dataset_mode == 'icdar2019') and output_dir is not None and gtpath != None:
        # merge table roots to one tree for each document
        xmldir = Path(gtpath)
        utils.merge_tables_to_document(pred, xmldir, output_dir / 'str')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Average stats: ', metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        # detection
        stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()

    return stats, coco_evaluator

def to_device(target, device):
    if target is None or isinstance(target, str) or isinstance(target, int):
        return target
    elif isinstance(target, (torch.Tensor, utils.NestedTensor)):
        return target.to(device)
    elif isinstance(target, (list, tuple)):
        targets = []
        for t in target:
            if isinstance(t, dict):
                for k, v in t.items():
                    if not isinstance(v, list):
                        v = to_device(v, device)
                        t[k] = v
            elif isinstance(t, torch.Tensor):
                t = t.to(device)
            else:
                raise NotImplementedError(f'to_device for t {t} is not implemented.')
            targets.append(t)
        if isinstance(target, tuple):
            targets = tuple(targets)
    elif isinstance(target, dict):
        targets = {}
        for k, v in target.items():
            if not isinstance(v, list):
                v = to_device(v, device)
            targets[k] = v
    else:
        raise NotImplementedError(f'type of {target} is not Implemented.')
    
    return targets