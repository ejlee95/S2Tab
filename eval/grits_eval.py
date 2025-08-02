"""
Copyright (C) 2021 Microsoft Corporation
Copied from table-transformer/src/eval.py
"""
import os
import sys
from collections import Counter
import json
import statistics as stat
from datetime import datetime
import multiprocessing
from itertools import repeat
from functools import partial
import tqdm
import math
import multiprocessing as mp
import multiprocessing.pool as mpp

import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fitz import Rect
from PIL import Image

# sys.path.append("../detr")
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from eval.grits_util import postprocess
# import .grits as grits
from .grits import compute_fscore, cells_to_grid, cells_to_relspan_grid, output_to_dilatedbbox_grid, cell_to_dilatedbbox_grid
from .grits import grits_con, grits_top, grits_loc


structure_class_names = [
    'table', 'table column', 'table row', 'table column header',
    'table projected row header', 'table spanning cell', 'no object'
]
structure_class_map = {k: v for v, k in enumerate(structure_class_names)}
structure_class_thresholds = {
    "table": 0.5,
    "table column": 0.5,
    "table row": 0.5,
    "table column header": 0.5,
    "table projected row header": 0.5,
    "table spanning cell": 0.5,
    "no object": 10
}


normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def objects_to_cells(bboxes, labels, scores, page_tokens, structure_class_names, structure_class_thresholds, structure_class_map):
    bboxes, scores, labels = postprocess.apply_class_thresholds(bboxes, labels, scores,
                                            structure_class_names,
                                            structure_class_thresholds)

    table_objects = []
    for bbox, score, label in zip(bboxes, scores, labels):
        table_objects.append({'bbox': bbox, 'score': score, 'label': label})
        
    table = {'objects': table_objects, 'page_num': 0} 
    
    table_class_objects = [obj for obj in table_objects if obj['label'] == structure_class_map['table']]
    if len(table_class_objects) > 1:
        table_class_objects = sorted(table_class_objects, key=lambda x: x['score'], reverse=True)
    try:
        table_bbox = list(table_class_objects[0]['bbox'])
    except:
        table_bbox = (0,0,1000,1000)
    
    tokens_in_table = [token for token in page_tokens if postprocess.iob(token['bbox'], table_bbox) >= 0.5]
    
    # Determine the table cell structure from the objects
 
    table_structures, cells, confidence_score = postprocess.objects_to_cells(table, table_objects, tokens_in_table,
                                                                    structure_class_names,
                                                                    structure_class_thresholds)
    
    return table_structures, cells, confidence_score


def cells_to_adjacency_pair_list(cells, key='cell_text'):
    # Index the cells by their grid coordinates
    cell_nums_by_coordinates = dict()
    for cell_num, cell in enumerate(cells):
        for row_num in cell['row_nums']:
            for column_num in cell['column_nums']:
                cell_nums_by_coordinates[(row_num, column_num)] = cell_num

    # Count the number of unique rows and columns
    row_nums = set()
    column_nums = set()
    for cell in cells:
        for row_num in cell['row_nums']:
            row_nums.add(row_num)
        for column_num in cell['column_nums']:
            column_nums.add(column_num)
    num_rows = len(row_nums)
    num_columns = len(column_nums)

    # For each cell, determine its next neighbors
    # - For every row the cell occupies, what is the first cell to the right with text that
    #   also occupies that row
    # - For every column the cell occupies, what is the first cell below with text that
    #   also occupies that column
    adjacency_list = []
    adjacency_bboxes = []
    for cell1_num, cell1 in enumerate(cells):
        # Skip blank cells
        if cell1['cell_text'] == '':
            continue

        adjacent_cell_props = {}
        max_column = max(cell1['column_nums'])
        max_row = max(cell1['row_nums'])

        # For every column the cell occupies...
        for column_num in cell1['column_nums']:
            # Start from the next row and stop when we encounter a non-blank cell
            # This cell is considered adjacent
            for current_row in range(max_row+1, num_rows):
                try: # table might not be complete-grid. in case of wtw or incomplete table prediction
                    cell2_num = cell_nums_by_coordinates[(current_row, column_num)]
                    cell2 = cells[cell2_num]
                    if not cell2['cell_text'] == '':
                        adj_bbox = [(max(cell1['bbox'][0], cell2['bbox'][0])+min(cell1['bbox'][2], cell2['bbox'][2]))/2-3,
                                    cell1['bbox'][3],
                                    (max(cell1['bbox'][0], cell2['bbox'][0])+min(cell1['bbox'][2], cell2['bbox'][2]))/2+3,
                                    cell2['bbox'][1]]
                        adjacent_cell_props[cell2_num] = ('V', current_row - max_row - 1,
                                                        adj_bbox)
                        break
                except:
                    continue
        # For every row the cell occupies...
        for row_num in cell1['row_nums']:
            # Start from the next column and stop when we encounter a non-blank cell
            # This cell is considered adjacent
            for current_column in range(max_column+1, num_columns):
                try:
                    cell2_num = cell_nums_by_coordinates[(row_num, current_column)]
                    cell2 = cells[cell2_num]
                    if not cell2['cell_text'] == '':
                        adj_bbox = [cell1['bbox'][2],
                                    (max(cell1['bbox'][1], cell2['bbox'][1])+min(cell1['bbox'][3], cell2['bbox'][3]))/2-3,
                                    cell2['bbox'][0],
                                    (max(cell1['bbox'][1], cell2['bbox'][1])+min(cell1['bbox'][3], cell2['bbox'][3]))/2+3]
                        adjacent_cell_props[cell2_num] = ('H', current_column - max_column - 1,
                                                        adj_bbox)
                        break
                except:
                    continue

        for adjacent_cell_num, props in adjacent_cell_props.items():
            cell2 = cells[adjacent_cell_num]
            adjacency_list.append((cell1[key], cell2[key], props[0], props[1]))
            adjacency_bboxes.append(props[2])

    return adjacency_list, adjacency_bboxes


def cells_to_adjacency_pair_list_with_blanks(cells, key='cell_text'):
    # Index the cells by their grid coordinates
    cell_nums_by_coordinates = dict()
    for cell_num, cell in enumerate(cells):
        for row_num in cell['row_nums']:
            for column_num in cell['column_nums']:
                cell_nums_by_coordinates[(row_num, column_num)] = cell_num

    # Count the number of unique rows and columns
    row_nums = set()
    column_nums = set()
    for cell in cells:
        for row_num in cell['row_nums']:
            row_nums.add(row_num)
        for column_num in cell['column_nums']:
            column_nums.add(column_num)
    num_rows = len(row_nums)
    num_columns = len(column_nums)

    # For each cell, determine its next neighbors
    # - For every row the cell occupies, what is the next cell to the right
    # - For every column the cell occupies, what is the next cell below
    adjacency_list = []
    adjacency_bboxes = []
    for cell1_num, cell1 in enumerate(cells):
        adjacent_cell_props = {}
        max_column = max(cell1['column_nums'])
        max_row = max(cell1['row_nums'])

        # For every column the cell occupies...
        for column_num in cell1['column_nums']:
            # The cell in the next row is adjacent
            current_row = max_row + 1
            if current_row >= num_rows:
                continue
            try: # table might not be complete-grid. in case of wtw or incomplete table prediction
                cell2_num = cell_nums_by_coordinates[(current_row, column_num)]
                cell2 = cells[cell2_num]
                adj_bbox = [(max(cell1['bbox'][0], cell2['bbox'][0])+min(cell1['bbox'][2], cell2['bbox'][2]))/2-3,
                            cell1['bbox'][3],
                            (max(cell1['bbox'][0], cell2['bbox'][0])+min(cell1['bbox'][2], cell2['bbox'][2]))/2+3,
                            cell2['bbox'][1]]
                adjacent_cell_props[cell2_num] = ('V', current_row - max_row - 1,
                                                adj_bbox)
            except:
                continue

        # For every row the cell occupies...
        for row_num in cell1['row_nums']:
            # The cell in the next column is adjacent
            current_column = max_column + 1
            if current_column >= num_columns:
                continue
            try: # table might not be complete-grid. in case of wtw or incomplete table prediction
                cell2_num = cell_nums_by_coordinates[(row_num, current_column)]
                cell2 = cells[cell2_num]
                adj_bbox = [cell1['bbox'][2],
                            (max(cell1['bbox'][1], cell2['bbox'][1])+min(cell1['bbox'][3], cell2['bbox'][3]))/2-3,
                            cell2['bbox'][0],
                            (max(cell1['bbox'][1], cell2['bbox'][1])+min(cell1['bbox'][3], cell2['bbox'][3]))/2+3]
                adjacent_cell_props[cell2_num] = ('H', current_column - max_column - 1,
                                                adj_bbox)
            except:
                continue

        for adjacent_cell_num, props in adjacent_cell_props.items():
            cell2 = cells[adjacent_cell_num]
            adjacency_list.append((cell1[key], cell2[key], props[0], props[1]))
            adjacency_bboxes.append(props[2])

    return adjacency_list, adjacency_bboxes

def cells_to_adjacency_pair_list_with_blanks_coord(cells, key='cell_text'):
    # Index the cells by their grid coordinates
    cell_nums_by_coordinates = dict()
    for cell_num, cell in enumerate(cells):
        for row_num in cell['row_nums']:
            for column_num in cell['column_nums']:
                cell_nums_by_coordinates[(row_num, column_num)] = cell_num

    # Count the number of unique rows and columns
    row_nums = set()
    column_nums = set()
    for cell in cells:
        for row_num in cell['row_nums']:
            row_nums.add(row_num)
        for column_num in cell['column_nums']:
            column_nums.add(column_num)
    num_rows = len(row_nums)
    num_columns = len(column_nums)

    # For each cell, determine its next neighbors
    # - For every row the cell occupies, what is the next cell to the right
    # - For every column the cell occupies, what is the next cell below
    adjacency_list = []
    adjacency_bboxes = []
    for cell1_num, cell1 in enumerate(cells):
        adjacent_cell_props = {}
        max_column = max(cell1['column_nums'])
        max_row = max(cell1['row_nums'])

        # For every column the cell occupies...
        for column_num in cell1['column_nums']:
            # The cell in the next row is adjacent
            current_row = max_row + 1
            if current_row >= num_rows:
                continue
            cell2_num = cell_nums_by_coordinates[(current_row, column_num)]
            cell2 = cells[cell2_num]
            # adj_bbox = [(max(cell1['bbox'][0], cell2['bbox'][0])+min(cell1['bbox'][2], cell2['bbox'][2]))/2-3,
            #             cell1['bbox'][3],
            #             (max(cell1['bbox'][0], cell2['bbox'][0])+min(cell1['bbox'][2], cell2['bbox'][2]))/2+3,
            #             cell2['bbox'][1]]
            adjacent_cell_props[cell2_num] = ('V', current_row - max_row - 1,
                                              cell1['bbox'], cell2['bbox'])

        # For every row the cell occupies...
        for row_num in cell1['row_nums']:
            # The cell in the next column is adjacent
            current_column = max_column + 1
            if current_column >= num_columns:
                continue
            cell2_num = cell_nums_by_coordinates[(row_num, current_column)]
            cell2 = cells[cell2_num]
            # adj_bbox = [cell1['bbox'][2],
            #             (max(cell1['bbox'][1], cell2['bbox'][1])+min(cell1['bbox'][3], cell2['bbox'][3]))/2-3,
            #             cell2['bbox'][0],
            #             (max(cell1['bbox'][1], cell2['bbox'][1])+min(cell1['bbox'][3], cell2['bbox'][3]))/2+3]
            adjacent_cell_props[cell2_num] = ('H', current_column - max_column - 1,
                                              cell1['bbox'], cell2['bbox'])

        for adjacent_cell_num, props in adjacent_cell_props.items():
            cell2 = cells[adjacent_cell_num]
            adjacency_list.append((cell1['bbox'], cell2['bbox'], props[0], props[1]))
            adjacency_bboxes.append(props[2])

    return adjacency_list, adjacency_bboxes

def dar_con(true_adjacencies, pred_adjacencies):
    """
    Directed adjacency relations (DAR) metric, which uses exact match
    between adjacent cell text content.
    """

    true_c = Counter()
    true_c.update([elem for elem in true_adjacencies])

    pred_c = Counter()
    pred_c.update([elem for elem in pred_adjacencies])

    num_true_positives = (sum(true_c.values()) - sum((true_c - pred_c).values()))

    fscore, precision, recall = compute_fscore(num_true_positives,
                                               len(true_adjacencies),
                                               len(pred_adjacencies))

    return recall, precision, fscore

def dar_loc(true_adjacencies, pred_adjacencies, iou_th=0.5):
    """
    Directed adjacency relations (DAR) metric, which uses iou (0.5) based matching
    between adjacent cell boxes.
    """
    # pred_c = [elem + tuple([0]) for elem in pred_adjacencies] # [0] = not-checked
    num_true_positives = 0

    for t_c in true_adjacencies: # (cell1_box, cell2_box, direction, num_blanks)
        for p_c in pred_adjacencies:
            if t_c[2] != p_c[2] or t_c[3] != p_c[3]:
                continue
            # calculate iou
            t_cell1_box, t_cell2_box = t_c[0], t_c[1]
            p_cell1_box, p_cell2_box = p_c[0], p_c[1]
            cell1_iou = calculate_iou(t_cell1_box, p_cell1_box)
            cell2_iou = calculate_iou(t_cell2_box, p_cell2_box)

            if cell1_iou >= iou_th and cell2_iou >= iou_th:
                num_true_positives += 1


    fscore, precision, recall = compute_fscore(num_true_positives,
                                               len(true_adjacencies),
                                               len(pred_adjacencies))
    
    return recall, precision, fscore, num_true_positives, len(true_adjacencies), len(pred_adjacencies)

def dar_con_original(true_cells, pred_cells):
    """
    Original DAR metric, where blank cells are disregarded.
    """
    true_adjacencies, _ = cells_to_adjacency_pair_list(true_cells)
    pred_adjacencies, _ = cells_to_adjacency_pair_list(pred_cells)

    return dar_con(true_adjacencies, pred_adjacencies)


def dar_con_new(true_cells, pred_cells):
    """
    New version of DAR metric where blank cells count.
    """
    true_adjacencies, _ = cells_to_adjacency_pair_list_with_blanks(true_cells)
    pred_adjacencies, _ = cells_to_adjacency_pair_list_with_blanks(pred_cells)

    return dar_con(true_adjacencies, pred_adjacencies)

def dar_loc_original(true_cells, pred_cells):

    true_adjacencies, _ = cells_to_adjacency_pair_list(true_cells, key='bbox')
    pred_adjacencies, _ = cells_to_adjacency_pair_list(pred_cells, key='bbox')

    return dar_loc(true_adjacencies, pred_adjacencies, iou_th=0.5)

def dar_loc_new(true_cells, pred_cells):

    true_adjacencies, _ = cells_to_adjacency_pair_list_with_blanks(true_cells, key='bbox')
    pred_adjacencies, _ = cells_to_adjacency_pair_list_with_blanks(pred_cells, key='bbox')

    return dar_loc(true_adjacencies, pred_adjacencies)

def compute_metrics(mode, true_bboxes, true_labels, true_scores, true_cells,
                    pred_bboxes, pred_labels, pred_scores, pred_cells):
    """
    Compute the collection of table structure recognition metrics given
    the ground truth and predictions as input.

    - bboxes, labels, and scores are required to compute GriTS_RawLoc, which
      is GriTS_Loc but on unprocessed bounding boxes, compared with the dilated
      ground truth bounding boxes the model is trained on.
    - Otherwise, only true_cells and pred_cells are needed.
    """
    # true_bboxes, true_labels, true_scores, 
    # pred_bboxes, pred_labels, pred_scores, 
    metrics = {}

    if 'grits' in mode:
        # Compute grids/matrices for comparison
        true_relspan_grid = np.array(cells_to_relspan_grid(true_cells))
        true_bbox_grid = np.array(cells_to_grid(true_cells, key='bbox'))
        true_text_grid = np.array(cells_to_grid(true_cells, key='cell_text'), dtype=object)
        pred_relspan_grid = np.array(cells_to_relspan_grid(pred_cells))
        pred_bbox_grid = np.array(cells_to_grid(pred_cells, key='bbox'))
        pred_text_grid = np.array(cells_to_grid(pred_cells, key='cell_text'), dtype=object)

        # Compute GriTS_Top (topology)
        (metrics['grits_top'],
        metrics['grits_precision_top'],
        metrics['grits_recall_top'],
        metrics['grits_top_upper_bound']) = grits_top(true_relspan_grid,
                                                    pred_relspan_grid)

        # Compute GriTS_Loc (location)
        (metrics['grits_loc'],
        metrics['grits_precision_loc'],
        metrics['grits_recall_loc'],
        metrics['grits_loc_upper_bound']) = grits_loc(true_bbox_grid,
                                                    pred_bbox_grid)
    if 'grits' in mode and mode != 'grits-wocon':
        # Compute GriTS_Con (text content)
        (metrics['grits_con'],
        metrics['grits_precision_con'],
        metrics['grits_recall_con'],
        metrics['grits_con_upper_bound']) = grits_con(true_text_grid,
                                                    pred_text_grid)

        # Compute content accuracy
        metrics['acc_con'] = int(metrics['grits_con'] == 1)

    if mode == 'grits-all': # I didn't debug codes in this option.
        # Compute grids/matrices for comparison
        # *_cell_dilatedbbox_grid[i][j] = Rect() of the cell box (x,y,x,y) where the grid belongs to.
        true_cell_dilatedbbox_grid = np.array(cell_to_dilatedbbox_grid(true_cells))
        pred_cell_dilatedbbox_grid = np.array(cell_to_dilatedbbox_grid(pred_cells))
        # true_cell_dilatedbbox_grid = np.array(output_to_dilatedbbox_grid(true_bboxes, true_labels, true_scores))
        # pred_cell_dilatedbbox_grid = np.array(output_to_dilatedbbox_grid(pred_bboxes, pred_labels, pred_scores))

        # Compute GriTS_RawLoc (location using unprocessed bounding boxes)
        (metrics['grits_rawloc'],
        metrics['grits_precision_rawloc'],
        metrics['grits_recall_rawloc'],
        metrics['grits_rawloc_upper_bound']) = grits_loc(true_cell_dilatedbbox_grid,
                                                        pred_cell_dilatedbbox_grid)

    if mode == 'dar-all' or mode == 'dar-con':
        # Compute original DAR (directed adjacency relations) metric
        (metrics['dar_recall_con_original'], metrics['dar_precision_con_original'],
        metrics['dar_con_original']) = dar_con_original(true_cells, pred_cells)

        # Compute updated DAR (directed adjacency relations) metric
        (metrics['dar_recall_con'], metrics['dar_precision_con'],
        metrics['dar_con']) = dar_con_new(true_cells, pred_cells)
    
    elif mode == 'dar-all' or mode == 'dar-loc':
        # Compute updated DAR (directed adjacency relations) - loc metric
        (metrics['dar_recall_loc'], metrics['dar_precision_loc'],
        metrics['dar_loc'], metrics['true_positive'],
        metrics['true_adjacency'], metrics['pred_adjacency']) = dar_loc_original(true_cells, pred_cells) #dar_loc_new(true_cells, pred_cells) # 

    return metrics


def compute_statistics(structures, cells):
    statistics = {}
    statistics['num_rows'] = len(structures['rows'])
    statistics['num_columns'] = len(structures['columns'])
    statistics['num_cells'] = len(cells)
    statistics['num_spanning_cells'] = len([cell for cell in cells if len(cell['row_nums']) > 1
                                            or len(cell['column_nums']) > 1])
    header_rows = set()
    for cell in cells:
        if cell['header']:
            header_rows = header_rows.union(set(cell['row_nums']))
    statistics['num_header_rows'] = len(header_rows)
    row_heights = [float(row['bbox'][3]-row['bbox'][1]) for row in structures['rows']]
    if len(row_heights) >= 2:
        statistics['row_height_coefficient_of_variation'] = stat.stdev(row_heights) / stat.mean(row_heights)
    else:
        statistics['row_height_coefficient_of_variation'] = 0
    column_widths = [float(column['bbox'][2]-column['bbox'][0]) for column in structures['columns']]
    if len(column_widths) >= 2:
        statistics['column_width_coefficient_of_variation'] = stat.stdev(column_widths) / stat.mean(column_widths)
    else:
        statistics['column_width_coefficient_of_variation'] = 0

    return statistics


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def get_bbox_decorations(data_type, label):
    if label == 0:
        if data_type == 'detection':
            return 'brown', 0.05, 3, '//'
        else:
            return 'brown', 0, 3, None 
    elif label == 1:
        return 'red', 0.15, 2, None
    elif label == 2:
        return 'blue', 0.15, 2, None
    elif label == 3:
        return 'magenta', 0.2, 3, '//'
    elif label == 4:
        return 'cyan', 0.2, 4, '//'
    elif label == 5:
        return 'green', 0.2, 4, '\\\\'
    
    return 'gray', 0, 0, None


def compute_metrics_summary(sample_metrics, mode):
    """
    Print a formatted summary of the table structure recognition metrics
    averaged over all samples.
    """

    metrics_summary = {}
    
    if mode == 'grits-wocon':
        metric_names = ['grits_top', 'grits_loc']
    else:    
        metric_names = ['acc_con', 'grits_top', 'grits_con', 'grits_loc']
    if mode == 'grits-all':
        metric_names += ['grits_rawloc', 'dar_con_original', 'dar_con']

    simple_samples = [entry for entry in sample_metrics if entry['num_spanning_cells'] == 0]
    metrics_summary['simple'] = {'num_tables': len(simple_samples)}
    if len(simple_samples) > 0:
        for metric_name in metric_names:
            metrics_summary['simple'][metric_name] = np.mean([elem[metric_name] for elem in simple_samples])

    complex_samples = [entry for entry in sample_metrics if entry['num_spanning_cells'] > 0]
    metrics_summary['complex'] = {'num_tables': len(complex_samples)}
    if len(complex_samples) > 0:
        for metric_name in metric_names:
            metrics_summary['complex'][metric_name] = np.mean([elem[metric_name] for elem in complex_samples])

    metrics_summary['all'] = {'num_tables': len(sample_metrics)}
    if len(sample_metrics) > 0:
        for metric_name in metric_names:
            metrics_summary['all'][metric_name] = np.mean([elem[metric_name] for elem in sample_metrics])

    return metrics_summary


def print_metrics_line(name, metrics_dict, key, min_length=18):
    if len(name) < min_length:
        name = ' '*(min_length-len(name)) + name
    try:
        print("{}: {:.4f}".format(name, metrics_dict[key]))
    except:
        print("{}: --".format(name))


def print_metrics_summary(metrics_summary, all=False):
    """
    Print a formatted summary of the table structure recognition metrics
    averaged over all samples.
    """

    print('-' * 100)
    for table_type in ['simple', 'complex', 'all']:
        metrics = metrics_summary[table_type]
        print("Results on {} tables ({} total):".format(table_type, metrics['num_tables']))
        print_metrics_line("Accuracy_Con", metrics, 'acc_con')
        print_metrics_line("GriTS_Top", metrics, 'grits_top')
        print_metrics_line("GriTS_Con", metrics, 'grits_con')
        print_metrics_line("GriTS_Loc", metrics, 'grits_loc')
        if all:
            print_metrics_line("GriTS_RawLoc", metrics, 'grits_rawloc')
            print_metrics_line("DAR_Con (original)", metrics, 'dar_con_original')
            print_metrics_line("DAR_Con", metrics, 'dar_con')
        print('-' * 50)


def eval_tsr_sample(target, pred, mode, **kwargs):
    """ 
    target: {
             'cells': 
             'img_path': 
             }
    pred: {
            'cells': 
    }    
    """
    if 'grits' in mode:
        true_img_size = list(reversed(target['orig_size'].tolist()))
        true_bboxes = target['boxes']
        true_bboxes = [elem.tolist() for elem in rescale_bboxes(true_bboxes, true_img_size)]
        true_bboxes = [elem.tolist() for elem in target['boxes']]
        true_labels = target['labels'].tolist()
        true_scores = [1 for elem in true_labels]
        img_words_filepath = target["img_words_path"]
        with open(img_words_filepath, 'r') as f:
            true_page_tokens = json.load(f)

        true_table_structures, true_cells, _ = objects_to_cells(true_bboxes, true_labels, true_scores,
                                                                true_page_tokens, structure_class_names,
                                                                structure_class_thresholds, structure_class_map)
    else:
        true_cells = target["cells"]

    # m = pred_logits.softmax(-1).max(-1)
    # pred_labels = list(m.indices.detach().cpu().numpy())
    # pred_scores = list(m.values.detach().cpu().numpy())
    # pred_bboxes = [elem.tolist() for elem in pred_bboxes]
    # # pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, true_img_size)]
    # _, pred_cells, _ = objects_to_cells(pred_bboxes, pred_labels, pred_scores,
    #                                     true_page_tokens, structure_class_names,
    #                                     structure_class_thresholds, structure_class_map)
    pred_cells = pred["cells"]
    pred_bboxes, pred_labels, pred_scores = None, None, None

    for k, v in kwargs.items():
        if k == 'bboxes':
            pred_bboxes = v
        elif k == 'labels':
            pred_labels = v
        elif k == 'scores':
            pred_scores = v

    if 'grits' in mode:
        metrics = compute_metrics(mode, true_bboxes, true_labels, true_scores, true_cells,
                                    pred_bboxes, pred_labels, pred_scores, pred_cells)
        statistics = compute_statistics(true_table_structures, true_cells)
        metrics.update(statistics)
    else:
        metrics = compute_metrics(mode, None, None, None, true_cells, 
                                  None, None, None, pred_cells)

    metrics['id'] = target["img_path"].split('/')[-1].split('.')[0]

    return metrics


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

def eval_tsr_mp_sample(fname, target, pred):
    target_dict = utils.from_scitsr_to_gritsform(target, 'xxyy')
    pred_dict = utils.from_scitsr_to_gritsform(pred, 'xywh') # utils.format_scitsr(v)
    grits_score = eval_tsr_sample(target=target_dict, 
                                    pred=pred_dict,
                                    mode='dar-loc')
    grits_score['filename'] = fname
    return grits_score

def eval_tsr_mp(gts, preds, dar_score):
    # dar
    mpp.Pool.istarmap = istarmap
    pool = mp.Pool(processes=16)
    fnames = list(preds.keys())
    pred_list = [preds[fname] for fname in fnames]
    gt_list = [gts[fname] for fname in fnames]
    for res in tqdm.tqdm(pool.istarmap(eval_tsr_mp_sample, zip(fnames, gt_list, pred_list)), 
                         total=len(fnames), 
                         desc='Calculate DAR metric'):
        for k, v in dar_score.items():
            v['all'][res['filename']] = res[k]
            v['mean'] += res[k]
    
    pool.close()
    pool.join()

    return dar_score

def eval_tsr_grits_mp_sample(fname, target, pred, mode='grits-wocon'):
    target_dict = target # coco format
    pred_dict = utils.from_scitsr_to_gritsform(pred, 'xywh')
    grits_score = eval_tsr_sample(target=target_dict, 
                                    pred=pred_dict,
                                    mode=mode)
    grits_score['filename'] = fname
    return grits_score

def eval_tsr_grits_mp(gts, preds, mode):
    tsr_metrics = []
    mpp.Pool.istarmap = istarmap
    pool = mp.Pool(processes=16)
    fnames = list(preds.keys())
    pred_list = [preds[fname] for fname in fnames]
    gt_list = [gts[fname] for fname in fnames]
    
    for res in tqdm.tqdm(pool.istarmap(eval_tsr_grits_mp_sample, zip(fnames, gt_list, pred_list, repeat(mode))),
                        total=len(fnames),
                        desc='Calculate GriTS metric'):
        tsr_metrics += [res]
    
    pool.close()
    pool.join()
    return tsr_metrics


def visualize(args, target, pred_logits, pred_bboxes):
    img_filepath = target["img_path"]
    img_filename = img_filepath.split("/")[-1]

    bboxes_out_filename = img_filename.replace(".jpg", "_bboxes.jpg")
    bboxes_out_filepath = os.path.join(args.debug_save_dir, bboxes_out_filename)

    img = Image.open(img_filepath)
    img_size = img.size

    m = pred_logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())
    pred_scores = list(m.values.detach().cpu().numpy())
    pred_bboxes = pred_bboxes.detach().cpu()
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    fig,ax = plt.subplots(1)
    ax.imshow(img, interpolation='lanczos')

    for bbox, label, score in zip(pred_bboxes, pred_labels, pred_scores):
        if ((args.data_type == 'structure' and not label > 5)
            or (args.data_type == 'detection' and not label > 1)
            and score > 0.5):
            color, alpha, linewidth, hatch = get_bbox_decorations(args.data_type,
                                                                  label)
            # Fill
            rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1],
                                     linewidth=linewidth, alpha=alpha,
                                     edgecolor='none',facecolor=color,
                                     linestyle=None)
            ax.add_patch(rect)
            # Hatch
            rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1],
                                     linewidth=1, alpha=0.4,
                                     edgecolor=color,facecolor='none',
                                     linestyle='--',hatch=hatch)
            ax.add_patch(rect)
            # Edge
            rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1],
                                     linewidth=linewidth,
                                     edgecolor=color,facecolor='none',
                                     linestyle="--")
            ax.add_patch(rect) 

    fig.set_size_inches((15, 15))
    plt.axis('off')
    plt.savefig(bboxes_out_filepath, bbox_inches='tight', dpi=100)

    if args.data_type == 'structure':
        img_words_filepath = os.path.join(args.table_words_dir, img_filename.replace(".jpg", "_words.json"))
        cells_out_filename = img_filename.replace(".jpg", "_cells.jpg")
        cells_out_filepath = os.path.join(args.debug_save_dir, cells_out_filename)

        with open(img_words_filepath, 'r') as f:
            tokens = json.load(f)

        _, pred_cells, _ = objects_to_cells(pred_bboxes, pred_labels, pred_scores,
                                            tokens, structure_class_names,
                                            structure_class_thresholds, structure_class_map)

        fig,ax = plt.subplots(1)
        ax.imshow(img, interpolation='lanczos')

        for cell in pred_cells:
            bbox = cell['bbox']
            if cell['header']:
                alpha = 0.3
            else:
                alpha = 0.125
            rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, 
                                    edgecolor='none',facecolor="magenta", alpha=alpha)
            ax.add_patch(rect)
            rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, 
                                    edgecolor="magenta",facecolor='none',linestyle="--",
                                    alpha=0.08, hatch='///')
            ax.add_patch(rect)
            rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, 
                                    edgecolor="magenta",facecolor='none',linestyle="--")
            ax.add_patch(rect) 

        fig.set_size_inches((15, 15))
        plt.axis('off')
        plt.savefig(cells_out_filepath, bbox_inches='tight', dpi=100)

    plt.close('all')


@torch.no_grad()
def evaluate(args, model, criterion, postprocessors, data_loader, base_ds, device):
    st_time = datetime.now()
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    if args.data_type == "structure":
        tsr_metrics = []
        pred_logits_collection = []
        pred_bboxes_collection = []
        targets_collection = []
        
    num_batches = len(data_loader)
    print_every = max(args.eval_step, int(math.ceil(num_batches / 100)))
    batch_num = 0

    for samples, targets in metric_logger.log_every(data_loader, print_every, header):
        batch_num += 1
        samples = samples.to(device)
        for t in targets:
            for k, v in t.items():
                if not k == 'img_path':
                    t[k] = v.to(device)

        outputs = model(samples)

        if args.debug:
            for target, pred_logits, pred_boxes in zip(targets, outputs['pred_logits'], outputs['pred_boxes']):
                visualize(args, target, pred_logits, pred_boxes)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if args.data_type == "structure":
            pred_logits_collection += list(outputs['pred_logits'].detach().cpu())
            pred_bboxes_collection += list(outputs['pred_boxes'].detach().cpu())

            for target in targets:
                for k, v in target.items():
                    if not k == 'img_path':
                        target[k] = v.cpu()
                img_filepath = target["img_path"]
                img_filename = img_filepath.split("/")[-1]
                img_words_filepath = os.path.join(args.table_words_dir, img_filename.replace(".jpg", "_words.json"))
                target["img_words_path"] = img_words_filepath
            targets_collection += targets

            if batch_num % args.eval_step == 0 or batch_num == num_batches:
                arguments = zip(targets_collection, pred_logits_collection, pred_bboxes_collection,
                                repeat(args.mode))
                with multiprocessing.Pool(args.eval_pool_size) as pool:
                    metrics = pool.starmap_async(eval_tsr_sample, arguments).get()
                tsr_metrics += metrics
                pred_logits_collection = []
                pred_bboxes_collection = []
                targets_collection = []

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()

    if args.data_type == "structure":
        # Save sample-level metrics for more analysis
        if len(args.metrics_save_filepath) > 0:
            with open(args.metrics_save_filepath, 'w') as outfile:
                json.dump(tsr_metrics, outfile)

        # Compute metrics averaged over all samples
        metrics_summary = compute_metrics_summary(tsr_metrics, args.mode)

        # Print summary of metrics
        print_metrics_summary(metrics_summary)

    print("Total time taken for {} samples: {}".format(len(base_ds), datetime.now() - st_time))

    return stats, coco_evaluator


def eval_coco(args, model, criterion, postprocessors, data_loader_test, dataset_test, device):
    """
    Use this function to do COCO evaluation. Default implementation runs it on
    the test set.
    """
    pubmed_stats, coco_evaluator = evaluate(args, model, criterion, postprocessors,
                                            data_loader_test, dataset_test,
                                            device)
    print("COCO metrics summary: AP50: {:.3f}, AP75: {:.3f}, AP: {:.3f}, AR: {:.3f}".format(
        pubmed_stats['coco_eval_bbox'][1], pubmed_stats['coco_eval_bbox'][2],
        pubmed_stats['coco_eval_bbox'][0], pubmed_stats['coco_eval_bbox'][8]))


def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    # box: x1,y1,x2,y2
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    union_area = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection_area

    # Calculate IoU
    iou = max(intersection_area / union_area, 0) # to prevent NaN

    return iou