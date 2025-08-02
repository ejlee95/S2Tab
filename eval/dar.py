"""
Copyright (C) 2021 Microsoft Corporation
"""

from collections import Counter
from textwrap import indent

def compute_fscore(num_true_positives, num_true, num_positives):
    """
    Compute the f-score or f-measure for a collection of predictions.

    Conventions:
    - precision is 1 when there are no predicted instances
    - recall is 1 when there are no true instances
    - fscore is 0 when recall or precision is 0
    """
    if num_positives > 0:
        precision = num_true_positives / num_positives
    else:
        precision = 1
    if num_true > 0:
        recall = num_true_positives / num_true
    else:
        recall = 1
        
    if precision + recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0

    return fscore, precision, recall 

def cells_to_adjacency_pair_list(cells, key='cell_text'):
    """
    cells: {'bbox': [x0,y0,x1,y1], # maybe
            'column_nums': [column_num], 
            'row_nums': [row_num], 
            'header': bool,
            'spans': [?] # no need
            'cell_text': str,
            }
    """
    ## TODO - cell_box mode!
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

        # For every row the cell occupies...
        for row_num in cell1['row_nums']:
            # Start from the next column and stop when we encounter a non-blank cell
            # This cell is considered adjacent
            for current_column in range(max_column+1, num_columns):
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

        for adjacent_cell_num, props in adjacent_cell_props.items():
            cell2 = cells[adjacent_cell_num]
            adjacency_list.append((cell1['cell_text'], cell2['cell_text'], props[0], props[1]))
            adjacency_bboxes.append(props[2])

    return adjacency_list, adjacency_bboxes


def cells_to_adjacency_pair_list_with_blanks(cells, key='cell_text'):
    ## TODO - cell_box mode
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
            adj_bbox = [(max(cell1['bbox'][0], cell2['bbox'][0])+min(cell1['bbox'][2], cell2['bbox'][2]))/2-3,
                        cell1['bbox'][3],
                        (max(cell1['bbox'][0], cell2['bbox'][0])+min(cell1['bbox'][2], cell2['bbox'][2]))/2+3,
                        cell2['bbox'][1]]
            adjacent_cell_props[cell2_num] = ('V', current_row - max_row - 1,
                                              adj_bbox)

        # For every row the cell occupies...
        for row_num in cell1['row_nums']:
            # The cell in the next column is adjacent
            current_column = max_column + 1
            if current_column >= num_columns:
                continue
            cell2_num = cell_nums_by_coordinates[(row_num, current_column)]
            cell2 = cells[cell2_num]
            adj_bbox = [cell1['bbox'][2],
                        (max(cell1['bbox'][1], cell2['bbox'][1])+min(cell1['bbox'][3], cell2['bbox'][3]))/2-3,
                        cell2['bbox'][0],
                        (max(cell1['bbox'][1], cell2['bbox'][1])+min(cell1['bbox'][3], cell2['bbox'][3]))/2+3]
            adjacent_cell_props[cell2_num] = ('H', current_column - max_column - 1,
                                              adj_bbox)

        for adjacent_cell_num, props in adjacent_cell_props.items():
            cell2 = cells[adjacent_cell_num]
            adjacency_list.append((cell1['cell_text'], cell2['cell_text'], props[0], props[1]))
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

    return num_true_positives, len(true_adjacencies), len(pred_adjacencies)

    # fscore, precision, recall = compute_fscore(num_true_positives,
    #                                            len(true_adjacencies),
    #                                            len(pred_adjacencies))

    # return recall, precision, fscore


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

def convert_format(ori_cells):
    cells = []
    ## TODO

if __name__ == "__main__":
    import json
    from pathlib import Path

    rootdir = Path('./eval')
    truedir = rootdir / 'scitsr'
    preddir = rootdir / 'pred'
    fnames = preddir.glob('*.json')

    num_true_positives, num_true, num_positives = 0, 0, 0
    
    for fname in fnames:
        with open(fname, 'r') as f:
            pred_cells = json.load(f)['cells']
        with open(truedir / fname.name, 'r') as f:
            true_cells = json.load(f)['cells']

        TP, T, P = dar_con_original(true_cells, pred_cells)

        num_true_positives += TP
        num_true += T
        num_positives += P

    
    fscore, precision, recall = compute_fscore(num_true_positives,
                                               num_true,
                                               num_positives)
    with open(rootdir / 'metric.json', 'w') as f:
        json.dump({'f1': fscore, 'p': precision, 'r': recall}, f, indent=2)