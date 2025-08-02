#!/bin/bash

## TRAIN - sample
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/ftn_base_text.yaml \
--coco_path ./data/input/sample --output_dir ./data/saved/sample --log_dir log

## TEST - sample
CUDA_VISIBLE_DEVICES=0 python main.py --eval \
--config ./configs/ftn_base_text.yaml --pp 1 \
--coco_path ./data/input/sample \
--gtpath ./data/input/sample/FinTabNet_1.0.0_table_cell_test.jsonl \
--output_dir ./data/output/sample/ftn --resume ./data/saved/ftn/checkpoint_ftn.pth

CUDA_VISIBLE_DEVICES=0 python main.py --eval \
--config ./configs/ptn_base_text.yaml --pp 1 \
--coco_path ./data/input/sample \
--gtpath ./data/input/sample/FinTabNet_1.0.0_table_cell_test.jsonl \
--output_dir ./data/output/sample/ptn --resume ./data/saved/ptn/checkpoint_ptn.pth
