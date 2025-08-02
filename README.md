<div align="center">
  <h2>Single-stage table structure recognition approach via efficient sequence modeling</h2>
</div>

Official source code for the paper **"Single-stage table structure recognition approach via efficient sequence modeling"**(_Pattern Recognition Letters_ accepted). This repository provides a table recognition method using a novel sequence-format table representation.

## Overview
S2Tab predicts table structure (named Cell-Code) as a sequence from the input table image.

## Getting Started

### 1. Installation
Entire code is based on [DETR](https://github.com/facebookresearch/detr).
We use [huggingface/transformers](https://huggingface.co/docs/transformers/index) library.

Using conda environment,
```
conda create -n s2tab python=3.11 -y
conda activate s2tab
bash ./install.sh
```

### 2. Dataset Preparation
Input data directory constructed as follows:
- dataset_file = coco_content, coco_cell, or coco_instance
```
input-path/
  annotations/ # annotations directory
    instances_train2017.json
    instances_val2017.json
    instances_test2017.json
  train2017/ # training images directory
  val2017/ # validation images directory
  test2017/ # test images directory
```
Refer to sample.zip file for example.

- For inference, only test2017 folder is necessary.


## Usage

### 1. Training
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
--config configs/config_ftn.yaml --output_dir ./data/saved/my_checkpoint
```
We also provide an example in exec.sh

#### arguments
- coco_path: input dataset directory
- max_seq_len: max length of decoder output sequence (we use 512 for training)
- output_dir: directory where the checkpoints are saved in
- log_dir: tensorboard log directory (inside output_dir)

More detailed arguments can be found using "python main.py --help"


### 2. Test

```
CUDA_VISIBLE_DEVICES=0 python main.py --eval --max_seq_len 720 \
--config configs/ftn_base_text.yaml --output_dir ./data/output/my_checkpoint \
--resume ./saved/my_checkpoint/checkpoint.pth --pp 1 --dataset_mode fintabnet
```
We also provide an example in exec.sh

### 3. Checkpoints
We provide our trained model:
1. S2Tab trained on FinTabNet training data [Link](https://drive.google.com/file/d/1zmfbtVL4jqTHeEKBPF2pCA7KLDqWxgli/view?usp=sharing).
2. S2Tab trained on PubTabNet training data [Link](https://drive.google.com/file/d/1IEpqaJ0xYkmVL9WHkZUEbxVVuCLKgm0e/view?usp=sharing).


## License
S2Tab is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
TBU