# -----------------------------------------------------------------------
# S2Tab official code : main.py
# -----------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -----------------------------------------------------------------------

import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path
from copy import deepcopy
from omegaconf import OmegaConf

import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(4)
from torch.utils.data import DataLoader, DistributedSampler

from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model
import util.misc as utils
from engine import train_one_epoch, evaluate

def get_args_parser():
    parser = argparse.ArgumentParser('arguments for S2Tab: table recognition network', add_help=False)

    parser.add_argument('--config', type=str, help="config yaml file path.")

    # Training parameters
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--val_batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--use_amp', action='store_true', help='use amp')

    # Model parameters
    parser.add_argument('--max_seq_len', type=int,
                        help="Max sequence length of decoder's output")

    # Dataset parameters
    parser.add_argument('--coco_path', type=str, help='input data directory')
    parser.add_argument('--dataset_file', type=str, help='input data type')
    parser.add_argument('--image_path', type=str, help='input image directory (Optional)')
    parser.add_argument('--testdir', choices=('test', 'val'),
                        help='test directory name (except 2017) - "test" or "val", default: test')
    parser.add_argument('--gtpath', help='gt jsonl file path, only used for TEDS-struc evaluation.')

    # Execution parameters
    parser.add_argument('--output_dir', default=None,
                        help="Path where to save (model or evaluation results), empty for no saving, default: output")
    parser.add_argument('--log_dir',
                        help="Path where to save tensorboard log, empty for no saving, default: None")
    parser.add_argument('--device', default='cuda',
                        help="Device to use for training / testing, default: cuda")
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume',
                        help="Path to resume from (checkpoint), default: ''")
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--dataset_mode', \
                            help='dataset mode. yields different structure output (during inference). \
                                    among {icdar2019, pubtabnet, fintabnet, scitsr, wtw-dar, wtw-teds}, default: pubtabnet')
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--pp', type=int,
                        help='post-processing mode. 0: no structure processing, 1: structure processing')
    parser.add_argument('--draw_attn', action='store_true',
                        help='draw attention flag')
    parser.add_argument('--no_eval', action='store_true', default=False)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--apply_gt_ocr', action='store_true', help='apply ground truth content matching')
    parser.add_argument('--apply_as', action='store_true', help='apply attention sharpening during test')

    # Distributed training parameters
    parser.add_argument('--world_size', type=int,
                        help='Number of distributed processes')
    parser.add_argument('--dist_url',
                        help="Url used to set up distributed training")

    return parser

def get_config(args):
    fname = args.config
    if fname is None:
        return args
    cfg = OmegaConf.load(fname)
    cfg = utils.overload_cfg(cfg, args)

    return cfg

def main(args):
    # Set distributed training
    utils.init_distributed_mode(args)
    if args.exec.output_dir:
        output_dir = Path(args.exec.output_dir)
        OmegaConf.save(config=cfg, f=(output_dir / 'config.yaml'))
    else:
        output_dir = None

    device = torch.device(args.exec.device)

    # Fix the seed for reproducibility
    seed = args.exec.seed + utils.get_rank()
    torch.manual_seed(seed)
    if args.exec.device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if args.exec.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Tensorboard
    if 'AMLT_OUTPUT_DIR' in os.environ:
        args.exec.log_dir = os.environ['AMLT_OUTPUT_DIR']
        print(f'update args.exec.log_dir to {args.exec.log_dir}')
        log_dir = Path(args.exec.log_dir)
    elif args.exec.output_dir and args.exec.log_dir:
        log_dir = output_dir / args.exec.log_dir
    elif args.exec.log_dir:
        raise ValueError(f'if you want to save tensorboard log in {args.exec.log_dir}, \
                            you should have args.exec.output_dir also.')
    if utils.is_main_process() and args.exec.log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir = log_dir)
    else:
        log_writer = None

    # Build model/loss
    model, criterion, postprocessors = build_model(args, args.vocab)
    model.to(device)
    criterion.to(device)

    model_without_ddp = model
    if args.exec.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.exec.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of parameters: {:,}'.format(n_parameters))
    if args.exec.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(str(model_without_ddp))
            f.write("number of parameters: {:,}\n".format(n_parameters))

    # Build dataset/dataloader
    if not args.exec.eval:
        dataset_train = build_dataset(image_set='train', args=args, vocab=args.vocab, model=model_without_ddp)
        args_val = deepcopy(args)
        args_val.dataset.dataset_file = 'coco_cell' if ('cell' in args_val.dataset.seq_version or 'both' in args_val.dataset.seq_version) else 'coco_content'
        dataset_val = build_dataset(image_set='val', args=args_val, vocab=args.vocab, model=model_without_ddp)
    else:
        if args.dataset.dataset_file == 'instance':
            args.dataset.dataset_file = 'coco_cell' if ('cell' in args.dataset.seq_version or 'both' in args.dataset.seq_version) else 'coco_content'
            print(f'Changed args.dataset.dataset_file from "instances" to "{args.dataset.dataset_file}"')
        dataset_val = build_dataset(image_set=args.dataset.testdir, args=args, vocab=args.vocab, model=model_without_ddp)

    if args.exec.distributed:
        if not args.exec.eval:
            sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        if not args.exec.eval:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    pw = True if args.exec.num_workers > 0 else False
    if not args.exec.eval:
        if args.dataset.fix_batch_image_size:
            collate_cls_tr = utils.Collate((args.train.image_max, args.train.image_min))
        else:
            collate_cls_tr = utils.Collate(None)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.train.batch_size, drop_last=True
        )
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                collate_fn=collate_cls_tr.collate_fn, num_workers=args.exec.num_workers,
                                pin_memory=False, persistent_workers=pw, worker_init_fn=utils.worker_init_fn)
    if args.exec.eval:
        bs = args.test.batch_size
    else:
        bs = args.train.val_batch_size
    if args.dataset.fix_batch_image_size:
        collate_cls_vl = utils.Collate((args.test.image_max, args.test.image_min))
    else:
        collate_cls_vl = utils.Collate(None)
    data_loader_val = DataLoader(dataset_val, bs, sampler=sampler_val,
                    drop_last=False, collate_fn=collate_cls_vl.collate_fn, num_workers=args.exec.num_workers,
                    pin_memory=False, persistent_workers=pw, worker_init_fn=utils.worker_init_fn)

    base_ds = get_coco_api_from_dataset(dataset_val)

    # Training hyper-param
    if not args.exec.eval:
        # Training hyper-param
        total_batch_size = args.train.batch_size * utils.get_world_size()
        dataset_size_train = len(dataset_train)
        steps_per_epoch = dataset_size_train // total_batch_size
    else:
        steps_per_epoch = 1000 # dontcare

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters()
                        if "projection" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                            if "projection" in n and p.requires_grad],
            "lr": args.train.lr_backbone
        },
    ]
    if hasattr(criterion, "t"):
        # Add "t" and "b" to param_dicts
        param_dicts.append({"params": [criterion.t, criterion.b]})

    # Build optimizer
    optimizer = torch.optim.AdamW(param_dicts, lr=args.train.lr, weight_decay=args.train.weight_decay)

    if args.train.lr_decay == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.train.lr_drop, args.train.gamma)
    elif args.train.lr_decay == 'cosineannealing':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train.T_max, args.train.eta_min)
    elif args.train.lr_decay == 'cosineannealingwarm':
        T_max = steps_per_epoch * args.train.T_max
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_max, args.train.T_mult, args.train.eta_min)
    elif args.train.lr_decay == 'reduceonplateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', args.train.gamma, args.train.patience, threshold=1e-4)
    elif args.train.lr_decay == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12, 17], gamma=args.train.gamma) # only 20-epochs setting
    elif args.train.lr_decay == 'onecycle':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.train.lr, steps_per_epoch=steps_per_epoch, epochs=args.train.epochs, pct_start=0.2, cycle_momentum=True)


    scaler = torch.cuda.amp.GradScaler(enabled=args.train.use_amp)
    # Resume checkpoint if need
    if args.exec.resume:
        if args.exec.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                                args.exec.resume, map_location='cpu', check_hash=True)
            removed_keys = ('decoder', 'class_embed', 'bbox_embed', 'query_embed')
            removed_list = []
            for k in checkpoint['model'].keys():
                if sum([x in k for x in removed_keys]):
                    removed_list.append(k)
            for k in removed_list:
                del checkpoint['model'][k]
        elif not args.exec.resume.endswith('.pth'):
            candidates = ('DETR', 'detr_resnet101', 'detr_resnet101_dc5', 'detr_resnet50', 'detr_resnet50_dc5')
            if not args.exec.resume in candidates:
                raise ValueError(f'not supported resume path: {args.exec.resume}.')
            repo = 'facebookresearch/detr:main'
            checkpoint = torch.hub.load(repo, args.exec.resume, pretrained=True)
        else:
            checkpoint = torch.load(args.exec.resume, map_location=device)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if args.exec.eval:
            print('Load model: ', checkpoint['epoch'])
            if args.exec.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write('Load model: %d' % checkpoint['epoch'])
        if not args.exec.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if 'epoch' in checkpoint:
                args.exec.start_epoch = checkpoint['epoch'] + 1
        if not args.exec.eval and 'criterion' in checkpoint and hasattr(criterion, "t"):
            criterion.load_state_dict(checkpoint['criterion'])

        if args.train.use_amp and 'scaler' in checkpoint:
            scaler = checkpoint['scaler']

    if args.exec.eval:
        test_stats, coco_evaluator = evaluate(model, postprocessors,
                                                data_loader_val, base_ds, device, output_dir,
                                                draw=True, dataset_mode=args.dataset.dataset_mode,
                                                pp=args.exec.pp, draw_attention=args.exec.draw_attn, gtpath=args.dataset.gtpath,
                                                apply_ocr='gt' if args.exec.apply_gt_ocr else None,
                                                content=True if 'content' in args.dataset.seq_version else False)
        if dataset_val.no_eval and utils.is_main_process():
            annotation_path = Path(args.dataset.coco_path) / "annotations" / 'instances_test2017.json'
            dataset_val.delete_dummy_file(path=annotation_path)

        if args.exec.output_dir and coco_evaluator is not None:
            if 'bbox' in coco_evaluator.coco_eval:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
                if utils.is_main_process():
                    test_stats['coco_eval_bbox'] = utils.eval_log(test_stats['coco_eval_bbox'], 'bbox')
                    with open(output_dir / "test_stats.json", "w") as f:
                        json.dump(test_stats, f, indent=2)
        return

    # Training
    print("Start training")
    start_time = time.time()
    prev_score = 0.
    if args.exec.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(f"steps per epoch: {steps_per_epoch}" + "\n")
    for epoch in range(args.exec.start_epoch, args.train.epochs):
        if args.exec.distributed:
            sampler_train.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * steps_per_epoch)
        lr_schedule_scheme = [0, -1]
        if args.train.lr_decay == 'onecycle' or 'cosineannealing' in args.train.lr_decay:
            lr_schedule_scheme = [steps_per_epoch, 0]
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            log_writer, args.train.clip_max_norm,
            scaler, args.train.use_amp, lr_scheduler, lr_schedule_scheme)

        if not args.exec.no_eval:
            test_stats, coco_evaluator = evaluate(model, postprocessors,
                                                    data_loader_val, base_ds, device,
                                                    output_dir=None,
                                                    draw=False)

            cur_score = test_stats['coco_eval_bbox'][0]
            if log_writer is not None:
                log_writer.update(val_acc=cur_score, head="acc")
        else:
            cur_score = 0
            test_stats = {}

        if args.train.lr_decay == 'reduceonplateau':
            lr_scheduler.step(cur_score)
        elif args.train.lr_decay == 'onecycle' or 'cosineannealing' in args.train.lr_decay:
            pass
        else:
            lr_scheduler.step()

        if args.exec.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before every save_step
            if (epoch + 1) % args.exec.save_step == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

            if prev_score < cur_score:
                checkpoint_paths.append(output_dir / f'best/checkpoint{epoch:04}_{cur_score:.4f}.pth')
                prev_score = cur_score

            for checkpoint_path in checkpoint_paths:
                model_dict = model_without_ddp.state_dict()
                del model_dict['transformer.decoder.embeddings.position_ids']
                utils.save_on_master({
                    'model': model_dict, #model_without_ddp.state_dict(),
                    'criterion': criterion.state_dict(),
                    'optimizer': optimizer.state_dict() if utils.is_main_process() else None,
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'scaler': scaler,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch,
                    'n_parameters': n_parameters}

        if args.exec.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if log_writer is not None:
                log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')
    if args.exec.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(f'Training time {total_time_str}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('S2Tab training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    cfg = get_config(args)

    if cfg.exec.output_dir == 'None' or cfg.exec.output_dir == 'none':
        cfg.exec.output_dir = None
    if cfg.dataset.gtpath == 'None' or cfg.dataset.gtpath == 'none':
        cfg.dataset.gtpath = None
    if 'image_path' in cfg.dataset and (cfg.dataset.image_path == 'none' or cfg.dataset.image_path == 'None'):
        cfg.dataset.image_path = None
    if cfg.exec.output_dir:
        Path(cfg.exec.output_dir).mkdir(parents=True, exist_ok=True)
        if not cfg.exec.eval:
            (Path(cfg.exec.output_dir) / 'best').mkdir(parents=True, exist_ok=True)

    main(cfg)