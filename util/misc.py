# -----------------------------------------------------------------------
# S2Tab official code : util/misc.py
# -----------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -----------------------------------------------------------------------

import pickle
import os
from collections import defaultdict, deque, OrderedDict
from typing import Optional, List
import time, datetime
import argparse
import collections
from pathlib import Path
from copy import deepcopy
from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
import psutil
from omegaconf import OmegaConf

import torch
import torch.distributed as dist
from torch import Tensor
import numpy as np

from torch.utils.tensorboard import SummaryWriter

# Data
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask
    
    def to(self, device, non_blocking=False):
        # type: (device, non_blocking) -> NestedTensor
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)
    
    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for i, item in enumerate(sublist):
            maxes[i] = max(maxes[i], item)
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[Tensor], defined_size=None, fill=0):
    if tensor_list[0].ndim == 3: # image
        if defined_size is None:
            max_size = _max_by_axis([list(img.shape) for img in tensor_list])
            batch_shape = [len(tensor_list)] + max_size
        else:
            batch_shape = [len(tensor_list)] + defined_size
        bs, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        padded_tensor = torch.ones(batch_shape, dtype=dtype, device=device) * fill
        mask = torch.zeros((bs, h, w), dtype=torch.bool, device=device) # False = need pad
        for img, pad_img, m in zip(tensor_list, padded_tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]].fill_(True)
    elif tensor_list[0].ndim == 2: # seq elem
        max_size = _max_by_axis([list(seq.shape) for seq in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        bs, n, c = batch_shape # ASSUME batch_first
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        padded_tensor = torch.ones(batch_shape, dtype=dtype, device=device) * fill
        mask = torch.zeros((bs, n), dtype=torch.bool, device=device) # False = need pad
        for seq, pad_seq, m in zip(tensor_list, padded_tensor, mask):
            pad_seq[:seq.shape[0], :seq.shape[1]] = seq
            m[:seq.shape[0]].fill_(True)
    elif tensor_list[0].ndim == 1:
        if defined_size is None:
            max_size = _max_by_axis([list(elem.shape) for elem in tensor_list])
            batch_shape = [len(tensor_list)] + max_size
        else:
            batch_shape = [len(tensor_list)] + [defined_size]
        bs, n = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        padded_tensor = torch.ones(batch_shape, dtype=dtype, device=device) * fill
        mask = torch.zeros((bs, n), dtype=torch.bool, device=device)
        for elem, pad_elem, m in zip(tensor_list, padded_tensor, mask):
            pad_elem[:elem.shape[0]] = elem
            m[:elem.shape[0]].fill_(True)
    else:
        raise ValueError("not supported tensor")
    return NestedTensor(padded_tensor, mask)

class Collate(object):
    def __init__(self, image_size=None):
        """
        image_size = (image_max, image_min)
        """
        super().__init__()
        if image_size is None:
            self.defined_size = None
        else:
            self.defined_size = [3, image_size[0], image_size[0]]

    def collate_fn(self, batch):
        batch = list(zip(*batch))
        batch[0] = nested_tensor_from_tensor_list(batch[0], defined_size=self.defined_size)
        return tuple(batch)

def get_least_loaded_cores():
    cpu_percentages = psutil.cpu_percent(percpu=True, interval=0.1)
    # num_available_cores = len(available_cores)
    return sorted(available_cores, key=lambda x: cpu_percentages[x]) # [num_available_cores//2:]

# List of available cores
available_cores = psutil.Process().cpu_affinity()


def worker_init_fn(worker_id):
    torch.multiprocessing.set_sharing_strategy('file_system')
    # Cores of Low-usage
    least_loaded_cores = get_least_loaded_cores()
    # 사용 가능한 코어 중에서 사용량이 가장 낮은 core 선택
    available_least_loaded_cores = [core for core in least_loaded_cores if core in available_cores]
    if not available_least_loaded_cores:
        raise ValueError("No available cores with low usage")

    core_to_use = available_least_loaded_cores[worker_id % len(available_least_loaded_cores)]

    # 해당 worker의 프로세스에 CPU affinity 설정
    current_process = psutil.Process()
    current_process.cpu_affinity([core_to_use])

# Logger
class SmoothedValue(object):
    """ Track a series of values and provide access to smoothed values
    over a window or the global series average. """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
    
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
    
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float32, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
    
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]
    
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
    
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                f"{name}: {meter}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        # iterable = data_loader
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def add_graph(self, model, images=None):
        self.writer.add_graph(model, images)

    def add_histogram(self, head='vector', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            assert isinstance(v, (torch.Tensor, np.array, str))
            self.writer.add_histogram(head + "/" + k, v, self.step if step is None else step)

    def add_image(self, img, tag='input', step=None):
        self.writer.add_image(tag, img, global_step=self.step if step is None else step)

    def add_embedding(self, v, label=None, tag='embedding', step=None):
        self.writer.add_embedding(v, metadata=label, global_step=self.step if step is None else step, tag=tag)

    def flush(self):
        self.writer.flush()

# Miscellaneous for distributed training

def setup_for_distributed(is_master):
    """
    Disables printing when not in master process (rank 0)
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    
    __builtin__.print = print
    
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.exec.rank = int(os.environ['RANK'])
        args.exec.world_size = int(os.environ['WORLD_SIZE'])
        args.exec.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.exec.distributed = False
        return
    
    args.exec.distributed = True

    torch.cuda.set_device(args.exec.gpu)
    args.exec.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.exec.rank, args.exec.dist_url), flush=True)
    dist.init_process_group(backend=args.exec.dist_backend, init_method=args.exec.dist_url,
                            world_size=args.exec.world_size, rank=args.exec.rank)
    dist.barrier()
    setup_for_distributed(args.exec.rank == 0)

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool) : whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that tehy are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}

    return reduced_dict

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    
    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to('cuda')

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device='cuda')
    size_list = [torch.tensor([0], device='cuda') for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device='cuda'))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device='cuda')
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

# Prepare to load dit pretrained model
def adjust_checkpoint(load_path, save_path, config):
    """ 
    load_path : .pth path
    save_path : dir 
    """
    if '224' in save_path:
        return load_path
    if (Path(save_path) / 'pytorch_model.bin').exists():
        return save_path

    checkpoint_model = torch.load(Path(load_path) / 'dit-base.pth')

    # interpolate position embedding
    if 'embeddings.position_embeddings' in checkpoint_model.state_dict():    
        image_size = config.image_size if isinstance(config.image_size, collections.abc.Iterable) \
                                            else (config.image_size, config.image_size)
        patch_size = config.patch_size if isinstance(config.patch_size, collections.abc.Iterable) \
                                            else (config.patch_size, config.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        pos_embed_checkpoint = checkpoint_model.embeddings.position_embeddings
        embedding_size = pos_embed_checkpoint.shape[-1]
        # num_patches = new_num_patches #model.patch_embed.num_patches
        num_extra_tokens = num_patches + 1 - num_patches # +1 means only 1 extra token ('CLS' token) at the beginning
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model.embeddings.position_embeddings.data = new_pos_embed

            # config update
            checkpoint_model.config = config

            # save modified pretrained model
            checkpoint_model.save_pretrained(save_path)
            print(f"Newly updated backbone model saved in {save_path}.")
        
        return save_path

# others
@torch.no_grad()
def accuracy(output, target, reduction='sum'):
    """
    output: logits of shape (batch, num_classes, N)
    target: target indices of shape (batch, N)
    """
    pred = torch.argmax(output, dim=1)
    if reduction == 'sum':
        acc = (pred == target).sum()
    elif reduction == 'mean':
        acc = (pred == target).sum() / pred.shape[0] / pred.shape[-1]
    else:
        raise ValueError(f'Do not support the reduction type: {reduction}')

    return acc

def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ('true', 't', 'y', '1'):
        return True
    elif v.lower() in ('false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value required.')
 
# bucketize..
def de_bucketize(indices, bin, original_size):
    """
    indices: (bucketize, right=False), type: torch.int64, shape: (bs, num)
    bin: [0., s, 2*s, ..., 1.] (float32), shape: (bin_num)
    original_size: denormalized size (int), shape: (bs)
    return original integer values (may have minor error) 
    bin에 속한 (original) integer가 여러개이면 범위 내에서 제일 작은 값으로 & 
    bin에 속한 (original) integer가 없으면 (wrong prediction) bin의 floor된 integer로
    """
    bs, num = indices.shape
    # bin[ind-1] < result integer value <= bin[ind]
    trivial_left = (-1) * (indices<=0) # including (real) zero points and noise points
    left = bin[indices-1] * original_size[:, None]
    left = trivial_left.to(indices.device) * (indices==0) + left * (indices!=0)
    right = bin[indices] * original_size[:, None]

    original_integers = []
    for i, s in enumerate(original_size):
        integer = torch.arange(0, s+1, 1, dtype=torch.float32).to(indices.device) # (original_size+1)
        integer = torch.stack([integer] * num, dim=0) # (num, original_size+1) : 0,1,...,original_size

        left_mask = torch.where(left[i, :, None] < integer, torch.ones_like(integer), torch.zeros_like(integer)) # (num, original_size+1)
        right_mask = torch.where(right[i, :, None] >= integer, torch.ones_like(integer), torch.zeros_like(integer)) # (num, original_size+1)
        # mask = (left_mask * right_mask).to(torch.bool) # (num, original_size+1)
        original_integer = torch.minimum((1 - left_mask).sum(-1), right_mask.sum(-1) - 1)
        
        original_integers.append(original_integer.to(torch.int64))

    original_integers = nested_tensor_from_tensor_list(original_integers)
    original_integers.mask = torch.where(indices >= 0, torch.zeros_like(indices).to(torch.bool), torch.ones_like(indices).to(torch.bool))

    return original_integers # (bs, num)

import re

def deal_duplicate_bb(thead_part):
    """
    Deal duplicate <b> or </b> after replace.
    Keep one <b></b> in a <td></td> token.
    :param thead_part:
    :return:
    """
    # 1. find out <td></td> in <thead></thead>.
    td_pattern = "<td rowspan=\"(\d)+\" colspan=\"(\d)+\">(.+?)</td>|" \
                 "<td colspan=\"(\d)+\" rowspan=\"(\d)+\">(.+?)</td>|" \
                 "<td rowspan=\"(\d)+\">(.+?)</td>|" \
                 "<td colspan=\"(\d)+\">(.+?)</td>|" \
                 "<td>(.*?)</td>"
    td_iter = re.finditer(td_pattern, thead_part)
    td_list = [t.group() for t in td_iter]

    # 2. is multiply <b></b> in <td></td> or not?
    new_td_list = []
    for td_item in td_list:
        if td_item.count('<b>') > 1 or td_item.count('</b>') > 1:
            # multiply <b></b> in <td></td> case.
            # 1. remove all <b></b>
            td_item = td_item.replace('<b>','').replace('</b>','')
            # 2. replace <tb> -> <tb><b>, </tb> -> </b></tb>.
            td_item = td_item.replace('<td>', '<td><b>').replace('</td>', '</b></td>')
            new_td_list.append(td_item)
        else:
            new_td_list.append(td_item)

    # 3. replace original thead part.
    for td_item, new_td_item in zip(td_list, new_td_list):
        thead_part = thead_part.replace(td_item, new_td_item)
    return thead_part

def deal_bb(result_token, tag_='thead'):
    """
    In our opinion, <b></b> always occurs in <thead></thead> text's context.
    This function will find out all tokens in <thead></thead> and insert <b></b> by manual.
    :param result_token:
    :param tag_:
    :return:
    Copy-paste from MTL-TabNet
    """
    # find out <thead></thead> parts.
    thead_pattern = '<' + tag_ + '>(.*?)</' + tag_ + '>'
    if re.search(thead_pattern, result_token) is None:
        return result_token
    thead_part = re.search(thead_pattern, result_token).group()
    origin_thead_part = deepcopy(thead_part)

    if tag_ == 'tbody':
        return result_token

    # check "rowspan" or "colspan" occur in <thead></thead> parts or not .
    span_pattern = "<td rowspan=\"(\d)+\" colspan=\"(\d)+\">|<td colspan=\"(\d)+\" rowspan=\"(\d)+\">|<td rowspan=\"(\d)+\">|<td colspan=\"(\d)+\">"
    span_iter = re.finditer(span_pattern, thead_part)
    span_list = [s.group() for s in span_iter]
    has_span_in_head = True if len(span_list) > 0 else False

    if not has_span_in_head:
        # <thead></thead> not include "rowspan" or "colspan" branch 1.
        # 1. replace <td> to <td><b>, and </td> to </b></td>
        # 2. it is possible to predict text include <b> or </b> by Text-line recognition,
        #    so we replace <b><b> to <b>, and </b></b> to </b>
        thead_part = thead_part.replace('<td>', '<td><b>')\
            .replace('</td>', '</b></td>')\
            .replace('<b><b>', '<b>')\
            .replace('</b></b>', '</b>')
    else:
        # <thead></thead> include "rowspan" or "colspan" branch 2.
        # Firstly, we deal rowspan or colspan cases.
        # 1. replace > to ><b>
        # 2. replace </td> to </b></td>
        # 3. it is possible to predict text include <b> or </b> by Text-line recognition,
        #    so we replace <b><b> to <b>, and </b><b> to </b>

        # Secondly, deal ordinary cases like branch 1

        # replace ">" to "<b>"
        replaced_span_list = []
        for sp in span_list:
            replaced_span_list.append(sp.replace('>', '><b>'))
        for sp, rsp in zip(span_list, replaced_span_list):
            thead_part = thead_part.replace(sp, rsp)

        # replace "</td>" to "</b></td>"
        thead_part = thead_part.replace('</td>', '</b></td>')

        # remove duplicated <b> by re.sub
        mb_pattern = "(<b>)+"
        single_b_string = "<b>"
        thead_part = re.sub(mb_pattern, single_b_string, thead_part)

        mgb_pattern = "(</b>)+"
        single_gb_string = "</b>"
        thead_part = re.sub(mgb_pattern, single_gb_string, thead_part)

        # ordinary cases like branch 1
        thead_part = thead_part.replace('<td>', '<td><b>').replace('<b><b>', '<b>')


    # convert <tb><b></b></tb> back to <tb></tb>, empty cell has no <b></b>.
    # but space cell(<tb> </tb>)  is suitable for <td><b> </b></td>
    thead_part = thead_part.replace('<td><b></b></td>', '<td></td>')
    # deal with duplicated <b></b>
    thead_part = deal_duplicate_bb(thead_part)
    # replace original result with new thead part.
    result_token = result_token.replace(origin_thead_part, thead_part)
    return result_token

def eval_log(_list, mode='bbox'):
    if mode == 'bbox':
        _dict = {
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=512 ]": _list[0],
            "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=512 ]": _list[1],
            "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=512 ]": _list[2],
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=512 ]": _list[3],
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=512 ]": _list[4],
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=512 ]": _list[5],
            "Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=512 ]": _list[6],
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]": _list[7],
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=512 ]": _list[8],
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=512 ]": _list[9],
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=512 ]": _list[10],
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=512 ]": _list[11]}
    else:
        _dict = {
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=512 ]": _list[0],
            "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=512 ]": _list[1],
            "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=512 ]": _list[2],
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=512 ]": _list[3],
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=512 ]": _list[4],
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=512 ]": _list[5],
            "Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=512 ]": _list[6],
            "Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets=512 ]": _list[7],
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=512 ]": _list[8],
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=512 ]": _list[9]}

    return _dict

### Post-processing

def format_scitsr(v):
    structures = deepcopy(v['structures'])
    if structures == []: 
        return {'cells': []}
    box = deepcopy(v['boxes']).tolist()
    lab = deepcopy(v['labels']).tolist()
    if 'tokens' in v:
        tokens = v['tokens']
    else:
        if 'empty' in v:
            tokens = ['cell' * (1-int(x)) + int(x) * '' for x in v['empty']]
        else:
            tokens = ['cell' for _ in structures]
    cells = []
    for i in range(len(structures)):
        sc,sr,ec,er,_ = structures[i]
        content = tokens[i]
        cell = {'id': i,
                'tex': content,
                'content': list(content),
                'start_row': sr,
                'start_col': sc,
                'end_row': er,
                'end_col': ec,
                'box': box[i],
                'pos': box[i]} # x,y,w,h
        cells.append(cell)
    return {'cells': cells}

def from_scitsr_to_gritsform(v, mode='xxyy'):
    ## TODO
    cells = v['cells']
    formed_cells = []
    for c in cells:
        box = c['pos']
        if mode == 'xxyy':
            box = [box[0], box[2], box[1], box[3]]
        elif mode == 'xywh':
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        else:
            raise NotImplementedError(f'mode {mode} is not implemented.')
        sr, sc, er, ec = c['start_row'], c['start_col'], c['end_row'], c['end_col']
        if sc > ec or sr > er: # invalid prediction or gt
            continue
        cell = {'bbox': box,
                'column_nums': list(range(sc, ec+1)),
                'row_nums': list(range(sr, er+1)),
                'header': False,
                'subheader': False,
                # 'spans': ,
                'cell_text': c['tex']
                }
        formed_cells.append(cell)
    return {'cells': formed_cells, 'img_path': v['filename'] if 'filename' in v else None}

def format_icdar2019(v, gt_xml_path, table_id):
    tree = ElementTree.parse(gt_xml_path)
    ref_root = tree.getroot()
    table = ref_root.findall('table')[table_id]
    table_coord = table.findall('Coords')[0].attrib['points']
    table_coord_list = table_coord.split(' ')
    table_coord_list = [list(map(int, x.split(','))) for x in table_coord_list]
    table_x1, table_y1, table_x2, table_y2 = table_coord_list[0][0], table_coord_list[0][1], table_coord_list[2][0], table_coord_list[2][1]

    table_obj = Element('table')
    # positions = "%d,%d %d,%d %d,%d %d,%d" % (positions[0][0], positions[0][1], positions[1][0], positions[1][1], positions[2][0], positions[2][1], positions[3][0], positions[3][1])
    SubElement(table_obj, 'Coords', points=table_coord)

    structures = deepcopy(v['structures'])
    if structures == []: 
        return table_obj
    boxes = deepcopy(v['boxes']).tolist()
    if 'tokens' in v:
        tokens = v['tokens']
    else:
        tokens = ['content'] * len(boxes)

    for struc, box, token in zip(structures, boxes, tokens):
        if len(token) == 0:
            continue
        start_col,start_row,end_col,end_row,_ = list(map(str, struc))

        x, y, w, h = box
        x = max(x + table_x1 - 2, 0)
        y = max(y + table_y1 - 2, 0)
        x2, y2 = x + w, y + h
        # [locations[output_table.TL_ind], locations[output_table.BL_ind], locations[output_table.BR_ind], locations[output_table.TR_ind]]
        positions = "%d,%d %d,%d %d,%d %d,%d" % (x, y, x, y2, x2, y2, x2, y)

        cell_obj = SubElement(table_obj, 'cell')
        attrib_dict = OrderedDict()
        attrib_dict['start-row'] = start_row
        attrib_dict['start-col']=start_col
        attrib_dict['end-row']=end_row
        attrib_dict['end-col']=end_col
        cell_obj.attrib = attrib_dict
        SubElement(cell_obj, 'Coords', points=positions)

    indent(table_obj)
    # dump(root)

    # et = ElementTree.ElementTree(root)
    # et.write(xml_filename, encoding='UTF-8', xml_declaration=True)
    return table_obj

def merge_tables_to_document(pred, xmldir, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    for k, v in pred.items():
        # read gt document tree
        xml_name = f'{k}.xml'
        tree = ElementTree.parse(xmldir / xml_name)
        root = tree.getroot()

        # remove gt table nodes
        original = [x for x in root.findall('table')]
        for x in original:
            root.remove(x)
        
        # insert pred table nodes
        for table_id, table_node in v.items():
            root.insert(table_id, table_node)
        
        indent(root)

        tree.write(outdir / xml_name, encoding='UTF-8', xml_declaration=True)


def format_ptn(v, dataset_mode='pubtabnet'):
    structures = deepcopy(v['structures'])
    if structures == []:
        return {'cells': [], 'structure': {'tokens': []}}
    boxes = deepcopy(v['boxes']).tolist()
    labels = deepcopy(v['labels']).tolist()
    structures = sorted(structures, key=lambda x: (x[1], x[0]))
    if 'tokens' in v:
        tokens = v['tokens']
    else:
        tokens = [['cell'] for _ in structures]
    # html = ['<html>', '<body>', '<table border="1", cellspacing="0" bordercolor="gray">', '<tr>']
    if dataset_mode == 'fintabnet':
        html = ['<table>', '<tr>']
    else: # 'pubtabnet'
        html = ['<thead>', '<tr>']
    head_end = -1
    cells = []
    cur_row = 0
    cur_empty = [] # sc,ec
    cur_rowspan = [] # sc,ec
    for lab, box, structure, tok in zip(labels, boxes, structures, tokens):
        sc,sr,ec,er,is_header = structure
        cell = {'tokens': tok,
                'bbox': [box[0], box[1], box[0]+box[2], box[1]+box[3]],
                }
        cells.append(cell)

        if is_header == 1:
            head_end = sr

        if sr > cur_row:
            cur_row = sr
            html += ['</tr>', '<tr>']
        
        rowspan, colspan = er-sr+1, ec-sc+1
        if is_header == -1:
            if cur_row == 0: # cur_row == sr
                if rowspan > 1: 
                    cur_rowspan += list(range(sc,ec+1))
                    extend_head_end = max(extend_head_end, rowspan - 1)
                if 0 < lab <= 8 and lab % 2 == 0: # 빈 cell
                    cur_empty += list(range(sc,ec+1))

            if cur_row == head_end + 1:
                if 0 < lab <= 8 and lab % 2 == 0:
                    if len([x for x in range(sc,ec+1) if x not in cur_empty]) == 0: # 빈 cell 두개가 연속인 경우
                        extend_head_end = max(extend_head_end, rowspan)

        if rowspan > 1 or colspan > 1:
            html.append('<td')
            if rowspan > 1:
                html.append(f' rowspan="{rowspan}"')
            if colspan > 1:
                html.append(f' colspan="{colspan}"')
            html.append('>')
        else:
            html.append('<td>')
        html.append('</td>')
    if dataset_mode == 'fintabnet':
        html += ['</tr>', '</table>']
    else:
        html += ['</tr>', '</tbody>']
        rowind = [i for i in range(len(html)) if html[i] == '</tr>']

        if len(rowind) <= head_end or len(html) <= rowind[head_end]+1:
            print(v['filename'], f': head_end {head_end}, while len(rowind) = {len(rowind)} and len(html) = {len(html)}')
        html.insert(rowind[head_end]+1, '</thead>')
        html.insert(rowind[head_end]+2, '<tbody>')

    return {'cells': cells, 'structure': {'tokens': html}}


def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
         if level and (not elem.tail or not elem.tail.strip()):
             elem.tail = i

# args
def overload_cfg(cfg, args):
    overloaded_keys_map = {
                           'batch_size': ['train', 'test'],
                           'epochs': 'train',
                           'use_amp': 'train',
                           'val_batch_size': 'train',
                           'max_seq_len': ['train', 'test'],
                           'apply_as': 'test',
                           'coco_path': 'dataset',
                           'dataset_file': 'dataset',
                           'image_path': 'dataset',
                           'testdir': 'dataset',
                           'gtpath': 'dataset',
                           'dataset_mode': 'dataset',
                           'output_dir': 'exec',
                           'log_dir': 'exec',
                           'device': 'exec',
                           'resume': 'exec',
                           'eval': 'exec',
                           'num_workers': 'exec',
                           'pp': 'exec',
                           'draw_attn': 'exec',
                           'no_eval': 'exec',
                           'deterministic': 'exec',
                           'apply_ocr': 'exec',
                           'apply_gt_ocr': 'exec',
                           'world_size': 'exec',
                           'dist_url': 'exec',
                           'seed': 'exec',
                            }
    
    for k, v in vars(args).items():
        if k in overloaded_keys_map or k not in cfg:
            if v == None:
                continue
            if k == 'max_seq_len' or k == 'batch_size':
                continue
            elif k == 'config':
                continue
            if v == 'none':
                v = None
            if k not in overloaded_keys_map:
                OmegaConf.update(cfg, k, v)
            else:
                OmegaConf.update(cfg, f'{overloaded_keys_map[k]}.{k}', v)
            
                
    idx = 1 if cfg.exec.eval else 0
    for k in ['batch_size', 'max_seq_len']:
        if k in args and vars(args)[k] is not None:
            OmegaConf.update(cfg, f'{overloaded_keys_map[k][idx]}.{k}', vars(args)[k])

    return cfg

import json
def summarize_teds_score(scores, filenames, output_dir, eval_type, gt):
    mean_score = float(np.mean(np.asarray(list(scores.values()))))
    scores_dict = {'scores': scores}
    scores_dict['mean'] = mean_score
    with open(output_dir / f'teds_{eval_type}.json', 'w') as f:
        json.dump(scores_dict, f, indent=2)

    # simple / complex
    complex_fnames = [x for x in filenames if 'colspan' in gt[x]['html'] or 'rowspan' in gt[x]['html']]
    simple_fnames = [x for x in filenames if x not in complex_fnames]
    simple_scores = {k: scores[k] for k in simple_fnames}
    mean_simple_score = float(np.mean(np.asarray(list(simple_scores.values()))))
    simple_scores = {'scores': simple_scores, 'mean': mean_simple_score}
    complex_scores = {k: scores[k] for k in complex_fnames}
    mean_complex_score = float(np.mean(np.asarray(list(complex_scores.values()))))
    complex_scores = {'scores': complex_scores, 'mean': mean_complex_score}
    with open(output_dir / f'teds_{eval_type}_simple.json', 'w') as f:
        json.dump(simple_scores, f, indent=2)
    with open(output_dir / f'teds_{eval_type}_complex.json', 'w') as f:
        json.dump(complex_scores, f, indent=2)
    
    # large / small
    large_fnames = [x for x in filenames if gt[x]['html'].count('</td>') >= 150]
    small_fnames = [x for x in filenames if x not in large_fnames]
    large_scores = {k: scores[k] for k in large_fnames}
    mean_large_score = float(np.mean(np.asarray(list(large_scores.values()))))
    large_scores = {'scores': large_scores, 'mean': mean_large_score}
    small_scores = {k: scores[k] for k in small_fnames}
    mean_small_score = float(np.mean(np.asarray(list(small_scores.values()))))
    small_scores = {'scores': small_scores, 'mean': mean_small_score}
    with open(output_dir / f'teds_{eval_type}_larger_than_100.json', 'w') as f:
        json.dump(large_scores, f, indent=2)
    with open(output_dir / f'teds_{eval_type}_less_than_100.json', 'w') as f:
        json.dump(small_scores, f, indent=2)

    # many empty cells / less empty cells (or none)
    empty_fnames = [x for x in filenames if ''.join(gt[x]['html']).count('<td></td>') >= 25]
    nonempty_fnames = [x for x in filenames if x not in large_fnames]
    empty_scores = {k: scores[k] for k in empty_fnames}
    mean_empty_score = float(np.mean(np.asarray(list(empty_scores.values()))))
    empty_scores = {'scores': empty_scores, 'mean': mean_empty_score}
    nonempty_scores = {k: scores[k] for k in nonempty_fnames}
    mean_nonempty_scores = float(np.mean(np.asarray(list(nonempty_scores.values()))))
    nonempty_scores = {'scores': nonempty_scores, 'mean': mean_nonempty_scores}
    with open(output_dir / f'teds_{eval_type}_empty_morethan_25.json', 'w') as f:
        json.dump(empty_scores, f, indent=2)
    with open(output_dir / f'teds_{eval_type}_empty_lessthan_25.json', 'w') as f:
        json.dump(nonempty_scores, f, indent=2)

    print(f'TEDS-{eval_type}. Total mean ({len(scores)} samples): {mean_score}, \
            Simple mean ({len(simple_scores["scores"])}): {mean_simple_score}, \
            Complex mean ({len(complex_scores["scores"])}) : {mean_complex_score}.')
    print(f'Cells >= 150 mean ({len(large_scores["scores"])}): {mean_large_score}, \
            < 150 mean ({len(small_scores["scores"])}) : {mean_small_score}.')
    print(f'Empty >= 25 mean ({len(empty_scores["scores"])}): {mean_empty_score}, \
            Empty < 25 mean ({len(nonempty_scores["scores"])}) : {mean_nonempty_scores}.')
    with (output_dir / "log.txt").open("a") as f:
        f.write(f'TEDS-{eval_type}. Total mean ({len(scores)} samples): {mean_score}, \
            Simple mean ({len(simple_scores["scores"])}): {mean_simple_score}, \
            Complex mean ({len(complex_scores["scores"])}) : {mean_complex_score}.\n')
        f.write(f'Cells >= 150 mean ({len(large_scores["scores"])}): {mean_large_score}, \
            < 150 mean ({len(small_scores["scores"])}) : {mean_small_score}.\n')
        f.write(f'Empty >= 25 mean ({len(empty_scores["scores"])}): {mean_empty_score}, \
            Empty < 25 mean ({len(nonempty_scores["scores"])}) : {mean_nonempty_scores}.\n')