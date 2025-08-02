# -----------------------------------------------------------------------
# S2Tab official code : model/__init__.py
# -----------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -----------------------------------------------------------------------

def build_model(args, vocab=None):
    if args.dataset.seq_version == 'cell':
        from .s2tab_cellbox import build
    elif args.dataset.seq_version == 'content':
        from .s2tab_contentbox import build
    else:
        raise NotImplementedError(f'Model for seq_version {args.dataset.seq_version} is not implemented.')

    return build(args, vocab)
