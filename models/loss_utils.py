# -----------------------------------------------------------------------
# S2Tab official code : models/loss_utils.py
# -----------------------------------------------------------------------

import torch
import torch.nn.functional as func
from torchvision.utils import _log_api_usage_once

# loss functions
def softmax_focal_loss(inputs: torch.Tensor,
                       targets: torch.Tensor,
                       gamma: float = 2,
                       reduction: str = 'none',
                       num_classes: int = None
                       ) -> torch.Tensor:
    p = torch.softmax(inputs, dim=1)
    ce_loss = func.cross_entropy(inputs, targets, reduction='none')
    target_onehot = func.one_hot(targets, num_classes=num_classes)
    target_onehot = target_onehot.permute(0, 2, 1)
    p_t = p * target_onehot
    p_t = p_t.sum(1)
    loss = ce_loss * ((1 - p_t) ** gamma)
    
    if reduction == 'none':
        pass
    elif reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid reduction {reduction}, only 'none', 'mean', 'sum' allowed."
        )
    
    return loss

def heatmap_focal_loss(inputs: torch.Tensor,
                       targets: torch.Tensor,
                       gamma: float = 2,
                       beta: float = 4,
                       reduction: str = 'none',
                       positive_thresholds: torch.Tensor = None,
                       ):
    """
    Loss used in CornerNet: Detecting Objects as Paired Keypoints.

    Modified from https://pytorch.org/vision/main/generated/torchvision.ops.sigmoid_focal_loss.html

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        beta (float): Weighting factor in range (0,1) to balance
                positive vs negative examples, according to the target value or -1 for ignore. Default: ``4``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(heatmap_focal_loss)
    p = torch.sigmoid(inputs)
    if positive_thresholds is not None:
        binary_targets = torch.greater_equal(targets, positive_thresholds[:, None, None, None]).to(targets.dtype)
    else:
        binary_targets = (targets >= 0.98).to(targets.dtype) # downsampled-heatmap contains very few exact-1 points. near-1 is observed as positive
    ce_loss = func.binary_cross_entropy_with_logits(inputs, binary_targets, reduction="none")
    p_t = p * binary_targets + (1 - p) * (1 - binary_targets)
    gaussian_bump = (1 - targets) * (1 - binary_targets) + binary_targets
    loss = ce_loss * ((1 - p_t) ** gamma) * (gaussian_bump ** beta)

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss