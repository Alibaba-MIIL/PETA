import torch
from torch import nn as nn, Tensor


class AsymmetricLoss(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading during forward pass'''

    def __init__(self, args, class_task=None):
        super(AsymmetricLoss, self).__init__()
        self.args = args
        self.gamma_neg = args.gamma_neg if args.gamma_neg is not None else 4
        self.gamma_pos = args.gamma_pos if args.gamma_pos is not None else 0.05
        self.clip = args.clip if args.clip is not None else 0.05
        # self.class_task = class_task
        self.multiset_rank = args.multiset_rank  # Used also to identify multi-task training

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.targets_weights = None

    def forward(self, logits, targets_inputs):
        if not self.training:  # this is a complicated loss. for validation, just return 0
            return 0

        if self.targets is None or self.targets.shape != targets_inputs.shape:
            self.targets = targets_inputs.clone()
        else:
            self.targets.copy_(targets_inputs)
        targets = self.targets

        # initial calculations
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        targets_weights = self.targets_weights
        targets, targets_weights, xs_neg = edit_targets_parital_labels(self.args, targets, targets_weights,
                                                                       xs_neg)
        anti_targets = 1 - targets

        # construct weight matrix for multi-set
        # if False and self.multiset_rank is not None:
        #     self.targets_weights = get_multiset_target_weights(self.targets, self.targets_weights,
        #                                                        self.class_task,
        #                                                        self.multiset_rank)

        # One sided clipping
        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)

        # CE loss calculation
        BCE_loss = targets * torch.log(torch.clamp(xs_pos, min=1e-8))
        if self.args.alpha_pos is not None:
            BCE_loss.mul_(self.args.alpha_pos)
        neg_loss = anti_targets * torch.log(torch.clamp(xs_neg, min=1e-8))
        if self.args.alpha_neg is not None:
            neg_loss.mul_(self.args.alpha_neg)
        BCE_loss.add_(neg_loss)

        # Adding asymmetric gamma weights
        with torch.no_grad():
            asymmetric_w = torch.pow(1 - xs_pos * targets - xs_neg * anti_targets,
                                     self.gamma_pos * targets + self.gamma_neg * anti_targets)
        BCE_loss *= asymmetric_w

        # partial labels weights
        BCE_loss *= targets_weights

        # multi-task weights
        if hasattr(self, "weight_task_batch"):
            BCE_loss *= self.weight_task_batch

        return -BCE_loss.sum()


def edit_targets_parital_labels(args, targets, targets_weights, xs_neg):
    # targets_weights is and internal state of AsymmetricLoss class. we don't want to re-allocate it every batch
    if args.partial_loss_mode is None:
        targets_weights = 1.0
    elif args.partial_loss_mode == 'negative':
        # set all unsure targets as negative
        targets[targets == -1] = 0
        targets_weights = 1.0
    elif args.partial_loss_mode == 'negative_backprop':
        if targets_weights is None or targets_weights.shape != targets.shape:
            targets_weights = torch.ones(targets.shape, device=torch.device('cuda'))
        else:
            targets_weights[:] = 1.0
        num_top_confused_classes_to_remove_backprop = args.num_classes_to_remove_negative_backprop * \
                                                      targets_weights.shape[0]  # 50 per sample
        negative_backprop_fun_jit(targets, xs_neg, targets_weights,
                                  num_top_confused_classes_to_remove_backprop)

        # set all unsure targets as negative
        targets[targets == -1] = 0

    elif args.partial_loss_mode == 'real_partial':
        # remove all unsure targets (targets_weights=0)
        targets_weights = torch.ones(targets.shape, device=torch.device('cuda'))
        targets_weights[targets == -1] = 0

    return targets, targets_weights, xs_neg


@torch.jit.script
def negative_backprop_fun_jit(targets: Tensor, xs_neg: Tensor, targets_weights: Tensor,
                              num_top_confused_classes_to_remove_backprop: int):
    with torch.no_grad():
        targets_flatten = targets.flatten()
        cond_flatten = torch.where(targets_flatten == -1)[0]
        targets_weights_flatten = targets_weights.flatten()
        xs_neg_flatten = xs_neg.flatten()
        ind_class_sort = torch.argsort(xs_neg_flatten[cond_flatten])
        targets_weights_flatten[
            cond_flatten[ind_class_sort[:num_top_confused_classes_to_remove_backprop]]] = 0
