import torch
import numpy as np
from itertools import chain
from fastai2.torch_core import to_detach, flatten_check, store_attr
from fastai2.metrics import Metric

class AccumMetricG(Metric):
    "Stores predictions and targets on CPU in accumulate to perform final calculations with `func`."

    def __init__(self, func, dim_argmax=None, sigmoid=False, thresh=None, to_np=False, invert_arg=False,
                 flatten=True):
        store_attr(self,'func,dim_argmax,sigmoid,thresh,flatten')
        self.to_np,self.invert_args = to_np, invert_arg
        self.album_voting = 'attention'
        self.reset()

    def reset(self): self.targs,self.preds = [],[]

    def accumulate(self, x, y):
        pred = x.argmax(dim=self.dim_argmax) if self.dim_argmax else x
        if self.sigmoid:
            pred = torch.sigmoid(pred)
        if self.thresh:
            pred = (pred >= self.thresh)
        targ = y
        pred,targ = to_detach(pred),to_detach(targ)
        if self.flatten: pred,targ = flatten_check(pred,targ)
        self.preds.append(pred)
        self.targs.append(targ)

    # def accumulate_by_album(self, x, y, filenames):
    #     albums = [filename.rpartition('/')[0] for filename in filenames]
    #     cur_albums = list(set(albums))
    #     pred = x
    #     if self.album_voting == 'attention':
    #         pred = torch.zeros(len(y), device=x.device, dtype=x.dtype)
    #         imgs4album = x.shape[1]-1
    #     targ = y
    #     for album in cur_albums:
    #         ind_album = [i for i, x in enumerate(albums) if x == album]
    #         if self.album_voting == 'per_image_th':
    #             # pred_album = np.mean(x[albums==album]>per_img_th)
    #             pred_album = np.mean(x[ind_album,:] > self.thresh, axis=0)
    #             pred[ind_album] = pred_album
    #         elif self.album_voting == 'total_conf_th':
    #             pred_album = np.mean(x[ind_album,:], axis=0)
    #             pred[ind_album] = pred_album
    #         elif self.album_voting == 'attention' :
    #             ialbum = (ind_album[0])// imgs4album
    #             pred_album = x[ialbum,0,1:] #    .cpu().numpy()
    #             pred[ind_album] = pred_album
    #             # self.targs.append(targ)
    #             # self.filenames.append(album)
    #
    #     # pred = x.argmax(dim=self.dim_argmax) if self.dim_argmax else x
    #     # if self.sigmoid:
    #     #     pred = torch.sigmoid(pred)
    #     # if self.thresh:
    #     #     pred = (pred >= self.thresh)
    #     pred,targ = to_detach(pred),to_detach(targ)
    #     if self.flatten: pred,targ = flatten_check(pred,targ)
    #     self.preds.append(pred)
    #     self.targs.append(targ)


    def value(self, func, **kwargs):
        if len(self.preds) == 0: return
        preds = torch.cat(self.preds)
        preds,targs = torch.cat(self.preds),torch.cat(self.targs)
        if 'filenames' in kwargs.keys():
            # kwargs.filenames = list(chain(*kwargs.filenames))
            kwargs['filenames'] = list(chain.from_iterable(kwargs['filenames']))
        if self.to_np: preds,targs = preds.numpy(),targs.numpy()
        return func(targs, preds, **kwargs) if self.invert_args else self.func(preds, targs, **kwargs)

#
# def accuracy(output, target, topk=(1,)):
#     """ Computes the precision@k for the specified values of k """
#     maxk = max(topk)
#     batch_size = target.size(0)
#
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res

def accuracy(inp, targ, axis=-1):
    "Compute accuracy with `targ` when `pred` is bs * n_classes"
    pred,targ = flatten_check(inp.argmax(dim=axis), targ)
    return (pred == targ).float().mean()


def average_precision(output, target):
    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_/(total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100*ap.mean()
