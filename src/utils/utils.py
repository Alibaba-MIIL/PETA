import os
from functools import partial
import torch
from fastai2.torch_core import to_detach
from torchvision.datasets import Kinetics400
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
from src.augmentations.generate_transforms import generate_validation_transform
from src.datasets.pytorch_datasets_parser import DatasetFromList
from src.samplers.samplers import ValOrderedSampler
from src.report_manager.utils import AccumMetricG, accuracy, mAP

def trunc_normal_(x, mean=0., std=1.):
  "Truncated normal initialization (approximation)"
  # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
  return x.normal_().fmod_(2).mul_(std).add_(mean)


def vid_transform_fn(x, fn):
  return [fn(Image.fromarray(X.squeeze(dim=0).data.numpy())) for X in x]


def create_val_dataset(args, transform, add_extra_data=True):

    source = args.val_dir

    val_dl = DatasetFromList(source, transform=transform, args=args)

    return val_dl


def fast_collate(batch, clip_length=None):
  targets = torch.tensor([b[1] for b in batch])
  batch_size = len(targets)
  dims = (batch[0][0].shape[2], batch[0][0].shape[0], batch[0][0].shape[1])  # HWC to CHW
  tensor_uint8_CHW = torch.empty((batch_size, *dims), dtype=torch.uint8)
  for i in range(batch_size):
    tensor_uint8_CHW[i] = torch.from_numpy(batch[i][0]).permute(2, 0, 1)  # # HWC to CHW
    # tensor_uint8_CHW[i] = batch[i][0].permute(2, 0, 1)  # # HWC to CHW # (Added)
  targets = targets.view(batch_size // clip_length, clip_length, -1)[:, 0]
  return tensor_uint8_CHW.float(), targets  # , extra_data


# def video_collate(batch):
#   is_np = isinstance(batch[0][0][0], np.ndarray)
#   T = len(batch[0][0])  # number of images
#
#   targets = torch.tensor([b[1] for b in batch])
#   batch_size = len(batch)
#   if is_np:
#     dims = (batch[0][0][0].shape[2], batch[0][0][0].shape[0], batch[0][0][0].shape[1])
#     tensor_uint8_CHW = torch.empty((T * batch_size, *dims), dtype=torch.uint8)
#     for i in range(batch_size):
#       for t in range(T):
#         tensor_uint8_CHW[i * T + t] = \
#           torch.from_numpy(batch[i][0][t]).permute(2, 0, 1)
#     return tensor_uint8_CHW, targets
#
#   else:
#     dims = (batch[0][0][0].shape[0], batch[0][0][0].shape[1], batch[0][0][0].shape[2])
#     tensor_float_CHW = torch.empty((T * batch_size, *dims), dtype=torch.float)
#     for i in range(batch_size):
#       for t in range(T):
#         tensor_float_CHW[i * T + t] = batch[i][0][t]
#     return tensor_float_CHW, targets



def create_dataloader(args):
  val_bs = args.batch_size

  val_transform = generate_validation_transform(args) #, do_prefetch=False)
  # val_transform = generate_validation_transform(args,  do_prefetch=False)
  '''
  # if args.input_size == 448:  # squish
  val_tfms = transforms.Compose(
      [transforms.Resize((args.input_size, args.input_size))])
  # else:  # crop
  #   val_tfms = transforms.Compose(
  #     [transforms.Resize(int(args.input_size / args.val_zoom_factor)),
  #      transforms.CenterCrop(args.input_size)])
  val_tfms.transforms.append(transforms.ToTensor())
  '''
  # val_dataset1 = ImageFolder(args.val_dir, val_transform)
  val_dataset = create_val_dataset(args, val_transform) #val_tfms)

  # creating samplers
  # train_sampler, val_sampler = create_samplers(args, train_data, valid_data)
  val_sampler = ValOrderedSampler(val_dataset, args=args)

  # load data
  # val_loader = torch.utils.data.DataLoader(
  #   val_dataset, batch_size=val_bs, shuffle=False,
  #   num_workers=args.num_workers, collate_fn=video_collate, pin_memory=True, drop_last=False)

  valid_dl_pytorch = torch.utils.data.DataLoader(
    val_dataset, sampler=val_sampler, batch_size=args.batch_size, shuffle=False, pin_memory=True,
    num_workers=args.num_workers, drop_last=False, collate_fn=partial(fast_collate, clip_length=args.album_clip_length))
  # valid_dl_pytorch = PrefetchLoader(valid_dl_pytorch, args=args)

  return valid_dl_pytorch # val_loader

def accumulate_scores_targets_filenames(targets_, scores_):
  return targets_, scores_

def enable_detach():
  to_detach.__defaults__ = (True, True)



# def accuracy(output, target, topk=(1,)):
#   """Computes the precision@k for the specified values of k"""
#   maxk = max(topk)
#   batch_size = target.size(0)
#   _, pred = output.topk(maxk, 1, True, True)
#   pred = pred.t()
#   correct = pred.eq(target.view(1, -1).expand_as(pred))
#   return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self): self.reset()

  def reset(self): self.val = self.avg = self.sum = self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def validate(model, val_loader, classes_list, threshold):

  accum = AccumMetricG(None, dim_argmax=None, sigmoid=False, thresh=None, to_np=False,
                       invert_arg=True, flatten=False)

  with torch.no_grad():
    for input, target in val_loader:
      input = input.cuda() / 255.0
      target = target.cuda()
      thresh = 0.6
      logits = model(input)
      preds = torch.sigmoid(logits)
      np.where(preds > thresh)[0]

      # map
      # logits = model(output, filenames=filenames)
      # logits = model(output)
      accum.accumulate(logits, target)

      # prec1, prec5 = accuracy(output, target, topk=(1, 5))
      # prec1_m.update(prec1.item(), output.size(0))
      # prec5_m.update(prec5.item(), output.size(0))

      # if (last_batch or batch_idx % 100 == 0):
      #   log_name = 'Kinetics Test'
      #   print(
      #     '{0}: [{1:>4d}/{2}]  '
      #     'Prec@1: {top1.val:>7.2f} ({top1.avg:>7.2f})  '
      #     'Prec@5: {top5.val:>7.2f} ({top5.avg:>7.2f})'.format(
      #       log_name, batch_idx, last_idx,
      #       top1=prec1_m, top5=prec5_m))

  # Modify class idx for calculating mAP
  class_map = {0: "Birthday", 1: "Christmas", 2: "Graduation", 3: "Personal_sports", 4: "Show", 5: "ThemePark"}

  targs, preds = accum.value(lambda x, y: (x, y))
  # ap_partial, map, map_macro, cnt_class_with_no_labels, cnt_class_with_no_neg, cnt_class_with_no_pos = \

  # acc = accuracy(preds, targs)
  #for multi-label
  acc = mAP(targs, preds)

  enable_detach()

  return acc
