import torch
import torch.nn as nn
import torch.nn.functional as F
# from src.models.efficientnet.efficientnet import efficientnet_b1_pruned
# from src.models.ofa.model_zoo import ofa_flops_389m
from PIL import Image
import numpy as np


def distL1(v1, v2, th):
    v1[v1 > th] = 1
    v2[v2 > th] = 1
    d = torch.sum(torch.abs(v1 - v2))
    return d


# class FramePooling(nn.Module):
#   def __init__(self, pooling_args, num_input_frames=64, num_out_frame=4, pooling_type='model',
#                **kwargs):
#     super(FramePooling, self).__init__()
#     self.frame_pooling_type = pooling_type
#     self.downsample = 0.5
#     self.th = 0.6
#     print('pooling type ', pooling_type)
#     self.clip_length = num_input_frames
#     self.num_input_frames = num_input_frames
#     self.num_out_frame = num_out_frame
#     num_out = self.num_out_frame * self.num_input_frames
#
#     self.return_inds = False
#     if 'return_inds' in kwargs.keys():
#       self.return_inds = kwargs['return_inds']
#     pooling_model_name = 'ofa'
#
#     if self.frame_pooling_type == 'model':
#       pooling_args['embeddings_only'] = True
#       self.pooling_model = efficientnet_b1_pruned(pooling_args)  # efficientnet
#       feat_size = self.pooling_model.num_features
#       self.classifier = nn.Linear(feat_size * self.clip_length ,num_out)
#     elif self.frame_pooling_type == 'heuristic':
#       pooling_args['num_classes'] = 1600
#
#       if pooling_model_name == 'efficientnet':
#         pooling_args['embeddings_only'] = False
#         self.pooling_model = efficientnet_b1_pruned(pooling_args)
#       elif pooling_model_name == 'ofa':
#         pooling_args['args'].gpu_mobile_friendly = 0  # tal note - backward compatability. should be 1 after making sure results are the same
#         self.pooling_model = ofa_flops_389m(pooling_args)
#
#       # pooling_args['embeddings_only'] = False
#       # self.pooling_model = efficientnet_b1_pruned(pooling_args)
#
#
#     self.counter = 0
#     self.save_sample_frames = False
#
#   def forward(self, x):
#     #use attention vector which was stored from FrameSampler callback , and do matmul , similar to
#     #http://gitlab.alibaba-inc.com/israel-hq/FrameSelectionNetwork/blob/master/network/smart_pooling.py#L66
#     # print('x.shape ', x.shape)
#
#     batch = x.shape[0] // self.clip_length
#     width_orig = x.shape[2]
#     height_orig = x.shape[3]
#
#     if self.frame_pooling_type is None:
#       return x
#     elif self.frame_pooling_type == 'uniform':
#       o = []
#       for i in range(batch):
#         input = x[i*self.clip_length:(i+1)*self.clip_length]
#         skip = input.shape[0] // self.num_out_frame
#         if self.frame_pooling_type == 'random':
#           idx = torch.rand(self.num_out_frame).long()
#         else:
#           idx = torch.arange(0, input.shape[0], skip).long()[:self.num_out_frame]
#         o.append(input[idx])
#
#       o = torch.cat(o, dim=0)
#       return o
#
#     if self.downsample:
#       x_down = nn.functional.interpolate(x, scale_factor=self.downsample)
#     else:
#       x_down = x
#
#     feat = self.pooling_model(x_down)
#
#     if self.frame_pooling_type == 'heuristic':
#       #rank the frames by the average all categories
#       # get the top N frames
#
#       conf = self.pooling_model(x_down)
#       conf = torch.sigmoid(conf).float()
#       conf = torch.where(conf < self.th, torch.zeros_like(conf), conf)
#       x_inds = []
#       batch_inds = []
#       num_vids_in_batch = conf.size()[0] // self.clip_length
#
#       conf = conf.view((num_vids_in_batch, self.clip_length))
#       V_sort, ind_sort = torch.sort(conf, dim=-1)
#       indexes = ind_sort.narrow(1, -2, 2)
#       x_inds = [x[n * self.clip_length + ind] for n, ind in enumerate(indexes)]
#       x_inds = torch.cat(x_inds, dim=0)
#
#       # for v in range(num_vids_in_batch):
#       #   X_v = x[self.clip_length * v:self.clip_length * (v + 1)]
#       #   V = conf[self.clip_length * v:self.clip_length * (v + 1)]
#       #   V_sort, _ = torch.sort(V, dim=1)
#       #   _, ind_sort = torch.sort(torch.mean(V_sort[:, -5:], dim=1))
#       #   indexes = ind_sort[-2:]
#       #   batch_inds.extend(indexes)
#       #   x_inds.append(X_v[indexes])
#       # x_inds = torch.cat(x_inds, dim=0)
#
#       if self.return_inds:
#         return batch_inds
#       else:
#         return x_inds
#
#
#
#     feat = feat.reshape(batch, -1)
#     h = self.classifier(feat)
#     h = h.reshape(batch, self.num_out_frame, self.num_input_frames)
#     h = F.softmax(h, dim=-1)
#
#     return h[0].argmax(dim=1)



class Aggregate(nn.Module):
  def __init__(self, sampled_frames=None, nvids=None, args=None):
    super(Aggregate, self).__init__()
    self.clip_length = sampled_frames
    self.nvids = nvids
    self.args = args


  def forward(self, x, filenames=None):
    nvids = x.shape[0] // self.clip_length
    x = x.view((-1, self.clip_length) + x.size()[1:])
    o = x.mean(dim=1)
    return o
