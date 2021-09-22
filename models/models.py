import torch.nn as nn
from src.models.tresnet.tresnet import TResNet
from src.models.utils.registry import register_model
from src.models.aggregate.layers.frame_pooling_layer import Aggregate
from src.models.aggregate.layers.transformer_aggregate import TAggregate
# from src.models.resnet.resnet import Bottleneck as ResnetBottleneck
from src.models.resnet.resnet import ResNet

__all__ = ['MTResnetAggregate']


class fTResNet(TResNet):

  def __init__(self, aggregate=None, *args, **kwargs):
    super(fTResNet, self).__init__(*args, **kwargs)
    self.aggregate = aggregate
    # self.global_pool = 'avg'
    #   if 'global_pool' in kwargs:
    #       self.global_pool = kwargs[]

  def forward(self, x, filenames=None):
    x = self.body(x)
    self.embeddings = self.global_pool(x)

    if self.aggregate:
        # self.embeddings = self.aggregate(self.embeddings, filenames)
        # self.embeddings, attn_mat = self.aggregate(self.embeddings, filenames)
        if isinstance(self.aggregate,TAggregate):
           self.embeddings, self.attention = self.aggregate(self.embeddings, filenames)
           logits = self.head(self.embeddings)
        else: # CNN aggregation:
            logits = self.head(self.embeddings)
            # logits aggregation before softmax:
            # logits = self.aggregate(logits)
            logits = self.aggregate(nn.functional.softmax(logits, dim=1))
            ## Embeddings aggregation:
            # self.embeddings = self.aggregate(self.embeddings, filenames)
            # logits = self.head(self.embeddings)

    # if attn_mat is None:
    #     return logits
    # else:
    #     return (logits, attn_mat)
    return logits


class fResNet(ResNet):
  def __init__(self, aggregate=None, **kwargs):
    super().__init__(**kwargs)
    self.aggregate = aggregate

  def forward(self, x):
    x = self.body(x)
    if self.aggregate:
      x = self.head.global_pool(x)
      x, attn_weight = self.aggregate(x)
      logits = self.head.fc(self.head.FlattenDropout(x))

    else:
      logits = self.head(x)
    return logits


@register_model
def MTResnetAggregate(model_params):
    """Constructs a medium TResnet model.   Frame Pooling MTResNet (frame pooling)
    """

    in_chans = 3
    num_classes = model_params['num_classes']
    args = model_params['args']
    if 'global_pool' in args and args.global_pool is not None:
        global_pool = args.global_pool
    else:
        global_pool = 'avg'
    do_bottleneck_head = args.do_bottleneck_head
    bottleneck_features = args.bottleneck_features
    remove_model_jit = args.remove_model_jit

    aggregate = None
    if args.use_transformer:
      aggregate = TAggregate(args.album_clip_length, args=args)
    else:
      aggregate = Aggregate(args.album_clip_length, args=args)

    model = fTResNet(layers=[3, 4, 11, 3], num_classes=num_classes, in_chans=in_chans,
                    # global_pool=global_pool,
                    do_bottleneck_head=do_bottleneck_head,
                    bottleneck_features=bottleneck_features,
                    # remove_model_jit=remove_model_jit,
                    aggregate= aggregate)


    return model

'''
@register_model
def resnet101aggregate(model_params, in_chans=3, **kwargs):
    """Constructs a ResNet-101 model.
    """
    num_classes = model_params['num_classes']
    args = model_params['args']

    aggregate = None
    if args.use_transformer:
      aggregate = TAggregate(args.album_clip_length, args=args)
    else:
      aggregate = Aggregate(args.album_clip_length, args=args)

    model = fResNet(aggregate=aggregate, block=ResnetBottleneck, layers=[3, 4, 23, 3],
                    num_classes=num_classes,
                    in_chans=in_chans,
                    **kwargs)
    return model


@register_model
def resnet50aggregate(model_params, in_chans=3, **kwargs):
    """Constructs a ResNet-50 model.
    """
    num_classes = model_params['num_classes']
    args = model_params['args']

    aggregate = None
    if args.use_transformer:
      aggregate = TAggregate(args.album_clip_length, args=args)
    else:
      aggregate = Aggregate(args.album_clip_length, args=args)

    model = fResNet(aggregate=aggregate, block=ResnetBottleneck, layers=[3, 4, 6, 3],
                    num_classes=num_classes,
                    in_chans=in_chans,
                    **kwargs)

    return model
'''