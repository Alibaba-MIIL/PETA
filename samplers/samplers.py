import numpy as np
import math
import torch.distributed as dist
from torch.utils.data import Sampler

class ValOrderedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, args, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available() or not (dist.is_initialized()):
                # raise RuntimeError("Requires distributed package to be available")
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not (dist.is_initialized()):
                # raise RuntimeError("Requires distributed package to be available")
                rank = 0
            else:
                rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.args = args
        self.epoch = 0

        # compute num_images_per_album, album_indices
        albums = np.array([fname.rpartition('/')[0] for fname, _ in self.dataset.samples])
        unique_albums, self.album_indices, self.album_counts = np.unique(albums, return_index=True,
                                                                         return_counts=True)

        self.num_samples = int(math.ceil(len(unique_albums) * self.args.album_clip_length * 1.0 /
                                         self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas


    def __iter__(self):
        # indices = list(range(len(self.dataset)))
        album_indices, counts = self.album_indices, self.album_counts
        clip_length = self.args.album_clip_length
        indices_list = []
        if self.args.album_sample == 'uniform_ordered':
            for alb_ind, alb_cnt in zip(album_indices, counts):
                if alb_cnt >= clip_length:
                    step = alb_cnt//clip_length
                    rand_start = np.random.randint(0, high = step)
                    album_indices = np.linspace(alb_ind + rand_start, alb_ind + alb_cnt, num=clip_length, endpoint=False, dtype=int)
                else:
                    album_indices = np.linspace(alb_ind, alb_ind + alb_cnt, num=clip_length, endpoint=False, dtype=int)
                indices_list.append(album_indices)
            indices = list(np.concatenate(indices_list))

        elif self.args.album_sample=='rand_permute':
            if any(counts < clip_length):
                # complete albums with less than clip_length images:
                for alb_ind, alb_cnt in zip(album_indices, counts):
                    if alb_cnt>=clip_length:
                        album_indices = np.random.permutation(np.arange(alb_ind, alb_ind + alb_cnt))[:clip_length]
                    else:
                        album_indices = np.concatenate(
                            (np.random.permutation(np.arange(alb_ind, alb_ind + alb_cnt))[:clip_length],
                            np.random.permutation(np.arange(alb_ind, alb_ind + alb_cnt))[:clip_length - alb_cnt]))
                    indices_list.append(album_indices)
                indices = list(np.concatenate(indices_list))
            else:
                indices = list(np.concatenate(
                    [np.random.permutation(np.arange(alb_ind, alb_ind + alb_cnt))[:clip_length] for
                     alb_ind, alb_cnt in zip(album_indices, counts)]))

        # self.num_samples = int(math.ceil(len(indices) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples