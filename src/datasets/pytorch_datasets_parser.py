from PIL import Image
import torch.utils.data as data
import itertools
import os
import numpy as np

def default_loader(path):
    img = Image.open(path)
    return img.convert('RGB')

def get_impaths_from_dir(dirpath, args=None):
    impaths = []
    labels = []
    cls_dirs = os.listdir(dirpath)
    for cls in cls_dirs:
        dir_cls = os.path.join(dirpath,cls)
        albums = os.listdir(dir_cls)
        # albums = os.listdir(dirpath)
        for album in albums:
            # imgs_album = os.listdir(os.path.join(dirpath, album))
            imgs_album = os.listdir(os.path.join(dir_cls, album))
            # impaths.append([os.path.join(album, img) for img in imgs_album])
            labels_album = [cls]*len(imgs_album)
            impaths.append([os.path.join(cls, album, img) for img in imgs_album])
            labels.append(labels_album)


    # from collections import Counter, defaultdict, OrderedDict

    impaths = list(itertools.chain.from_iterable(impaths))  # flatten
    labels = list(itertools.chain.from_iterable(labels))  # flatten
    # classes_idx_set =  Counter(labels)
    # classes_idx = OrderedDict(classes_idx_set)
    # [args.class_to_idx[lbl] for lbl in labels]

    # labels_str = np.unique(labels)
    # classes_to_idx = {}
    # for i, cls in enumerate(labels_str):
    #     classes_to_idx[cls] = i

    # if not(args is None) and hasattr(args, 'num_classes'):
    #     num_cls = args.num_classes
    # else:
    #     num_cls = len(labels_str)
    # # labels_idx = [classes_to_idx[lbl]  for lbl in labels]

    lbls = []
    for lbl in labels:
        labels_onehot = args.num_classes * [0]
        labels_onehot[args.class_to_idx[lbl]] = 1
        lbls.append(labels_onehot)

    return impaths, lbls

class DatasetFromList(data.Dataset):
    """From List dataset."""

    def __init__(self, root, impaths=None, labels=None, scores=None,
                 transform=None, target_transform=None,
                 loader=default_loader, args=None):
        """
        Args:

            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if (impaths is None):
            impaths, labels = get_impaths_from_dir(root, args)

        self.root = root
        # self.classes = idx_to_class
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.samples = tuple(zip(impaths, labels))
        if not(scores is None):
            self.scores = scores


    def __getitem__(self, index):
        impath, target = self.samples[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform([target])
        return img, target

    def __len__(self):
        return len(self.samples)