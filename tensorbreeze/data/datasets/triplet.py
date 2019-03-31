"""
Dataset for general triplet training based tasks
"""

from __future__ import absolute_import
from __future__ import division

from six import integer_types

import os
import glob
import random

from PIL import Image

import torch.utils.data
from torchvision.transforms import Resize


def _find_image_files(root):
    class_sorted_files = {}
    all_files = glob.glob(os.path.join(root, '*', '*.[jp][pn]g'))
    for f in all_files:
        c = f.split(os.sep)[-2]
        if c in class_sorted_files:
            class_sorted_files[c].append(f)
        else:
            class_sorted_files[c] = [f]
    for k in class_sorted_files.keys():
        class_sorted_files[k].sort()
    return class_sorted_files


class TripletDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        num_iter=100,
        transform=None
    ):
        """
        Write documentations for this class
        """
        assert os.path.isdir(root), 'root path does not exist'
        assert isinstance(num_iter, integer_types), 'num_iter has to be an int'

        self.root = root
        self.num_iter = int(num_iter)

        if transform is not None:
            self.transform = transform

        # Retrieve classes and
        self.image_files = _find_image_files(self.root)
        self.classes = sorted(self.image_files.keys())

    def __len__(self):
        return self.num_iter

    def __getitem__(self, idx):
        # Choose 2 classes
        p_class, n_class = random.sample(self.classes, 2)

        # Choose anchor, positive and negative
        a_file, p_file = random.sample(self.image_files[p_class], 2)
        n_file = random.choice(self.image_files[n_class])

        # Read files and resize
        filenames = [a_file, p_file, n_file]
        images = [Image.open(f) for f in filenames]

        # Apply transform if needed
        if hasattr(self, 'transform'):
            images = [self.transform(i) for i in images]

        return images, filenames
