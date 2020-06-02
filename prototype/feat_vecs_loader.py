import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class FeatVecsDataset(Dataset):
    def __init__(self, feat_vecs, transform=None):
        self.targets = []  # DIM: Data count

        self.transform = transform
        self.target_transform = transforms.Compose(
            [transforms.ToTensor(), ])

        for class_label in range(feat_vecs.shape[0]):
            self.targets.extend(
                np.ones(feat_vecs[class_label].shape[0]) * class_label)

        self.vecs = np.reshape(feat_vecs, (-1, feat_vecs.shape[-1]))
        self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        vec, target = self.vecs[idx], self.targets[idx]

        # Transform should almost always be none
        if self.transform is not None:
            vec = self.transform(vec)

        return vec, target
