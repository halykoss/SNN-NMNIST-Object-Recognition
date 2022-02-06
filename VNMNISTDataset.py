from numpy import random
from torch.utils.data import Dataset
# from skimage.transform import resize
import tonic
import numpy as np
import cv2
import torch


class VNMNISTDataset(Dataset):
    def __init__(self, train=True, transform=None, dim=(50, 50), resize=(42, 34), mult=1.3):
        self.dataset = tonic.datasets.NMNIST(save_to='./data',
                                             train=train,
                                             transform=transform)  # ,
        self.dim = int(mult * len(self.dataset))
        self.width, self.height = dim
        _, _, xs, ys = self.dataset[0][0].shape
        self.img_size = np.random.randint(
            high=int(resize[0]),
            low=int(resize[1]),
            size=self.dim
        )

        self.coord = {
            "x": random.randint(self.width, size=self.dim),  # - self.xs
            "y": random.randint(self.height, size=self.dim)  # - self.ys
        }

    def augment_frames(self, idx):
        frames, label = self.dataset[idx]
        size = self.img_size[idx]
        img_outter = []
        for frame in frames:
            img_inner = []
            for idx_img in [0, 1]:
                resized = cv2.resize(
                    frame[idx_img], (size, size), interpolation=cv2.INTER_NEAREST)
                img_inner.append(resized)
            img_outter.append(img_inner)
        frames = np.array(img_outter)

        if self.coord["x"][idx] - size < 0:
            x = 0
        elif self.coord["x"][idx] - size >= self.width:
            x = self.width - size - 1
        else:
            x = self.coord["x"][idx] - size

        if self.coord["y"][idx] - size < 0:
            y = 0
        elif self.coord["y"][idx] - size >= self.width:
            y = self.width - size - 1
        else:
            y = self.coord["y"][idx] - size

        size_w = (x, self.width - size - x)
        size_h = (y, self.height - size - y)
        frames = np.pad(frames, ((0, 0), (0, 0), size_h, size_w), 'minimum')

        return frames, (torch.from_numpy(np.asarray([x, y, x + size, y + size])), label)

    def __len__(self):
        return self.dim

    def __getitem__(self, idx):
        return self.augment_frames(idx % len(self.dataset))
