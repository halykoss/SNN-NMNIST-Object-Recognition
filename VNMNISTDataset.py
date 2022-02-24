from numpy import random
from torch.utils.data import Dataset
import tonic
import numpy as np
import cv2
import torch
import os

class VNMNISTDataset(Dataset):
    '''
    This dataset augment the NMNIST frame resolution and place the number in
    some place of the space. A lazy approach has been choosen for solving 
    this task. An element is generated when - and only when -  it is required. 
    Starting image can be even randomly resized. 
    '''

    def __init__(self, train=True, transform=None, dim=(50, 50), resize=42, mult=1.3):
        self.dataset = tonic.datasets.NMNIST(save_to=os.path.abspath('data'),
                                             train=train,
                                             transform=transform)

        self.dim = int(mult * len(self.dataset))
        self.width, self.height = dim
        if resize:
            # How big each image should be
            self.img_size = np.random.randint(
                high=int(resize),
                low=int(34),
                size=self.dim
            )
        self.resize = resize
        # Where images will be place
        self.coord = {
            "x": random.randint(self.width, size=self.dim),  # - self.xs
            "y": random.randint(self.height, size=self.dim)  # - self.ys
        }

    def augment_frames(self, idx):
        frames, label = self.dataset[idx]
        size = self.img_size[idx] if self.resize else 34
        if self.resize:
            img_outter = []
            # Image resizing
            for frame in frames:
                img_inner = []
                for idx_img in [0, 1]:
                    resized = cv2.resize(
                        frame[idx_img], (size, size), interpolation=cv2.INTER_NEAREST)
                    img_inner.append(resized)
                img_outter.append(img_inner)
            frames = np.array(img_outter)
        # Constraint checking
        if self.coord["x"][idx] - size < 0:
            x = 0
        elif self.coord["x"][idx] - size >= self.width:
            x = self.width - size - 1
        else:
            x = self.coord["x"][idx] - size
        # More constraint checking
        if self.coord["y"][idx] - size < 0:
            y = 0
        elif self.coord["y"][idx] - size >= self.width:
            y = self.width - size - 1
        else:
            y = self.coord["y"][idx] - size

        # New image generation
        size_w = (x, self.width - size - x)
        size_h = (y, self.height - size - y)
        new_frames = np.pad(frames, ((0, 0), (0, 0), size_h, size_w), 'minimum')

        return new_frames, (torch.from_numpy(np.asarray([x, y, x + size, y + size])), label)

    def __len__(self):
        return self.dim

    def __getitem__(self, idx):
        return self.augment_frames(idx % len(self.dataset))
