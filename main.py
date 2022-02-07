from random import randrange
from torch.utils.data import DataLoader
from VNMNISTDataset import VNMNISTDataset
from util import plot_with_one_box

import os
import torch
import tonic
import tonic.transforms as transforms

from argparse import ArgumentParser

torch.manual_seed(1234)

# Where results will be stored
dirName = 'examples/'
if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("Directory ", dirName,  " Created!")

parser = ArgumentParser(
    description='Produce NMNIST larger frame examples')

parser.add_argument(
    "--n-frames", help="[TONIC] number of frames", default=12, type=int)
parser.add_argument("--img-dim", help="image dimension (w=h)",
                    default=64, type=int)
parser.add_argument("--batch-size", help="batch size", default=64, type=int)
parser.add_argument("--resize-max", help="Max resize image dim",
                    default=42, type=int)
parser.add_argument("--resize-min", help="Min resize image dim",
                    default=34, type=int)

args = parser.parse_args()

# Regarding dataset generation

sensor_size = tonic.datasets.NMNIST.sensor_size

denoise_transform = tonic.transforms.Denoise(filter_time=10000)
frame_transform = transforms.ToFrame(
    sensor_size=sensor_size, n_time_bins=args.n_frames)

transform = transforms.Compose([denoise_transform, frame_transform])

train_set = VNMNISTDataset(
    train=True,
    transform=transform,
    dim=(args.img_dim, args.img_dim),
    resize=(args.resize_max, args.resize_min)
)

test_set = VNMNISTDataset(
    train=False,
    transform=transform,
    dim=(args.img_dim, args.img_dim),
    resize=(args.resize_max, args.resize_min)
)

train_dataloader = DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True)

# Generating 100 random examples
for i in range(100):
    idx = randrange(60000)
    imgs, label = train_set[idx]
    plot_with_one_box(imgs, label, label[1], with_bb=True,
                      img_name='./examples/ex-' + str(i) + '.gif')

print("Done!")
