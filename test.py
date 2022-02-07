'''
This script is used for visualization over model results
'''

from VNMNISTDataset import VNMNISTDataset
from model import ConvNet, Model
from util import plot_with_box

import torch
from random import randrange
from torch.utils.data import DataLoader

import os
import tonic
import tonic.transforms as transforms

from argparse import ArgumentParser

torch.manual_seed(1234)

# Where results will be stored
dirName = 'generated/'
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
parser.add_argument(
    "--input-layer", help="[NETWORK] Input after convolutions", default=10080, type=int)
parser.add_argument(
    "--hidden-layer", help="[NETWORK] Size of the hidden layer", default=500, type=int)
parser.add_argument(
    "--hidden-layer-c", help="[NETWORK] Size of the hidden layer for classification", default=1000, type=int)

args = parser.parse_args()

# Regarding dataset
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

# Regarding model


snn = ConvNet(
    input_features=args.input_layer,
    hidden_features=args.hidden_layer,
    hidden_features_c=args.hidden_layer_c,
    dt=0.01
)

DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

snn.load_state_dict(torch.load("results/snn.pth"), strict=False)

model = Model(
    snn=snn
).to(DEVICE)

# Generating 100 random spiking sequence videos
for i in range(100):
    idx = randrange(60000)
    imgs, (label, _) = train_set[idx]
    new_label, pred_value = model(torch.tensor(
        imgs).unsqueeze(0).float().to(DEVICE))
    pred = pred_value.argmax(
        dim=1, keepdim=True
    ).cpu().numpy()[0][0]
    plot_with_box(imgs, new_label, label, pred, with_bb=True,
                  img_name='./generated/ex-' + str(i) + '.gif')

print("Done!")
