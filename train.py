import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import tonic
import tonic.transforms as transforms

from VNMNISTDataset import VNMNISTDataset
from model import Model, ConvNet
from util import train, test

from argparse import ArgumentParser


torch.manual_seed(1234)

parser = ArgumentParser(
    description='SNN NMNIST Object Detection & Classification')

parser.add_argument("--lr", help="learning rate", default=0.002, type=float)
parser.add_argument(
    "--n-frames", help="[TONIC] number of frames", default=12, type=int)
parser.add_argument("--img-dim", help="image dimension (w=h)",
                    default=84, type=int)
parser.add_argument("--batch-size", help="batch size", default=64, type=int)
parser.add_argument("--epochs", help="number of epochs", default=5, type=int)
parser.add_argument("--resize-max", help="Max resize image dim",
                    default=42, type=int)
parser.add_argument("--resize-min", help="Min resize image dim",
                    default=34, type=int)

args = parser.parse_args()

print('\n  ' + 'Options')
for k, v in vars(args).items():
    print('   ' * 2 + k + ': ' + str(v))
print("")


print("Loading the dataset...", end=' ')

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
test_dataloader = DataLoader(
    test_set, batch_size=args.batch_size, shuffle=True)

print("Done!")

INPUT_FEATURES = 3430
HIDDEN_FEATURES = 500
OUTPUT_FEATURES = 4


DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

print("Model to {}...".format(DEVICE), end=" ")

model = Model(
    snn=ConvNet(
        input_features=INPUT_FEATURES,
        hidden_features=HIDDEN_FEATURES,
        output_features=OUTPUT_FEATURES,
        dt=0.01
    )
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.MSELoss()

print("Done!")

training_losses = []
mean_losses = []
test_losses = []
accuracies = []

# test(model, DEVICE, loss_fn, test_dataloader)
test_loss, precision, recall = 0.0, 0.0, 0.0

for epoch in range(args.epochs):
    training_loss, mean_loss = train(model, DEVICE, train_dataloader, loss_fn, optimizer, epoch, test_loss, precision,
                                     recall, max_epochs=args.epochs)
    test_loss, precision, recall = test(
        model, DEVICE, loss_fn, test_dataloader)
    training_losses += training_loss
    mean_losses.append(mean_loss)
    test_losses.append(test_loss)
    accuracies.append((precision, recall))

print(
    f"final precision: {accuracies[-1][0]}, final recall: {accuracies[-1][1]}")
