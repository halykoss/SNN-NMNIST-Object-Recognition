import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import plotly.express as px
import pandas as pd

import tonic
import tonic.transforms as transforms

from VNMNISTDataset import VNMNISTDataset
from model import Model, ConvNet
from util import train, test

import os
from argparse import ArgumentParser


torch.manual_seed(1234)

dirName = 'plots/'
if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("Directory ", dirName,  " Created!")

parser = ArgumentParser(
    description='SNN NMNIST Object Detection & Classification')
parser.add_argument(
    "--n-frames", help="[TONIC] number of frames", default=12, type=int)
parser.add_argument("--img-dim", help="[DATASET] image dimension (w=h)",
                    default=84, type=int)
parser.add_argument("--resize-max", help="[DATASET] Max resize image dim",
                    default=42, type=int)
parser.add_argument("--resize-min", help="[DATASET] Min resize image dim",
                    default=34, type=int)
parser.add_argument(
    "--lr", help="[NETWORK] learning rate", default=0.002, type=float)
parser.add_argument(
    "--epochs", help="[NETWORK] number of epochs", default=5, type=int)
parser.add_argument(
    "--input-ly", help="[NETWORK] Input after convolutions", default=3430, type=int)
parser.add_argument(
    "--hidden-ly", help="[NETWORK] Size of the hidden layer", default=500, type=int)
parser.add_argument(
    "--batch-size", help="[NETWORK] batch size", default=64, type=int)
parser.add_argument("--threshold", help="[Detection] IoU threshold",
                    default=.5, type=float)

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

DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

print("Model to {}...".format(DEVICE), end=" ")

model = Model(
    snn=ConvNet(
        input_features=args.input_ly,
        hidden_features=args.hidden_ly,
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
precisions = []
recalls = []
# test(model, DEVICE, loss_fn, test_dataloader)
test_loss, precision, recall, accuracy = 0.0, 0.0, 0.0, 0.0

test_loss, precision, recall, accuracy = test(
    model, DEVICE, loss_fn, test_dataloader, threshold=args.threshold)

for epoch in range(args.epochs):
    training_loss, mean_loss = train(model, DEVICE, train_dataloader, loss_fn, optimizer, epoch, test_loss, precision,
                                     recall, accuracy, max_epochs=args.epochs)
    test_loss, precision, recall, accuracy = test(
        model, DEVICE, loss_fn, test_dataloader, threshold=args.threshold)
    training_losses += training_loss
    mean_losses.append(mean_loss)
    test_losses.append(test_loss)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)

df = pd.DataFrame(training_losses, columns=['Training loss'])
fig = px.line(df, markers=False)
fig.write_html('plots/training_losses.html', auto_open=False)

df = pd.DataFrame(
    {'Mean Train Loss': mean_losses,
     'Mean Test Loss': test_losses
     })
fig = px.line(df, markers=False)
fig.write_html('plots/train-test.html', auto_open=False)

df = pd.DataFrame(
    {'Accuracy': accuracies,
     'Precision': precisions,
     'Recall': recalls
     })
fig = px.line(df, markers=False)
fig.write_html('plots/apr.html', auto_open=False)

print(df)
