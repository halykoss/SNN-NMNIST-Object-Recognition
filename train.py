'''

This script is used to train the SNN model

'''

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import plotly.express as px
import pandas as pd

import tonic
import tonic.transforms as transforms

from VNMNISTDataset import VNMNISTDataset
from model import Model, ConvNet
from util import label_smoothing_loss, train, test

import os
from argparse import ArgumentParser


torch.manual_seed(1234)

# Directory where files will be stored
dirName = 'results/'
if not os.path.exists(dirName):
    os.makedirs(dirName)
    print("Directory ", dirName,  " Created!")

parser = ArgumentParser(
    description='SNN NMNIST Object Detection & Classification')
parser.add_argument(
    "--n-frames", help="[TONIC] number of frames", default=12, type=int)
parser.add_argument("--img-dim", help="[DATASET] image dimension (w=h)",
                    default=84, type=int)
parser.add_argument('--no-resize', action='store_true',
                    help='[DATASET] No frame resize')
parser.add_argument('--random-noise', action='store_true',
                    help='[NETWORK] Sum random value on the voltages')
parser.add_argument("--resize-max", help="[DATASET] Max resize image dim",
                    default=42, type=int)
parser.add_argument("--resize-min", help="[DATASET] Min resize image dim",
                    default=34, type=int)
parser.add_argument("--dataset-aug", help="[DATASET] Dataset augmentation multiplicator",
                    default=1.3, type=float)
parser.add_argument(
    "--lr", help="[NETWORK] learning rate", default=0.002, type=float)
parser.add_argument(
    "--epochs", help="[NETWORK] number of epochs", default=5, type=int)
parser.add_argument(
    "--input-layer", help="[NETWORK] Input after convolutions", default=10080, type=int)
parser.add_argument(
    "--hidden-layer", help="[NETWORK] Size of the hidden layer", default=500, type=int)
parser.add_argument(
    "--hidden-layer-c", help="[NETWORK] Size of the hidden layer for classification", default=1000, type=int)
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

# Part regarding dataset
sensor_size = tonic.datasets.NMNIST.sensor_size

denoise_transform = tonic.transforms.Denoise(filter_time=10000)
frame_transform = transforms.ToFrame(
    sensor_size=sensor_size, n_time_bins=args.n_frames)

transform = transforms.Compose([denoise_transform, frame_transform])

resize = False if args.no_resize else args.resize_max

train_set = VNMNISTDataset(
    train=True,
    transform=transform,
    dim=(args.img_dim, args.img_dim),
    resize=resize,
    mult=args.dataset_aug
)

test_set = VNMNISTDataset(
    train=False,
    transform=transform,
    dim=(args.img_dim, args.img_dim),
    resize=resize,
    mult=args.dataset_aug
)

train_dataloader = DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(
    test_set, batch_size=args.batch_size, shuffle=True)

print("Done!")

# Part regarding the network

DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

print("Model to {}...".format(DEVICE), end=" ")

snn = ConvNet(
    args=args,
    input_features=args.input_layer,
    hidden_features=args.hidden_layer,
    hidden_features_c=args.hidden_layer_c,
    dt=0.01
)

model = Model(
    snn=snn
).to(DEVICE)


loss_fn = nn.MSELoss()
loss_fn_c = label_smoothing_loss

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

print("Done!")

training_losses = []
mean_losses = []
test_losses = []
accuracies = []
precisions = []
top_ks = []

test_loss, precision, topk, accuracy = 0, 0, 0, 0

# Training loop
for epoch in range(args.epochs):

    training_loss, mean_loss = train(
        model, DEVICE, train_dataloader,
        loss_fn, loss_fn_c, optimizer,
        epoch, test_loss, precision,
        topk, accuracy, max_epochs=args.epochs
    )

    test_loss, precision, topk, accuracy = test(
        model, DEVICE, loss_fn,
        loss_fn_c, test_dataloader, threshold=args.threshold
    )

    training_losses += training_loss

    mean_losses.append(mean_loss)
    test_losses.append(test_loss)
    accuracies.append(accuracy)
    precisions.append(precision)
    top_ks.append(topk)

    print("Mean train loss: {:.2f} , Test loss: {}".format(
        mean_loss, test_loss))
    print("Precision: {:.2f}".format(precision))
    print("Top-3 accuracy: {:.2f}".format(topk))
    print("Accuracy: {:.2f}".format(accuracy))

# Results plotting and formatting
df = pd.DataFrame(training_losses, columns=['Training loss'])
fig = px.line(df, markers=False)
fig.write_html(dirName + 'training_losses.html', auto_open=False)

df = pd.DataFrame(
    {'Mean Train Loss': mean_losses,
     'Mean Test Loss': test_losses
     })
fig = px.line(df, markers=False)
fig.write_html(dirName + 'train-test.html', auto_open=False)

df = pd.DataFrame(
    {'Accuracy': accuracies,
     'Precision': precisions,
     'Top-3 accuracy:': top_ks
     })
fig = px.line(df, markers=False)
fig.write_html(dirName + 'apr.html', auto_open=False)

# Saving the model
torch.save(model.state_dict(), dirName + 'model.pth')
torch.save(snn.state_dict(), dirName + 'snn.pth')

print(df)
