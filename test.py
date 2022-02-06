from random import randrange
import imageio
import cv2
from torch.utils.data import DataLoader
from VNMNISTDataset import VNMNISTDataset
from model import ConvNet, Model

import torch
import numpy as np

import os
import tonic
import tonic.transforms as transforms


from argparse import ArgumentParser

torch.manual_seed(1234)

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

args = parser.parse_args()


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


def plot_with_box(frames, label, label_hat, value, with_bb=True, img_name='video.gif'):
    list_frames = []
    label = label[0]
    for img in range(len(frames)):
        float_img = frames[img][1] - frames[img][0]
        try:
            float_img = (float_img - np.min(float_img)) / \
                (np.max(float_img) - np.min(float_img))
        except:
            float_img = 0
        im = np.array(float_img * 255, dtype=np.uint8)
        threshed = cv2.adaptiveThreshold(
            im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
        img_rgb = cv2.cvtColor(threshed, cv2.COLOR_GRAY2RGB)
        start_point = (int(label[0]), int(label[1]))
        end_point = (int(label[2]), int(label[3]))
        start_point_hat = (int(label_hat[0]), int(label_hat[1]))
        end_point_hat = (int(label_hat[2]), int(label_hat[3]))
        color = (0, 128, 0)
        color_hat = (0, 0, 255)
        if with_bb:
            cv2.rectangle(img_rgb, start_point_hat,
                          end_point_hat, color_hat, 1)
            cv2.rectangle(img_rgb, start_point, end_point, color, 1)
            displace_y = (
                int(label[1])-3) if (int(label[1]) - 10 > 0) else (int(label[3]) + 10)
            cv2.putText(img_rgb, 'Pred.: ' + str(value), (int(label[0]), displace_y),
                        cv2.QT_FONT_NORMAL, .3, (36, 255, 12), 1, cv2.LINE_AA)
        list_frames.append(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))

    imageio.mimsave(img_name, list_frames, fps=10)


snn = ConvNet(
    input_features=10080,
    hidden_features=500,
    dt=0.01
)

DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

snn.load_state_dict(torch.load("results/snn.pth"), strict=False)

model = Model(
    snn=snn
).to(DEVICE)

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
