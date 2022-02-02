from random import randrange
import imageio
import cv2
from torch.utils.data import DataLoader
from VNMNISTDataset import VNMNISTDataset
import torch
import numpy as np

import tonic
import tonic.transforms as transforms


from argparse import ArgumentParser

torch.manual_seed(1234)

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


def plot_with_box(frames, label, with_bb=True, img_name='video.gif'):
    list_frames = []

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
        size_x = label[2] / 2
        size_y = label[3] / 2
        start_point = (int(label[0]), int(label[1]))
        end_point = (int(label[2]), int(label[3]))
        color = (0, 0, 255)
        if with_bb:
            cv2.rectangle(img_rgb, start_point, end_point, color, 1)
        list_frames.append(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))

    imageio.mimsave(img_name, list_frames, fps=10)


for i in range(100):
    idx = randrange(60000)
    imgs, label = train_set[idx]

    plot_with_box(imgs, label, with_bb=True,
                  img_name='./examples/ex-' + str(i) + '.gif')

# cv2_imshow(plot_with_box(imgs, label))
# with open('./examples/ex-' + str(i) +'.gif','rb') as f:
#     display(Image(data=f.read(), format='png'))
print("Done!")
