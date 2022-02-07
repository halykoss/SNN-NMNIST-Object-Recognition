'''
Here you can find some useful functions
used in the project
'''

import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import cv2
import imageio


def bb_intersection_over_union(boxA, boxB):
    '''
    This function calculate the IoU between two boxes
    '''
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def plot_with_box(frames, label, label_hat, value, with_bb=True, img_name='video.gif'):
    '''
    This function is used for making a video of the spiking sequence, two bb are printed
    '''
    list_frames = []
    label = label[0]
    for img in range(len(frames)):
        float_img = frames[img][1] - frames[img][0]
        # Moving everything to the RGB model
        try:
            float_img = (float_img - np.min(float_img)) / \
                (np.max(float_img) - np.min(float_img))
        except:
            float_img = 0
        im = np.array(float_img * 255, dtype=np.uint8)
        threshed = cv2.adaptiveThreshold(
            im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
        img_rgb = cv2.cvtColor(threshed, cv2.COLOR_GRAY2RGB)
        # Getting the bounding boxes
        start_point = (int(label[0]), int(label[1]))
        end_point = (int(label[2]), int(label[3]))
        start_point_hat = (int(label_hat[0]), int(label_hat[1]))
        end_point_hat = (int(label_hat[2]), int(label_hat[3]))
        # Color for the original bounding box
        color = (0, 128, 0)
        # Color for the pWredicted bounding box
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
    # Producing the GIF
    imageio.mimsave(img_name, list_frames, fps=10)


def train(model, device, train_loader, loss_fn, loss_fn_c, optimizer, epoch, test_loss, precision, topk, accuracy, max_epochs):
    '''
    This function is used for training over one epoch
    '''
    model.train()
    losses = []
    with tqdm(train_loader) as pbar:
        for (data, target) in pbar:
            pbar.set_description(
                'Epoch {} / {} '.format(epoch + 1, max_epochs))
            data, target_c, target = data.float().to(
                device), target[1].to(device), target[0].float().to(device)
            optimizer.zero_grad()
            output, output_c = model(data)
            loss = loss_fn(output, target) + \
                loss_fn_c(output_c, target_c)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(
                {
                    'Train loss (in progress)': loss.item(),
                    'Precision (test set)': precision,
                    'Top-3 accuracy (test set)': topk,
                    'Loss (test set)': test_loss,
                    'Accuracy (test set)': accuracy
                }
            )

    mean_loss = np.mean(losses)
    return losses, mean_loss


def test(model, device, loss_fn, loss_fn_c, test_loader, threshold=.5):
    '''
    This function is used for evaluation
    '''
    model.eval()
    test_loss = 0
    true_positive = 0
    false_negative = 0
    accuracy = 0
    losses = []
    correct = 0
    correct_k = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target_c, target = data.float().to(
                device), target[1].to(device), target[0].float().to(device)
            output, output_c = model(data)
            # get the loss
            loss = loss_fn(output, target) + \
                loss_fn_c(output_c, target_c)
            # get the index of the max log-probability
            pred = output_c.argmax(
                dim=1, keepdim=True
            )
            _, tk = output_c.topk(3, dim=1)

            correct_k += sum(1 for o, p in zip(target_c, tk)
                             if o.cpu().numpy() in p.cpu().numpy())

            correct += pred.eq(target_c.view_as(pred)).sum().item()
            # Calculating precision and recall
            for bb1, bb2 in zip(output, target):
                IoU = bb_intersection_over_union(bb1, bb2)
                if IoU > threshold:
                    true_positive += 1
                else:
                    false_negative += 1
            #
            losses.append(loss.item())

    test_loss = np.mean(losses)
    precision = true_positive / len(test_loader.dataset)
    topk = correct_k / len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    return test_loss, precision, topk, accuracy


def label_smoothing_loss(y_hat, y, alpha=0.1):
    '''

    Label smoothing:

        - Turns “hard” class label assignments to “soft” label assignments.
        - Operates directly on the labels themselves.
        - Can lead to a model that generalizes better.

    '''
    xent = F.nll_loss(y_hat, y, reduction="none")
    KL = -y_hat.mean(dim=1)
    loss = (1 - alpha) * xent + alpha * KL
    return loss.sum()


def plot_with_one_box(frames, label, value, with_bb=True, img_name='video.gif'):
    '''
    This function is used for making a video of the spiking sequence, only one bb is printed
    '''
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
        color = (0, 0, 255)
        if with_bb:
            cv2.rectangle(img_rgb, start_point, end_point, color, 1)
            displace_y = (
                int(label[1])-3) if (int(label[1]) - 10 > 0) else (int(label[3]) + 10)
            cv2.putText(img_rgb, 'Val.: ' + str(value), (int(label[0]), displace_y),
                        cv2.QT_FONT_NORMAL, .3, color, 1, cv2.LINE_AA)
        list_frames.append(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))

    imageio.mimsave(img_name, list_frames, fps=10)
