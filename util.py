import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


def bb_intersection_over_union(boxA, boxB):
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


def train(model, device, train_loader, loss_fn, optimizer, epoch, test_loss, precision, recall, accuracy, max_epochs):
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
                label_smoothing_loss(output_c, target_c)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(
                {
                    'Train loss (in progress)': loss.item(),
                    'Precision (test set)': precision,
                    'Recall (test set)': recall,
                    'Loss (test set)': test_loss,
                    'Accuracy (test set)': accuracy
                }
            )

    mean_loss = np.mean(losses)
    return losses, mean_loss


def test(model, device, loss_fn, test_loader, threshold=.5):
    model.eval()
    test_loss = 0
    true_positive = 0
    false_negative = 0
    accuracy = 0
    losses = []
    correct = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target_c, target = data.float().to(
                device), target[1].to(device), target[0].float().to(device)
            output, output_c = model(data)
            # get the loss
            loss = loss_fn(output, target) + \
                label_smoothing_loss(output_c, target_c)
            # get the index of the max log-probability
            pred = output_c.argmax(
                dim=1, keepdim=True
            )
            correct += pred.eq(target_c.view_as(pred)).sum().item()
            # Calculating precision and recall
            for bb1, bb2 in zip(output, target):
                IoU = bb_intersection_over_union(bb1, bb2)
                if IoU > 0.5:
                    true_positive += 1
                else:
                    false_negative += 1
            #
            losses.append(loss.item())

    test_loss = np.mean(losses)
    precision = true_positive / len(test_loader.dataset)
    recall = true_positive / (true_positive + false_negative)
    accuracy = correct / len(test_loader.dataset)

    return test_loss, precision, recall, accuracy


def label_smoothing_loss(y_hat, y, alpha=0.2):
    xent = F.nll_loss(y_hat, y, reduction="none")
    KL = -y_hat.mean(dim=1)
    loss = (1 - alpha) * xent + alpha * KL
    return loss.sum()
