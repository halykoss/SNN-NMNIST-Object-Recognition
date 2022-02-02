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


def train(model, device, train_loader, loss_fn, optimizer, epoch, test_loss, precision, recall, max_epochs):
    model.train()
    losses = []
    with tqdm(train_loader) as pbar:
        for (data, target) in pbar:
            pbar.set_description(
                'Epoch {} / {} '.format(epoch + 1, max_epochs))
            data, target = data.float().to(device), torch.from_numpy(
                np.asarray(target)).float().to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(
                {
                    'Train loss (in progress)': loss.item(),
                    'Precision (test set)': precision,
                    'Recall (test set)': recall,
                    'Loss (test set)': test_loss
                }
            )

    mean_loss = np.mean(losses)
    return losses, mean_loss


def test(model, device, loss_fn, test_loader):
    model.eval()
    test_loss = 0
    true_positive = 0
    false_negative = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.float().to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(
                output, target
            )  # sum up batch loss
            for bb1, bb2 in zip(output, target):
                IoU = bb_intersection_over_union(bb1, bb2)
                if IoU > 0.5:
                    true_positive += 1
                else:
                    false_negative += 1

    test_loss /= len(test_loader.dataset)
    precision = true_positive / len(test_loader.dataset)
    recall = true_positive / (true_positive + false_negative)
    #accuracy = 100.0 * correct / len(test_loader.dataset)

    return test_loss, precision, recall


def label_smoothing_loss(y_hat, y, alpha=0.2):
    log_probs = F.log_softmax(y_hat, dim=1)
    xent = F.nll_loss(log_probs, y, reduction="none")
    KL = -log_probs.mean(dim=1)
    loss = (1 - alpha) * xent + alpha * KL
    return loss.sum()