import torch
from torch import Tensor
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
import time
from tqdm import tqdm
from typing import Tuple
from utils import print_out
import numpy as np


def train_one_epoch(
    model: nn.Module,
    trainloader: DataLoader,
    device,  ## device: default "cpu",
    optimizer,  ### SGD: rule update theta: theta_new= theta_old - learning rate * gradient
    criterion: torch.nn.CrossEntropyLoss,
    debug: bool = False,
):
    """
    Train 1 epoch on trainloader
    """

    ## Set model to "train" model
    model.train()

    ## Keep track of loss and accuracy
    train_loss = 0.0

    mapper = {"0": "40X", "1": "100X", "2": "200X", "3": "400X"}
    train_acc = {"40X": 0.0, "100X": 0.0, "200X": 0.0, "400X": 0.0}
    num_imgs = {"40X": 0, "100X": 0, "200X": 0, "400X": 0}
    num_imgs_all = 0
    train_acc_all = 0

    n_batches = len(trainloader)
    ## Loop over all the batches
    for i, (images, labels, magnifications) in tqdm(
        enumerate(trainloader, 1), total=len(trainloader), desc=f"training 1 epoch..."
    ):  # train one batch (images, labels of 1 batch)
        # For each batch, we have:
        #     + `images`: `bath_size` images in training set
        #     + `labels`: labels of the images (`batch_size` labels)

        ## Reshape the input dimension if we use MLP: instead of 3d (num_channels, width, height),
        # we flatten it to 1d (num_channels * width * height)
        # For CNN, we don't need to do this because CNN can handle 3d input
        # breakpoint()
        if debug:
            print("debug!!!!")
            if i > 2:
                break
        # if flatten:
        #     images = images.reshape(images.shape[0], -1)
        # breakpoint()
        ## Move images, labels and magnifications to `device` (CPU or GPU)
        images = images.to(device)
        labels = labels.to(device)
        magnifications = magnifications.to(device)

        # Write your code in this block -------------------------------------------------------------------------------------------
        ## We use 5 following steps to train a Pytorch model

        ##### [YOUR CODE] Step 1. Forward pass: pass the data forward, the model try its best to predict what the output should be
        # You need to get the output from the model, store in a new variable named `logits`
        logits = model(images)  ## call forward funtion from class FirstNeuralNet

        ##### Step 2. Compare the output that the model gives us with the real labels
        ## You need to compute the loss, store in a new variable named `loss`
        loss = criterion(logits, labels)

        ##### Step 3. Clear the gradient buffer (because Pytorch accumulates gradients by default, so we need to clear the old gradients before computing the gradients of the current batch)
        optimizer.zero_grad()

        ##### Step 4. Backward pass: compute the gradients of the loss w.r.t parameters using backpropagation
        loss.backward()

        ##### Step 5. Update the parameters by stepping in the opposite direction of the gradient
        optimizer.step()

        # End of your code --------------------------------------------------------------------------------------------------------
        ## Compute loss and accuracy for this batch
        train_loss += (
            loss.detach().item()
        )  ## loss.detach().item(): loss of 1 batch.This line accumulates the loss for the current mini-batch in a variable called train_loss. The detach() method is used to detach the loss from the computation graph, so that we don't compute gradients for the loss itself.

        ## compute eval_acc based on magnification

        for i in range(4):
            logits_i = logits[magnifications == i]
            labels_i = labels[magnifications == i]
            batch_size_i = len(logits_i)
            if batch_size_i == 0:
                continue
            magnif = mapper[str(i)]
            num_imgs[magnif] += batch_size_i
            predict_i = torch.max(logits_i, 1)[1].view(labels_i.size()).data
            correct_i = (predict_i == labels_i.data).sum()
            train_acc[magnif] += correct_i.item()

            num_imgs_all += batch_size_i
            train_acc_all += correct_i.item()

    train_acc = {k: round(v * 100 / num_imgs[k], 2) for k, v in train_acc.items()}
    train_loss = train_loss / n_batches
    # compute average aval_acc
    train_acc["avg_acc"] = round(np.mean(list(train_acc.values())), 2)
    train_acc["all_acc"] = round(train_acc_all * 100 / num_imgs_all, 2)

    return train_acc, train_loss
