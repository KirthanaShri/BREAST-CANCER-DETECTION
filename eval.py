import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


@torch.no_grad()  # decorator
def eval_model(
    model: nn.Module,
    device: torch.device,
    evalloader: DataLoader,
    criterion: torch.nn.CrossEntropyLoss,
    cm_name: str,
    roc_name: str,
    is_confusion_matrix: bool = True,
    is_auc: bool = True,
    
):

    ## Set model to "evaluate" model
    model.eval()

    ## Keep track of loss and accuracy
    eval_loss = 0.0
    mapper = {"0": "40X", "1": "100X", "2": "200X", "3": "400X"}
    eval_acc = {"40X": 0.0, "100X": 0.0, "200X": 0.0, "400X": 0.0}
    num_imgs = {"40X": 0, "100X": 0, "200X": 0, "400X": 0}
    all_logits = {"40X": [], "100X": [], "200X": [], "400X": []}
    all_labels = {"40X": [], "100X": [], "200X": [], "400X": []}
    num_imgs_all = 0
    eval_acc_all = 0

    ## Number of batches
    n_batches = len(evalloader)

    for i, (images, labels, magnifications) in enumerate(evalloader):
        # if images.shape != (4, 3, 20, 20):
        #     breakpoint()

        # if flatten:
        #     images = images.reshape(images.shape[0], -1)
            # images = torch.flatten(start_dim=1)

        ## Move images and labels to `device` (CPU or GPU)
        images = images.to(device)
        labels = labels.to(device)
        magnifications = magnifications.to(device)
        ##### [YOUR CODE] Step 1. Forward pass: pass the data forward, the model try its best to predict what the output should be
        # You need to get the output from the model, store in a new variable named `logits`
        logits = model(images)  ## call forward funtion from class FirstNeuralNet

        ##### [YOUR CODE] Step 2. Compare the output that the model gives us with the real labels
        ## You need to compute the loss, store in a new variable named `loss`
        loss = criterion(logits, labels)

        # End of your code --------------------------------------------------------------------------------------------------------
        ## Compute loss and accuracy for this batch
        eval_loss += loss.detach().item()

        #  compute eval_acc based on magnification
        for i in range(4):
            logits_i = logits[magnifications == i]
            labels_i = labels[magnifications == i]
            batch_size_i = len(logits_i)
            if batch_size_i == 0:
                continue
            magnif = mapper[str(i)]
            all_logits[magnif].extend(logits_i)
            all_labels[magnif].extend(labels_i)
        
            num_imgs[magnif] += batch_size_i
            predict_i = torch.max(logits_i, 1)[1].view(labels_i.size()).data
            correct_i = (predict_i == labels_i.data).sum()
            eval_acc[magnif] += correct_i.item()

            # compute eval_acc_all
            num_imgs_all += batch_size_i
            eval_acc_all += correct_i.item()

    eval_acc = {k: round(v * 100 / num_imgs[k], 2) for k, v in eval_acc.items()}
    eval_loss = eval_loss / n_batches
    # compute average aval_acc
    eval_acc["avg_acc"] = round(np.mean(list(eval_acc.values())), 2)
    eval_acc["all_acc"] = round(eval_acc_all * 100 / num_imgs_all, 2)

    ## Confusion matrix plot:
    if is_confusion_matrix:
        # Get predictions by magnification
        for i in range(4):
            logits_i = torch.stack(all_logits[mapper[str(i)]])
            labels_i = torch.stack(all_labels[mapper[str(i)]])
            preds = torch.max(logits_i, 1)[1].view(labels_i.size()).data
            # Plot confusion matrix
            cm = confusion_matrix(labels_i.cpu().numpy(), preds.cpu().numpy())
            plt.figure()
            sns.heatmap(cm, annot=True, fmt="d")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.show()
            plt.savefig(f'{cm_name.replace(".png", f"_{mapper[str(i)]}.png")}')
        
    ## ROC plot:
    if is_auc:
        ## plot 4 ROC curves in one plot
        plt.figure()
        for i in range(4):
            logits_i = torch.stack(all_logits[mapper[str(i)]])
            labels_i = torch.stack(all_labels[mapper[str(i)]])
            probs = torch.nn.functional.softmax(logits_i, dim=1)[: , 1]
            
            try:
                fpr, tpr, threshold = roc_curve(labels_i.cpu().numpy(), probs.cpu().numpy())
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{mapper[str(i)]}: (AUC = {roc_auc:.2f})')
            except:
                continue
            
            
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.show()
        plt.savefig(f"{roc_name}")

    


            
    return eval_acc, eval_loss
