import glob
import pandas as pd
import random
import os
import torchvision.transforms as transforms
import torch
import time
import matplotlib.pyplot as plt
import pprint
import yaml
import itertools
import re
import ast
from collections import defaultdict
from typing import List, Dict
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from typing import Tuple
from dataset import BreastCancer

from sklearn.model_selection import train_test_split


def create_dataset_csv(root_dir: str, seed: int = 2023):
    """
    create the csv table with columns:
    - img_path: file path to each image
    - label: main type of image (benign/malignant)
    - subtype: subtypes of benign and malignant tumors (A/F/PT/TA/MC/DC/LC/PC)
    - magnification: maginification of each image (40X/100X/200X/400X)
    - split: split the dataset into train:eval:test = 60:20:20

    """

    img_paths = find_images(root_dir)
    df = pd.DataFrame(img_paths, columns=["img_path"])
    df["label"] = df["img_path"].str.split("/", expand=True).iloc[:, 8]
    df["subtype"] = (
        df["img_path"]
        .str.split("/", expand=True)
        .iloc[:, 10]
        .str.split("_", expand=True)
        .iloc[:, 2]
    )
    df["magnification"] = df["img_path"].str.split("/", expand=True).iloc[:, 11]
    #### create a dataframe consists of 4 cols: img_path, label, magnification, subtype and split (train/eval/test)    ## e.g.,
    ##   img_path                                        label           subtype     magnification    split
    ## '/.../SOB_M_MC-14-13418DE-100-009.png'           benign              MC             100X       train
    ## '/.../SOB_M_MC-14-13418DE-100-008.png'           benign              MC             100X       train
    ## '/.../SOB_M_MC-14-13418DE-100-003.png'           benign              MC             100X       train

    ## split the dataset into train:eval:test = 60:20:20, stratified by magnification
    train, valid_test = train_test_split(
        df, test_size=0.4, stratify=df["magnification"], random_state=seed
    )
    valid, test = train_test_split(
        valid_test, test_size=0.5, stratify=valid_test["magnification"], random_state=seed
    )
    train["split"] = "train"
    valid["split"] = "valid"
    test["split"] = "test"
    df = pd.concat([train, valid, test])

    return df


def find_images(directory, image_extensions=["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp"]):
    """
    Recursively finds all images in the directory with specified extensions.
    :param directory: The directory to search in.
    :param image_extensions: List of image file extensions to search for.
    :return: List of paths to the images found.
    """
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(directory, "**", extension), recursive=True))
    return image_paths


def compute_mean_std(loader):
    mean = 0.0
    for images, _, _ in tqdm(loader):
        #     print(images.shape) : 10, 3, 238, 374
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        #     print(images.shape): 10, 3, 89012
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    var = 0.0
    pixel_count = 0
    for images, _, _ in tqdm(loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
        pixel_count += images.nelement() / images.size(1)
    std = torch.sqrt(var / pixel_count)

    return list(mean.numpy()), list(std.numpy())

def get_dataloaders(batch_size: int, img_size: Tuple):
    ## prepare dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(img_size, antialias=True),
            transforms.Normalize((0.7844108, 0.6242002, 0.76210874), (0.122441396, 0.17505719, 0.10627644)),
        ]
    )

    trainset = BreastCancer(
        csv_path="breast_cancer_meta_data.csv",
        split="train",
        transform=transform,
    )
    validset = BreastCancer(
        csv_path="breast_cancer_meta_data.csv",
        split="valid",
        transform=transform,
    )
    testset = BreastCancer(
        csv_path="breast_cancer_meta_data.csv",
        split="test",
        transform=transform,
    )

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    validloader = DataLoader(
        validset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False
    )

    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False
    )
    return trainloader, validloader, testloader


def print_out(print_str, log):
    print(print_str)
    datetime_now = time.strftime("%Y-%m-%d_%H-%M-%S")
    ## append different models to log file:
    log.write(datetime_now + ": " + print_str + "\n")
    log.flush()


def plot_train_eval_summary(df_train_summary, df_eval_summary):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(df_train_summary["epoch"], df_train_summary["40X"], label="40X")
    ax1.plot(df_train_summary["epoch"], df_train_summary["100X"], label="100X")
    ax1.plot(df_train_summary["epoch"], df_train_summary["200X"], label="200X")
    ax1.plot(df_train_summary["epoch"], df_train_summary["400X"], label="400X")
    ax1.plot(
        df_train_summary["epoch"],
        df_train_summary["avg_acc"],
        label="Average Accuracy",
        color="black",
        linewidth=2,
        linestyle="--",
    )
    ax1.set_title("Training accuracy over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()

    ax2.plot(df_eval_summary["epoch"], df_eval_summary["40X"], label="40X")
    ax2.plot(df_eval_summary["epoch"], df_eval_summary["100X"], label="100X")
    ax2.plot(df_eval_summary["epoch"], df_eval_summary["200X"], label="200X")
    ax2.plot(df_eval_summary["epoch"], df_eval_summary["400X"], label="400X")
    ax2.plot(
        df_eval_summary["epoch"],
        df_eval_summary["avg_acc"],
        label="Average Accuracy",
        color="black",
        linewidth=2,
        linestyle="--",
    )
    ax2.set_title("Training accuracy over Epochs")
    ax2.set_title("Evaluation accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    plt.show()



def resnet_get_grid_search(config_path: str, 
                    optimizers: list, 
                    learning_rates: list,
                    num_blocks_list:list,
                    is_batchnorm:list) -> List[Dict]:
    """
    Read the config file and return a list of all possible combinations of hyperparameters
    """
    ## read the config file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    combinations = list(
        itertools.product(optimizers, learning_rates, num_blocks_list, is_batchnorm)
    )
    resnet_grid_search = []
    for opt, lr, num_blocks, bn in combinations:
        new_config = config.copy()
        new_config["optimizer"] = opt
        new_config["lr"] = lr
        new_config["num_blocks_list"] = num_blocks
        new_config["is_batchnorm"] = bn
        resnet_grid_search.append(new_config)

    return resnet_grid_search

def mlp_get_grid_search(config_path: str, 
                        optimizers: list,
                        learning_rates: list,
                        weight_decay:list) -> List[Dict]:
    """
    Read the config file and return a list of all possible combinations of hyperparameters
    """
    ## read the config file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    combinations = list(
        itertools.product(optimizers, learning_rates, weight_decay)
    )
    mlp_grid_search = []
    for opt, lr, reg in combinations:
        new_config = config.copy()
        new_config["optimizer"] = opt
        new_config["lr"]=lr
        new_config["weight_decay"] = reg
        mlp_grid_search.append(new_config)
    
    return mlp_grid_search

    
def cnn_get_grid_search(config_path: str,
                        optimizers: list,
                        learning_rates: list,
                        num_kernel_conv_list:list,
                        use_pooling:list)-> List[Dict]:


    """
    Read the config file and return a list of all possible combinations of hyperparameters
    """
    ## read the config file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    combinations = list(
        itertools.product(optimizers, learning_rates, num_kernel_conv_list, use_pooling)
    )
    cnn_grid_search = []
    for opt, lr, num_kernel_conv, pooling in combinations:
        new_config = config.copy()
        new_config["optimizer"] = opt
        new_config["lr"] = lr
        new_config["num_kernel_conv_list"] = num_kernel_conv
        new_config["use_pooling"] = pooling
        cnn_grid_search.append(new_config)
    return cnn_grid_search



def create_result_summary_dictionary(log_path: str) -> Dict:
    with open(log_path, 'r') as f:
        # Read the file
        log = f.read()

    # Extract the config dictionary string
    config_dict_str = re.search(r"config: ({.*})", log).group(1)
    # Convert the string to a config dictionary
    config = ast.literal_eval(config_dict_str)


    # Extract the total parameters
    try:
        total_parameters_match = re.search(r"Total parameters: ([\d,]+);", log)
        if total_parameters_match is not None:
                total_parameters = total_parameters_match.group(1)
        else:
            total_parameters_match = re.search(r"Total parameters: ([\d,]+)", log)
            if total_parameters_match is not None:
             total_parameters = total_parameters_match.group(1)
            else:
                total_parameters = "0"  # Default value if no match is found
    except AttributeError:
        total_parameters = "0"  # Default value if an error occurs

    # Remove commas and convert to integer
    total_parameters = int(total_parameters.replace(',', ''))
    config["total_parameters"] = total_parameters

    #Extract the final test accuracy dictionary string
    test_accuracy_dict_str = re.search(r"Final test accuracy: ({.*})", log)
    if test_accuracy_dict_str is not None:
        test_accuracy_dict_str = test_accuracy_dict_str.group(1)
        final_test_accuracy = ast.literal_eval(test_accuracy_dict_str)

        # Merge config and final_test_accuracy dictionaries
        config.update(final_test_accuracy)
        config["log_name"] = log_path.split("/")[-1]
        if config["log_name"].startswith("resnet"):
            config["is_batchnorm"] = True

        return config

def get_all_log_results():
    log_paths = glob.glob("logs/*.log")
    config_list = []
    for log_path in log_paths:
        config = create_result_summary_dictionary(log_path)
        if config is not None:
            config_list.append(config)
    df = pd.DataFrame(config_list)
    # df.drop_duplicates(inplace=True)
    df.sort_values(by=["avg_acc"], ascending=False, inplace=True)

    # df = df.loc[df.astype(str).drop_duplicates(subset=["lr","optimizer","num_blocks_list","is_batchnorm"], keep="first").index]
    df.reset_index(drop=True, inplace=True)
    df.to_csv("result_summary.csv")
    return df

if __name__ == "__main__":  ### put all test code in this block
    # log_path = "logs/v2_resnet_epoch30_lr0.0001_2023-12-03_18-15-17.log"
    # res = create_result_summary_dictionary(log_path = log_path)
    # print(res)

    # get all log files in logs folder
    get_all_log_results()
    



    # Open the file
    # path = "logs/resnet_epoch10_lr0.001_2023-12-02_14-27-14.log"

    # folder_path = "data_model/"
    # df = create_dataset_csv(folder_path)
    # print(df.sample(30))
    # df.to_csv("breast_cancer_meta_data.csv")
    # Calculate mean and std of trainloader
    # trainloader = get_dataloaders(batch_size=32, img_size=(224, 224))[0]
    # mean, std = compute_mean_std(trainloader) # get_mean_std(trainloader)
    # print(mean)
    # print(std)
    # pass