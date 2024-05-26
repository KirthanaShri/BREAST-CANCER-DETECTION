import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.io as io
from PIL import Image


class BreastCancer(Dataset):
    def __init__(self, csv_path: str, split: str, transform=None):
        """
        root_dir (string): Directory with all the images
        transform (callable, optional): Optional transform to be applied on a sample
        """
        self.transform = transform
        self.csv_path = csv_path
        df = pd.read_csv(csv_path)

        if split not in ["train", "valid", "test"]:
            raise ValueError("split must be one of: train, valid, test")
        self.df = df[df["split"] == split]

        self.class_map = {"benign": 0, "malignant": 1}
        self.magnification_map = {"40X": 0, "100X": 1, "200X": 2, "400X": 3}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        data_row = self.df.iloc[idx]
        img_path = data_row["img_path"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = data_row["label"]
        label = self.class_map[label]
        magnification = data_row["magnification"]
        magnification = self.magnification_map[magnification]

        return image, label, magnification  # always return numbers
