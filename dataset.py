import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class PixelDataset(Dataset):
    def __init__(
        self, data_path, df_path, tasks, transform=None, target_transform=None
    ):
        self.transform = transform
        self.target_transform = target_transform

        # Prepare the data
        df = pd.read_csv(df_path)
        self.img_paths = df.iloc[:, 0].apply(
            lambda x: os.path.join(data_path, x)
        )
        self.labels = df[tasks]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = torch.FloatTensor(self.labels.loc[idx].values)
        img = Image.open(self.img_paths.loc[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label
