import torch
import lightning.pytorch as pl
import torch.utils.data as data
import torchvision.transforms as T
from dataset import PixelDataset


class PixelDataModule(pl.LightningDataModule):
    def __init__(
        self, tasks, data_path, train_df_path, val_df_path, batch_size=64
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.tasks = tasks

        # Create the DataFrame paths
        self.train_df_path = train_df_path
        self.val_df_path = val_df_path

        # Setup the transforms
        self.train_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                T.Resize((320, 320), antialias=True),
                T.RandomRotation(15),
                T.RandomResizedCrop((290, 290), (0.8, 1.2), antialias=True),
                # T.Lambda(PixelDataModule._sort_values),
                # T.Lambda(PixelDataModule._round_values),
            ]
        )
        self.train_target_transform = None
        self.val_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                T.Resize((320, 320), antialias=True),
                # T.Lambda(PixelDataModule._sort_values),
                # T.Lambda(PixelDataModule._round_values),
            ]
        )
        self.val_target_transform = None

    def train_dataloader(self):
        train_dataset = PixelDataset(
            self.data_path,
            self.train_df_path,
            self.tasks,
            self.train_transform,
            self.train_target_transform,
        )
        return data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=10,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        val_dataset = PixelDataset(
            self.data_path,
            self.val_df_path,
            self.tasks,
            self.val_transform,
            self.val_target_transform,
        )
        return data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=10,
            shuffle=False,
        )

    def _sort_values(tensor):
        sorted_tensor, _ = tensor.sort()
        return sorted_tensor

    def _round_values(tensor):
        return torch.round(tensor / 0.05) * 0.05
