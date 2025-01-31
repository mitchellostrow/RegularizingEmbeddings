import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple
from dataclasses import dataclass
from omegaconf import DictConfig


class SequenceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: Tuple[TensorDataset, TensorDataset, TensorDataset],
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        """
        Args:
            datasets: Tuple of (train_dataset, val_dataset, test_dataset) as TensorDatasets
            batch_size: Batch size for all dataloaders
            num_workers: Number of workers for all dataloaders
            pin_memory: Whether to pin memory for all dataloaders
        """
        super().__init__()
        self.train_dataset, self.val_dataset, self.test_dataset = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def get_dataloader(self, dataset:TensorDataset, shuffle:bool=True) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
