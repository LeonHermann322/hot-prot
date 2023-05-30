from torch import nn as nn
from typing import Callable
from thermostability.autoencoder_dataset import (
    AutoEncoderDataset,
    zero_padding700_collate,
)
from thermostability.autoencoder import AutoEncoder
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from util.train_helper import train_model

train_ds = AutoEncoderDataset(dataset_filepath="data/train.csv", limit=30)
val_ds = AutoEncoderDataset(dataset_filepath="data/val.csv", limit=10)
test_ds = AutoEncoderDataset(dataset_filepath="data/test.csv", limit=10)

dataloaders = {
    "train": DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=zero_padding700_collate,
    ),
    "val": DataLoader(
        val_ds,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=zero_padding700_collate,
    ),
    "test": DataLoader(
        test_ds,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=zero_padding700_collate,
    ),
}


def should_stop(val_epoch_losses: "list[float]"):
    if len(val_epoch_losses) < 3:
        return False

    has_improved = (
        val_epoch_losses[-2] < val_epoch_losses[-3]
        or val_epoch_losses[-1] < val_epoch_losses[-3]
    )
    return not has_improved


criterion = torch.nn.MSELoss()

model = AutoEncoder()

weight_decay = 1e-5

optimizer_ft = torch.optim.Adam(
    model.parameters(), lr=0.00001, weight_decay=weight_decay
)
scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=12, gamma=0.5)


optimizer = scheduler.optimizer

train_model(
    model=model,
    scheduler=scheduler,
    dataloaders=dataloaders,
    criterions=criterion,
    num_epochs=30,
    should_stop=should_stop,
    best_model_path="results/autoencoder.pt",
)
