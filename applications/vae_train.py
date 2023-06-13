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
from util.telegram import TelegramBot
train_ds = AutoEncoderDataset(dataset_filepath="data/train.csv")
val_ds = AutoEncoderDataset(dataset_filepath="data/val.csv")
test_ds = AutoEncoderDataset(dataset_filepath="data/test.csv")

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
telegram = TelegramBot()
def should_stop(val_epoch_losses: "list[float]"):
    if len(val_epoch_losses) < 3:
        return False

    has_improved = (
        val_epoch_losses[-2] < val_epoch_losses[-3]
        or val_epoch_losses[-1] < val_epoch_losses[-3]
    )
    return not has_improved


model = AutoEncoder()

loss_criterion = lambda out , label : torch.nn.MSELoss()(out,label).to("cuda:0") + model.encoder.kl
weight_decay = 1e-5

criterions = {
    "train": loss_criterion,
    "val": loss_criterion,
    "test": loss_criterion,
}
optimizer_ft = torch.optim.Adam(
    model.parameters(), lr=0.00001, weight_decay=weight_decay
)
scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=12, gamma=0.5)


optimizer = scheduler.optimizer

train_model(
    model=model,
    scheduler=scheduler,
    dataloaders=dataloaders,
    criterions=criterions,
    num_epochs=30,
    should_stop=should_stop,
    best_model_path="results/autoencoder.pt",
    use_wandb=False,
    prepare_inputs= lambda x: x.to("cuda:0"),
    prepare_labels= lambda x: x.to("cuda:1"),
)
