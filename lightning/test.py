import os
from typing import Any, Optional
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim, nn, utils, Tensor
import wandb, gc
import torch
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from thermostability.autoencoder import Encoder
from thermostability.hotinfer_pregenerated import HotInferPregeneratedFC
from torch.utils.data import DataLoader
import lightning.pytorch.callbacks as cb
from thermostability.autoencoder_dataset import (
    AutoEncoderDataset,
    zero_padding700_collate,
)
from pytorch_lightning.loggers import WandbLogger

from util.train_helper import calculate_metrics

gc.collect()
torch.cuda.empty_cache()
class HotProtWithAutoEncoder(pl.LightningModule):
    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = encoder
        self.thermo_module = HotInferPregeneratedFC(input_len=1408)
        self.epoch_predictions = []
        self.epoch_labels = []
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
       
        with torch.no_grad():
            z = self.encoder(inputs)
        prediction = self.thermo_module(z)
        loss = nn.functional.mse_loss(prediction, labels)
        self.log("loss", loss)

        self.epoch_labels.append(labels)
        self.epoch_predictions.append(prediction)
        return loss

   
    def validation_step(self, batch, batch_idx):
        inputs, labels= batch
      
        with torch.no_grad():
            z = self.encoder(inputs)
        prediction = self.thermo_module(z)
        loss = nn.functional.mse_loss(prediction, labels)
        self.log("val_loss", loss)
        return prediction


    def configure_optimizers(self):
        optimizer = optim.Adam(self.thermo_module.parameters(), lr=1e-5 )
        return optimizer
    
    def on_train_epoch_end(self) -> None:
        self.epoch_predictions = torch.cat(self.epoch_predictions)
        self.epoch_labels = torch.cat(self.epoch_labels)
        metrics = calculate_metrics(self.epoch_predictions, self.epoch_labels, "s_s")
        self.log_dict(metrics)
        self.epoch_predictions = []
        self.epoch_labels = []

seed_everything(42, workers=True)
train_ds = AutoEncoderDataset(dataset_filepath="data/train.csv")
val_ds = AutoEncoderDataset(dataset_filepath="data/val.csv")
test_ds = AutoEncoderDataset(dataset_filepath="data/test.csv")
dataloaders = {
    "train": DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=zero_padding700_collate,
    ),
    "val": DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=zero_padding700_collate,
    ),
    "test": DataLoader(
        test_ds,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=zero_padding700_collate,
    )
}
# # TODO: load pretrained model in
wandb.init(project="autoencoder", entity="hotprot")
autoencoder = torch.load("results/trash_autoencoder.pt")
wandb_logger = WandbLogger(project="autoencoder", log_model=True)
model = HotProtWithAutoEncoder(encoder=autoencoder.encoder)
wandb_logger.watch(model)
trainer = pl.Trainer( accelerator="auto", min_epochs=50, max_epochs=50, logger=wandb_logger, strategy='ddp_find_unused_parameters_true')
trainer.fit(model, train_dataloaders=dataloaders["train"], val_dataloaders= dataloaders["val"])