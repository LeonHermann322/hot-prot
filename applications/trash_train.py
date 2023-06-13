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
from tqdm.notebook import tqdm
from util.telegram import TelegramBot
import sys
from matplotlib.pyplot import plot 
import matplotlib.pyplot as plt 


telegram = TelegramBot()
def should_stop(val_epoch_losses: "list[float]"):
    if len(val_epoch_losses) < 3:
        return False

    has_improved = (
        val_epoch_losses[-2] < val_epoch_losses[-3]
        or val_epoch_losses[-1] < val_epoch_losses[-3]
    )
    return not has_improved

def execute_epoch(
    model: nn.Module,
    criterion: nn.modules.loss._Loss,
    dataloader: torch.utils.data.DataLoader,
    prepare_inputs: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    prepare_labels: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    on_batch_done: Callable[
        [int, torch.Tensor, float, float], None
    ] = lambda idx, outputs, loss, running_mad: None,
    optimizer: torch.optim.Optimizer = None,
):
    losses = []
    running_loss = 0.0
    epoch_mad = 0.0
    # Iterate over data.
    for idx, inputss  in enumerate(dataloader):
        inputs = inputss[0]
        inputs = prepare_inputs(inputs)
        # zero the parameter gradients
        if optimizer:
            optimizer.zero_grad()
        outputs = model(inputs.to("cuda:0"))
        loss = criterion(outputs, inputs.to("cuda:1"))
        loss += model.encoder.kl.to("cuda:1")
        
        losses.append(loss.item())

        # statistics
        batch_loss = loss.item()

        running_loss += batch_loss

        running_mad = epoch_mad / (idx + 1)
        on_batch_done(idx, outputs, loss, running_mad)

    epoch_mad = epoch_mad / len(dataloader)
    epoch_loss = running_loss / len(dataloader)
    return (
        epoch_loss,
        epoch_mad,
    )


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

criterion = torch.nn.MSELoss()

model = AutoEncoder()

weight_decay = 1e-5

optimizer_ft = (
        torch.optim.Adam(
            model.parameters(), lr=0.00001, weight_decay=weight_decay
        )
)
scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=12, gamma=0.5)



optimizer = scheduler.optimizer


epoch_losses = {"train": [], "val": []}
best_epoch_predictions = torch.tensor([])
best_epoch_loss = sys.float_info.max

best_epoch_actuals = torch.tensor([])
epoch_mads = {"train": [], "val": []}
num_epochs = 40
telegram.send_telegram("Starting training trash_train...")
epoch_msg = telegram.send_telegram("Epoch")["result"]
phase_msg = telegram.send_telegram("train..")["result"]
try:
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:

            def on_batch_done(idx, outputs, loss, running_mad):
                if phase == "train":
                    if not torch.isnan(loss):
                        loss.backward()
                        
                        threshold = 10
                        for p in model.parameters():
                            if p.grad is not None:
                                if p.grad.norm() > threshold:
                                    torch.nn.utils.clip_grad_norm_(p, threshold)
                        optimizer.step()
                    if torch.isnan(loss).any():
                        print(f"Nan loss: {torch.isnan(loss)}| Loss: {loss}")
                if idx % 10 == 0:
                    tqdm.write(
                        "Epoch: [{}/{}], Batch: [{}/{}], batch loss: {:.6f}, epoch abs diff mean {:.6f}".format(
                            epoch,
                            num_epochs,
                            idx + 1,
                            len(dataloaders[phase]),
                            loss,
                            running_mad,
                        ),
                        end="\r",
                    )
                    telegram.edit_text_message(epoch_msg["message_id"],"Epoch: [{}/{}], Batch: [{}/{}], batch loss: {:.6f}".format(
                            epoch,
                            num_epochs,
                            idx + 1,
                            len(dataloaders[phase]),
                            loss,
                        ), )


            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            with torch.set_grad_enabled(phase == "train"):
                epoch_loss, epoch_mad = execute_epoch(
                    model,
                    criterion,
                    dataloaders[phase],
                    on_batch_done=on_batch_done,
                    optimizer=optimizer,
                )
            epoch_mads[phase].append(epoch_mad)
            epoch_losses[phase].append(epoch_loss)

            if phase == "train":
                scheduler.step()

            print(f"{phase} Loss: {epoch_loss:.4f}")
            telegram.send_telegram(f"{phase} Loss: {epoch_loss:.4f}" )

            if phase == "val":
                if epoch_loss < best_epoch_loss:
                    best_val_mad = epoch_mad
                    best_epoch_loss = epoch_loss
                    torch.save(model.encoder, "results/trash_auto_encoder.pt")
                    torch.save(model.decoder, "results/trash_auto_decoder.pt")
            print()
            if phase == "val" and should_stop(epoch_losses["val"]):
                print("Stopping early...")
                telegram.send_telegram("stopping early ..")
                break
        
    print(f"Best val Acc: {best_epoch_loss:4f}")
    telegram.send_telegram(f"Best val Acc: {best_epoch_loss:4f}")


    if dataloaders["test"]:
        print("Executing validation on test set...")
        test_loss, test_mad = execute_epoch(
            model,
            criterion,
            dataloaders["test"],
            on_batch_done=on_batch_done,
            optimizer=optimizer,
        )
        print()


    print({
            "model": model,
            "best_epoch_loss": best_epoch_loss,
            "test_loss": test_loss,
        })



    fig = plt.figure()
    plt.plot(range(len(epoch_losses["train"])),epoch_losses["train"])
    plt.savefig("train_vae.png")
    telegram.send_photo("train_vae.png", "Trainloss")
    plt.plot(range(len(epoch_losses["val"])),epoch_losses["val"])
    plt.savefig("val_vae.png")
    telegram.send_photo("val_vae.png", "val loss")
except Exception as e:
    telegram.send_telegram(e)




