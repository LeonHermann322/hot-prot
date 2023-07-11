import torch
import time
from torch import nn as nn
from tqdm.notebook import tqdm
import sys
import wandb
from typing import Callable
from scipy.stats import spearmanr
import pandas as pd
from typing_extensions import TypedDict, Literal
from util.telegram import TelegramBot


def calculate_metrics(predictions, labels, key: str):
    predictions = predictions.detach().cpu()
    labels = labels.detach().cpu()
    diffs = pd.Series([abs(pred - labels[i]) for (i, pred) in enumerate(predictions)])
    return {
        f"best_epoch_spearman_r_s_{key}": spearmanr(predictions, labels).correlation,
        f"best_epoch_max_abs_diff_{key}": diffs.max(),
        f"best_epoch_median_abs_diff_{key}": diffs.median(),
        f"best_epoch_mean_abs_diff_{key}": diffs.mean(),
    }


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
    epoch_predictions = torch.tensor([])
    epoch_actuals = torch.tensor([])
    running_loss = 0.0
    epoch_mad = 0.0
    # Iterate over data.
    for idx, (inputs, labels) in enumerate(dataloader):
        inputs = prepare_inputs(inputs)
        labels = prepare_labels(labels)
        # zero the parameter gradients
        if optimizer:
            optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, torch.unsqueeze(labels, 1))
        epoch_predictions = torch.cat((epoch_predictions, outputs.cpu()))
        epoch_actuals = torch.cat((epoch_actuals, labels.cpu()))
        # statistics
        batch_loss = loss.item()

        running_loss += batch_loss
        mean_abs_diff = (
            torch.abs(outputs.squeeze().sub(labels.squeeze())).squeeze().mean().item()
        )
        epoch_mad += mean_abs_diff
        running_mad = epoch_mad / (idx + 1)
        on_batch_done(idx, outputs, loss, running_mad)

    epoch_mad = epoch_mad / len(dataloader)
    epoch_loss = running_loss / len(dataloader)
    return (
        epoch_loss,
        epoch_mad,
        epoch_actuals.squeeze().tolist(),
        epoch_predictions.squeeze().tolist(),
    )


class TrainResponse(TypedDict):
    model: nn.Module
    best_epoch_loss: float
    best_val_mad: float
    epoch_mads: 'dict[Literal["train", "val"], "list[float]"]'
    best_epoch_actuals: "list[float]"
    best_epoch_predictions: "list[float]"
    test_loss: float
    test_mad: float
    test_actuals: "list[float]"
    test_predictions: "list[float]"


def train_model(
    model,
    criterions,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    dataloaders,
    use_wandb,
    num_epochs=25,
    best_model_path: str = None,
    max_gradient_clip: float = 10,
    prepare_inputs: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    prepare_labels: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    should_stop: Callable[["list[float]"], bool] = lambda epoch_val_losses: False,
) -> TrainResponse:
    telegram = TelegramBot()
    telegram.send_telegram("Starting training..")
    epoch_msg = telegram.send_telegram("Epoch..")["result"]
    loss_msg = telegram.send_telegram("loss..")["result"]

    optimizer = scheduler.optimizer
    since = time.time()

    if best_model_path:
        torch.save(model, best_model_path)
    best_val_mad = sys.float_info.max
    epoch_losses = {"train": [], "val": []}
    best_epoch_loss = sys.float_info.max
    best_epoch_predictions = torch.tensor([])
    best_epoch_actuals = torch.tensor([])
    epoch_mads = {"train": [], "val": []}
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:

            def on_batch_done(idx, outputs, loss, running_mad):
                if phase == "train":
                    if not torch.isnan(loss):
                        loss.backward()
                        if max_gradient_clip:
                            threshold = max_gradient_clip
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
                    telegram.edit_text_message(
                        message_id=epoch_msg["message_id"],
                        editedText="Epoch: [{}/{}], Batch: [{}/{}], batch loss: {:.6f}, epoch abs diff mean {:.6f}".format(
                            epoch,
                            num_epochs,
                            idx + 1,
                            len(dataloaders[phase]),
                            loss,
                            running_mad,
                        ),
                    )

            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            with torch.set_grad_enabled(phase == "train"):
                epoch_loss, epoch_mad, epoch_actuals, epoch_predictions = execute_epoch(
                    model,
                    criterions[phase],
                    dataloaders[phase],
                    prepare_inputs,
                    prepare_labels,
                    on_batch_done=on_batch_done,
                    optimizer=optimizer,
                )
            epoch_mads[phase].append(epoch_mad)
            epoch_losses[phase].append(epoch_loss)

            if use_wandb:
                wandb.log(
                    {
                        f"epoch_mad_{phase}": epoch_mad,
                    }
                )
            if phase == "train":
                scheduler.step()

            print(f"{phase} Loss: {epoch_loss:.4f}")
            telegram.edit_text_message(
                message_id=loss_msg["message_id"],
                editedText=f"{phase} Loss: {epoch_loss:.4f}",
            )

            if phase == "val":
                if use_wandb:
                    wandb.log({"loss": epoch_loss})
                if epoch_loss < best_epoch_loss:
                    best_val_mad = epoch_mad
                    best_epoch_loss = epoch_loss
                    if best_model_path:
                        torch.save(model, best_model_path)

                    best_epoch_actuals = epoch_actuals
                    best_epoch_predictions = epoch_predictions
        print()
        if phase == "val" and should_stop(epoch_losses["val"]):
            print("Stopping early...")
            telegram.send_telegram("stopping early ..")
            break

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_epoch_loss:4f}")
    telegram.send_telegram(
        f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    )
    telegram.send_telegram(f"Best val Acc: {best_epoch_loss:4f}")

    # load best model weights
    if best_model_path:
        model = torch.load(best_model_path)

    if dataloaders["test"]:
        print("Executing validation on test set...")
        test_loss, test_mad, test_actuals, test_predictions = execute_epoch(
            model,
            criterions["test"],
            dataloaders["test"],
            prepare_inputs,
            prepare_labels,
            on_batch_done=on_batch_done,
            optimizer=optimizer,
        )
        print()

    return {
        "model": model,
        "best_epoch_loss": best_epoch_loss,
        "best_val_mad": best_val_mad,
        "epoch_mads": epoch_mads,
        "best_epoch_actuals": best_epoch_actuals,
        "best_epoch_predictions": best_epoch_predictions,
        "test_loss": test_loss,
        "test_mad": test_mad,
        "test_actuals": test_actuals,
        "test_predictions": test_predictions,
    }
