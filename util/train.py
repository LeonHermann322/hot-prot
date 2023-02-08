import torch
import time
import copy
from torch import nn as nn
from tqdm.notebook import tqdm
import sys
import wandb
import pylab as pl
from typing import Callable


def train_model(
    model,
    criterion: nn.modules.loss._Loss,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    dataloaders,
    dataset_sizes,
    use_wandb,
    num_epochs=25,
    return_best_model=True,
    max_gradient_clip: float = 10,
    prepare_inputs: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    prepare_labels: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
):
    optimizer = scheduler.optimizer
    since = time.time()
    best_model_path = "results/best_model.pt"
    if return_best_model:
        torch.save(model, best_model_path)

    best_epoch_loss = sys.float_info.max
    losses = []
    batchEnumeration = []
    bestEpochPredictions = torch.tensor([])
    bestEpochLabels = torch.tensor([])
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)
        currentEpochPredictions = torch.tensor([])
        currentEpochLabels = torch.tensor([])
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for idx, (inputs, labels) in enumerate(dataloaders[phase]):
               
                inputs = prepare_inputs(inputs)
                labels = prepare_labels(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, torch.unsqueeze(labels, 1))

                    if phase == "val":
                        currentEpochPredictions = torch.cat(
                            (currentEpochPredictions, outputs.cpu())
                        )
                        currentEpochLabels = torch.cat(
                            (currentEpochLabels, labels.cpu())
                        )
                    # backward + optimize only if in training phase
                    if phase == "train":

                        if not torch.isnan(loss):
                            loss.backward()
                            if max_gradient_clip:
                                threshold = max_gradient_clip
                                for p in model.parameters():
                                    if p.grad != None:
                                        if p.grad.norm() > threshold:
                                            torch.nn.utils.clip_grad_norm_(p, threshold)
                            optimizer.step()
                        if torch.isnan(loss).any():
                            print(
                                f"Nan loss: {torch.isnan(loss)}| Loss: {loss}| inputs: {inputs}"
                            )
                # statistics
                batch_size = len(inputs)
                batch_loss = loss.item() * batch_size
                losses.append(batch_loss)
                batchEnumeration.append(
                    batchEnumeration[-1] + 1 if len(batchEnumeration) > 0 else 0
                )

                running_loss += batch_loss
                mean_abs_diff = (
                    torch.abs(outputs.squeeze().sub(labels.squeeze()))
                    .squeeze()
                    .mean()
                    .item()
                )
                if use_wandb:
                    wandb.log({"mean_abs_diff": mean_abs_diff})
                if idx % 1 == 0:
                    tqdm.write(
                        "Epoch: [{}/{}], Batch: [{}/{}], loss: {:.6f}, batch abs diff mean {:.6f}".format(
                            epoch,
                            num_epochs,
                            idx + 1,
                            len(dataloaders[phase]),
                            batch_loss / float(batch_size),
                            mean_abs_diff,
                        ),
                        end="\r",
                    )

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f}")

            # deep copy the model
            if phase == "val" and epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                if return_best_model:
                    torch.save(model, best_model_path)
                if use_wandb:
                    wandb.log({"mse_loss": epoch_loss})
                bestEpochLabels = currentEpochLabels
                bestEpochPredictions = currentEpochPredictions
        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_epoch_loss:4f}")
    # load best model weights
    if return_best_model:
        model = torch.load(best_model_path)

    pl.scatter(
        bestEpochPredictions.squeeze().tolist(), bestEpochLabels.squeeze().tolist()
    )
    train_size =dataset_sizes["train"]
    val_size = dataset_sizes["val"]
    plotPath = f"results/predictions_epochs{num_epochs}_gradClip{max_gradient_clip}_trainSize{train_size}_valSize{val_size}.png"
    pl.title(f"Loss: {best_epoch_loss}")
    pl.xlabel("Predictions")
    pl.ylabel("Labels")
    pl.savefig(plotPath)
    print(f"Saved predictions as scatter plot at {plotPath}")
    if use_wandb:
        artifact = wandb.Artifact("results", type="train")
        artifact.add_file(plotPath)
        wandb.log_artifact(artifact)
    return model, best_epoch_loss