import torch
from torch.utils.data import DataLoader
from torch import nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from thermostability.thermo_pregenerated_dataset import (
    ThermostabilityPregeneratedDataset,
)
from thermostability.hotinfer import HotInferModelParallel
from thermostability.hotinfer_pregenerated import HotInferPregeneratedFC
from thermostability.cnn_pregenerated import CNNPregeneratedFC, CNNPregenerated
from tqdm.notebook import tqdm
import sys
from thermostability.thermo_dataset import ThermostabilityDataset
from thermostability.thermo_pregenerated_dataset import (
    zero_padding_collate,
    zero_padding700_collate,
)
import wandb
import argparse
import pylab as pl
import os 

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

cpu = torch.device("cpu")
torch.cuda.empty_cache()
torch.cuda.list_gpu_processes()
from util.train_helper import train_model
from datetime import datetime as dt
from util.experiments import store_experiment
from thermostability.uni_prot_dataset import UniProtDataset

def run_train_experiment(results_path, config: dict = None, use_wandb=True):
    representation_key = config["representation_key"]
    model_parallel = config["model_parallel"] == "true"
    val_on_trainset = config["val_on_trainset"] == "true"
    limit = config["dataset_limit"]
    seq_length = config["seq_length"]
    train_ds = (
        UniProtDataset("data/train.csv", limit=limit, seq_length=seq_length)
        if representation_key == "uni_prot"
        else ThermostabilityPregeneratedDataset(
            "data/train.csv", limit=limit, usePerProteinRep=True
        )
        if representation_key == "s_s_0_avg"
        else ThermostabilityDataset("data/train.csv", limit=limit)
    )

    valFileName = "data/train.csv" if val_on_trainset else "data/val.csv"
    print("valFileName", valFileName)
    eval_ds = (
        UniProtDataset(valFileName, limit=limit, seq_length=seq_length)
        if representation_key == "uni_prot"
        else ThermostabilityPregeneratedDataset(
            valFileName, limit=limit, usePerProteinRep=True
        )
        if representation_key == "s_s_0_avg"
        else ThermostabilityDataset(valFileName, limit=limit)
    )
    dataloaders = {
        "train": DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
            # collate_fn=zero_padding700_collate,
        ),
        "val": DataLoader(
            eval_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
            # collate_fn=zero_padding700_collate,
        ),
    }
    input_sizes = {
        "esm_s_B_avg": 2560,
        "uni_prot": 1024,
        "s_s_0_A": 148 * 1024,
        "s_s_0_avg": 1024,
    }

    input_size = input_sizes[representation_key]
    thermo = (
        HotInferPregeneratedFC(
            input_len=input_size,
            num_hidden_layers=config["model_hidden_layers"],
            first_hidden_size=config["model_first_hidden_units"],
            p_dropout=config["model_dropoutrate"]
        )
        if config["model"] == "fc"
        else CNNPregeneratedFC(
            input_seq_len=input_size,
            num_hidden_layers=config["model_hidden_layers"],
            first_hidden_size=config["model_first_hidden_units"],
        )
    )

    model = (
        thermo
        if not model_parallel
        else HotInferModelParallel(representation_key, thermo_module=thermo)
    )
    if not model_parallel:
        model = model.to("cuda:0")

    if use_wandb:
        wandb.watch(thermo)
    criterion = nn.MSELoss()
    weight_decay = 1e-5 if config["weight_regularizer"] else 0
    optimizer_ft = (
        torch.optim.Adam(
            thermo.parameters(), lr=config["learning_rate"], weight_decay=weight_decay
        )
        if config["optimizer"] == "adam"
        else torch.optim.SGD(
            thermo.parameters(),
            lr=config["learning_rate"],
            momentum=0.9,
            weight_decay=weight_decay,
        )
    )
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.5)
    if not use_wandb:
        os.makedirs(results_path, exist_ok=True)
    model, best_epoch_loss, best_epoch_mad, epoch_mads,best_epoch_actuals, best_epoch_predictions = train_model(
        model,
        criterion,
        exp_lr_scheduler,
        dataloaders,
        use_wandb,
        num_epochs=config["epochs"],
        prepare_inputs=lambda x: x.to("cuda:0"),
        prepare_labels=lambda x: x.to("cuda:0")
        if not model_parallel
        else x.to("cuda:1"),
        best_model_path = os.path.join(results_path, "model.pt") if not use_wandb else None,
    )
    if use_wandb:
        data = [
            [x, y]
            for (x, y) in zip(
                best_epoch_predictions,
                best_epoch_actuals,
            )
        ]
        table = wandb.Table(data=data, columns=["predictions", "labels"])
        wandb.log({"predictions": wandb.plot.scatter(table, "predictions", "labels")})
    else:
        store_experiment(results_path,best_epoch_loss, best_epoch_mad, best_epoch_predictions, best_epoch_actuals, config, epoch_mads)

    return best_epoch_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.3)
    parser.add_argument("--model_hidden_layers", type=int, default=1)
    parser.add_argument("--model_first_hidden_units", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--val_on_trainset", type=str, choices=["true", "false"])
    parser.add_argument("--dataset_limit", type=int, default=1000000)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model", type=str, default="fc", choices=["fc", "cnn"])
    parser.add_argument("--model_parallel", type=str, choices=["true", "false"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--representation_key", type=str, default="uni_prot")
    parser.add_argument("--model_dropoutrate", type=float, default=0.3)
    parser.add_argument("--weight_regularizer", type=bool, default=True)
    parser.add_argument("--seq_length", type=int, default=700)

    args = parser.parse_args()

    argsDict = vars(args)
    use_wandb = argsDict["wandb"]
    del argsDict["wandb"]
    representation_key = argsDict["representation_key"]
    currentTime = dt.now().strftime("%d-%m-%y_%H:%M:%S")
    results_path = f"results/train/{representation_key}/{currentTime}"
    

    if use_wandb:
        with wandb.init(config=argsDict):
            run_train_experiment(
                config=wandb.config, use_wandb=True, results_path=results_path
            )

    else:
        run_train_experiment(
            config=argsDict, use_wandb=False, results_path=results_path
        )
