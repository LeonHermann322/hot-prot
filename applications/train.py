import torch
from torch.utils.data import DataLoader
from torch import nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from thermostability.thermo_pregenerated_dataset import (
    ThermostabilityPregeneratedDataset,
    zero_padding700_collate,
)
from thermostability.hotinfer import HotInferModel

from thermostability.hotinfer_pregenerated import (
    HotInferPregeneratedFC,
    HotInferPregeneratedSummarizerFC,
)
from thermostability.cnn_pregenerated import CNNPregeneratedFC
from thermostability.thermo_dataset import ThermostabilityDataset
import wandb
import argparse
import os
from thermostability.repr_summarizer import (
    RepresentationSummarizerSingleInstance,
    RepresentationSummarizer700Instance,
)
from util.weighted_mse import Weighted_MSE_Loss
from util.train_helper import train_model, calculate_metrics
from datetime import datetime as dt
from util.experiments import store_experiment
from thermostability.uni_prot_dataset import UniProtDataset

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

cpu = torch.device("cpu")
torch.cuda.empty_cache()
torch.cuda.list_gpu_processes()


def run_train_experiment(
    results_path, config: dict = None, use_wandb=True, should_log=True
):
    representation_key = config["representation_key"]
    model_parallel = config["model_parallel"] == "true"
    val_on_trainset = config["val_on_trainset"] == "true"
    limit = config["dataset_limit"]
    seq_length = config["seq_length"]
    train_ds = (
        UniProtDataset("data/train.csv", limit=limit, seq_length=seq_length)
        if config["dataset"] == "uni_prot"
        else ThermostabilityPregeneratedDataset(
            "data/train.csv", limit=limit, representation_key=representation_key
        )
        if config["dataset"] == "pregenerated"
        else ThermostabilityDataset("data/train.csv", limit=limit)
    )

    valFileName = "data/train.csv" if val_on_trainset else "data/val.csv"

    eval_ds = (
        UniProtDataset(valFileName, limit=limit, seq_length=seq_length)
        if config["dataset"] == "uni_prot"
        else ThermostabilityPregeneratedDataset(
            valFileName, limit=limit, representation_key=representation_key
        )
        if config["dataset"] == "pregenerated"
        else ThermostabilityDataset(valFileName, limit=limit)
    )
    dataloaders = {
        "train": DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
            collate_fn=zero_padding700_collate
            if config["dataset"] == "pregenerated"
            else None,
        ),
        "val": DataLoader(
            eval_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
            collate_fn=zero_padding700_collate
            if config["dataset"] == "pregenerated"
            else None,
        ),
    }

    train_mean, train_var = train_ds.norm_distr()
    val_mean, val_var = eval_ds.norm_distr()

    criterions = {
        "train": Weighted_MSE_Loss(train_mean, train_var)
        if config["loss"] == "weighted_mse"
        else nn.MSELoss(),
        "val": Weighted_MSE_Loss(val_mean, val_var)
        if config["loss"] == "weighted_mse"
        else nn.MSELoss(),
    }

    summarizer = (
        RepresentationSummarizerSingleInstance(
            per_residue_output_size=config["summarizer_per_residue_out_size"],
            num_hidden_layers=config["summarizer_num_layers"],
        )
        if config["summarizer_type"] == "single_instance"
        else RepresentationSummarizer700Instance(
            per_residue_output_size=config["summarizer_per_residue_out_size"],
            num_hidden_layers=config["summarizer_num_layers"],
        )
        if config["summarizer_type"] == "700_instance"
        else None
    )

    input_sizes = {
        "esm_s_B_avg": 2560,
        "uni_prot": 1024,
        "s_s_0_A": 148 * 1024,
        "s_s_0_avg": 1024,
        "s_s_avg": 1024,
        "s_s": 1024 * 700,
    }

    input_size = input_sizes[representation_key]

    thermo = (
        HotInferPregeneratedFC(
            input_len=input_size,
            num_hidden_layers=config["model_hidden_layers"],
            first_hidden_size=config["model_first_hidden_units"],
            p_dropout=config["model_dropoutrate"],
        )
        if config["model"] == "fc"
        else CNNPregeneratedFC(
            input_seq_len=input_size,
            num_hidden_layers=config["model_hidden_layers"],
            first_hidden_size=config["model_first_hidden_units"],
        )
        if config["model"] == "cnn"
        else HotInferPregeneratedSummarizerFC(
            p_dropout=config["model_dropoutrate"],
            summarizer=summarizer,
            thermo_module=HotInferPregeneratedFC(
                input_len=700 * summarizer.per_residue_output_size,
                num_hidden_layers=config["model_hidden_layers"],
                first_hidden_size=config["model_first_hidden_units"],
                p_dropout=config["model_dropoutrate"],
            ),
        )
    )

    model = (
        thermo
        if not model_parallel
        else HotInferModel(representation_key, thermo_module=thermo)
    )
    if not model_parallel:
        model = model.to("cuda:0")

    if use_wandb:
        wandb.watch(thermo)

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
    if not use_wandb and should_log:
        os.makedirs(results_path, exist_ok=True)
    (
        model,
        best_epoch_loss,
        best_epoch_mad,
        epoch_mads,
        best_epoch_actuals,
        best_epoch_predictions,
    ) = train_model(
        model,
        criterions,
        exp_lr_scheduler,
        dataloaders,
        use_wandb,
        num_epochs=config["epochs"],
        prepare_inputs=lambda x: x.to("cuda:0"),
        prepare_labels=lambda x: x.to("cuda:0")
        if not model_parallel
        else x.to("cuda:1"),
        best_model_path=os.path.join(results_path, "model.pt")
        if not use_wandb and should_log
        else None,
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
        metrics = calculate_metrics(best_epoch_predictions, best_epoch_actuals)
        wandb.log(metrics)
    elif should_log:
        store_experiment(
            results_path,
            best_epoch_loss,
            best_epoch_mad,
            best_epoch_predictions,
            best_epoch_actuals,
            config,
            epoch_mads,
        )

    return best_epoch_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.3)
    parser.add_argument("--model_hidden_layers", type=int, default=1)
    parser.add_argument("--model_first_hidden_units", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--val_on_trainset", type=str, choices=["true", "false"])
    parser.add_argument("--dataset_limit", type=int, default=1000000)
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd"]
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--model", type=str, default="fc", choices=["fc", "cnn", "summarizer"]
    )
    parser.add_argument("--model_parallel", type=str, choices=["true", "false"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--representation_key", type=str, default="s_s_avg")
    parser.add_argument("--model_dropoutrate", type=float, default=0.3)
    parser.add_argument("--weight_regularizer", type=bool, default=True)
    parser.add_argument("--seq_length", type=int, default=700)
    parser.add_argument("--nolog", action="store_true")
    parser.add_argument("--summarizer_per_residue_out_size", type=int, default=1)
    parser.add_argument(
        "--summarizer_type",
        default=None,
        choices=[None, "single_instance", "700_instance"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["pregenerated", "end_to_end", "uni_prot"],
        default="pregenerated",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["weighted_mse", "mse"],
        default="mse",
    )
    parser.add_argument("--summarizer_num_layers", type=int, default=1)
    args = parser.parse_args()

    argsDict = vars(args)
    use_wandb = argsDict["wandb"]
    del argsDict["wandb"]
    should_log = argsDict["nolog"]
    del argsDict["nolog"]
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
            config=argsDict,
            use_wandb=False,
            results_path=results_path,
            should_log=should_log,
        )
