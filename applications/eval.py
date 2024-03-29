from tqdm.notebook import tqdm
from util.train_helper import execute_epoch
from datetime import datetime as dt
import torch
from torch.utils.data import DataLoader
from torch import nn as nn
import torch.backends.cudnn as cudnn
from thermostability.thermo_pregenerated_dataset import (
    ThermostabilityPregeneratedDataset,
)
import argparse
from util.experiments import store_experiment

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

cpu = torch.device("cpu")
torch.cuda.empty_cache()
torch.cuda.list_gpu_processes()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=1000000,
        help="Maximum number of samples to run evaluation on",
    )

    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default="test",
        help="Name is appended to the different log file names",
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="data/test.csv",
        help="Path to dataset csv with format: sequence, melting point",
    )

    parser.add_argument(
        "--representation_key",
        type=str,
        default="s_s_avg",
        choices=["s_s", "s_s_avg", "prott5_avg"],
        help='Representation key. Use "s_s" (per residue representation), "s_s_avg"(averaged s_s -> whole protein representation) for esm representations or "prott5_avg"(averaged prott5 embeddings)',
    )

    parser.add_argument(
        "--bin_width",
        "-w",
        type=int,
        default=20,
        help="Width of temperature ranges when evaluating performance with subsets",
    )

    currentTime = dt.now().strftime("%d-%m-%y_%H:%M:%S")
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=f"results/eval/{currentTime}",
        help="Path to directory where evaluation plots will be saved in",
    )
    args = parser.parse_args()
    argsDict = vars(args)

    limit = argsDict["limit"]
    model = torch.load(argsDict["model"]).to(device)
    dsPath = argsDict["dataset"]
    ds = ThermostabilityPregeneratedDataset(
        dsPath, limit=limit, representation_key=argsDict["representation_key"]
    )
    dataloader = DataLoader(ds, argsDict["batch_size"], shuffle=False)
    with torch.no_grad():
        epoch_loss, epoch_mad, epoch_actuals, epoch_predictions = execute_epoch(
            model,
            nn.MSELoss(),
            dataloader,
            prepare_inputs=lambda x: x.to(device),
            prepare_labels=lambda x: x.to(device),
            on_batch_done=lambda idx, _, batch_loss, running_mad: tqdm.write(
                "Batch: [{}/{}], batch loss: {:.6f}, epoch abs diff mean {:.6f}".format(
                    idx + 1,
                    len(dataloader),
                    batch_loss,
                    running_mad,
                ),
                end="\r",
            ),
        )

    print(
        f"""Evaluation done with \n- total average loss {epoch_loss:.2f}\n- total
          mean absolute difference {epoch_mad:.2f}"""
    )
    output_dir_path = argsDict["output_dir"]
    name = argsDict["name"]
    store_experiment(
        output_dir_path,
        name,
        epoch_loss,
        epoch_mad,
        epoch_predictions,
        epoch_actuals,
        argsDict,
    )
