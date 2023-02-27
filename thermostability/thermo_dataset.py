from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import pickle
from IPython.display import clear_output, display
import sys
import csv
import numpy as np


def calc_norm(temps: "list[float]"):
    temps_np = np.array(temps)
    return temps_np.mean(), temps_np.var()


class ThermostabilityDataset(Dataset):
    def __init__(
        self,
        dataset_filename: str = "train.csv",
        limit: int = 100000,
        max_seq_len: int = 700,
    ) -> None:
        super().__init__()

        dsFilePath = os.path.join("data/", dataset_filename)
        if not os.path.exists(dsFilePath):
            raise Exception(f"{dsFilePath} does not exist.")

        self.limit = limit
        with open(dsFilePath, newline="\n") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=",", skipinitialspace=True)
            self.seq_thermos = [
                (seq, thermo)
                for (i, (seq, thermo)) in enumerate(spamreader)
                if i != 0 and len(seq) <= max_seq_len
            ]

    def norm_distr(self):
        temps = [thermo for (seq, thermo) in self.seq_thermos]
        return calc_norm(temps)

    def __len__(self):
        return min(len(self.seq_thermos), self.limit)

    def __getitem__(self, index):
        seq, thermo = self.seq_thermos[index]
        return seq, torch.tensor(float(thermo), dtype=torch.float32)
