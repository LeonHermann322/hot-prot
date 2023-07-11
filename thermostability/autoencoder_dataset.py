from torch.utils.data import Dataset
import torch
import os
import csv
from typing import Union
from torch.nn.functional import pad
from thermostability.thermo_dataset import calc_norm
from esm_custom.esm.esmfold.v1.esmfold import RepresentationKey


def zero_padding(single_repr: torch.Tensor, len: int) -> torch.Tensor:
    dif = len - single_repr.size(0)
    return pad(single_repr, (0, 0, dif, 0), "constant", 0)


def zero_padding_700(single_repr: torch.Tensor) -> torch.Tensor:

    return zero_padding(single_repr, 700)


def zero_padding_collate(
    s_s_list: "list[tuple[torch.Tensor, torch.Tensor]]",
    fixed_size: Union[int, None] = None,
):
    max_size = fixed_size if fixed_size else max([s_s.size(0) for s_s, _ in s_s_list])

    padded_s_s = []
    temps = []
    for s_s, temp in s_s_list:
        padded = zero_padding(s_s, max_size)
        padded_s_s.append(padded)
        temps.append(temp)
    results = torch.stack(padded_s_s, 0).unsqueeze(1), torch.stack(temps).unsqueeze(1)
    return results


def zero_padding700_collate(s_s_list: "list[tuple[torch.Tensor, torch.Tensor]]"):
    return zero_padding_collate(s_s_list, 704)



""" Loads pregenerated esmfold outputs (sequence representations s_s) """


class AutoEncoderDataset(Dataset):
    def __init__(
        self,
        dataset_filepath: str = "data/train.csv",
        ds_type: str = "train",
        limit: int = 1000000,
        max_seq_len:int = 700,
    ) -> None:
        super().__init__()
        self.representations_dir = "/hpi/fs00/scratch/hoangan.nguyen/data/s_s"
        self.limit = limit
        self.type = ds_type
        if not os.path.exists(dataset_filepath):
            raise Exception(f"{dataset_filepath} does not exist.")
        with open(f"{self.representations_dir}/sequences.csv", newline="\n") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=",", skipinitialspace=True)
            self.sequenceToFilename = {
                sequence: filename
                for (i, (sequence, filename)) in enumerate(spamreader)
                if i != 0
            }

        with open(dataset_filepath, newline="\n") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=",", skipinitialspace=True)
            seq_thermos = [
                (seq, float(thermo))
                for (i, (seq, thermo)) in enumerate(spamreader)
                if i != 0
            ]

            self.filename_thermo_seq = [
                (self.sequenceToFilename[seq], thermo, seq)
                for (seq, thermo) in seq_thermos
                if seq in self.sequenceToFilename
            ]
            diff = len(seq_thermos) - len(self.filename_thermo_seq)
            print(
                f"""Omitted {diff} samples of {os.path.basename(dataset_filepath)} because
                 
                 ptheir sequences have not been pregenerated"""
            )

    def __len__(self):
        return min(len(self.filename_thermo_seq), self.limit)


    def __getitem__(self, index):
       

        filename, thermo, seq = self.filename_thermo_seq[index]
        with open(os.path.join(self.representations_dir, filename), "rb") as f:
            s_s = torch.load(f)
        
        mean = s_s.mean()
        std = s_s.std()
        s_s = (s_s-mean)/std
        return s_s, torch.tensor(thermo, dtype=torch.float32)
