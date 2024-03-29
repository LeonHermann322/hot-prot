from torch.utils.data import Dataset
import torch
import os
import csv
from typing import Union, Callable
from torch.nn.functional import pad
from thermostability.thermo_dataset import calc_norm
from esm_custom.esm.esmfold.v1.esmfold import RepresentationKey


def zero_padding(single_repr: torch.Tensor, len: int, dim: int = 0) -> torch.Tensor:
    dif = len - single_repr.size(dim)
    return pad(single_repr, (0, 0, dif, 0), "constant", 0)


def zero_padding_700(single_repr: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return zero_padding(single_repr, 700, dim=dim)


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
    results = torch.stack(padded_s_s, 0), torch.stack(temps)
    return results

def zero_padding700_collate(s_s_list: "list[tuple[torch.Tensor, torch.Tensor]]"):
    return zero_padding_collate(s_s_list, 700)

def k_max_aggregate_collate(k: int, aggregateFunction: Callable[[torch.Tensor],torch.Tensor]):
    def execute(
        s_s_list: "list[tuple[torch.Tensor, torch.Tensor]]",
    ):
        padded_s_s = []
        temps = []
        for s_s, temp in s_s_list:
            # s_s: seq_len x 1024

            sums = aggregateFunction(s_s)
            index_sums = list(enumerate(sums))
            sorted_index_sums = sorted(index_sums, key=lambda index_sum: index_sum[1].item(), reverse=True)
            k_max_sums_indices = [i for i,_ in (sorted_index_sums[:k] if len(sorted_index_sums) > k else sorted_index_sums)]
            s_s_filtered = s_s.index_select(0, torch.LongTensor(k_max_sums_indices))

            padded = zero_padding(s_s_filtered, k)
            padded_s_s.append(padded)
            temps.append(temp)
        results = torch.stack(padded_s_s, 0), torch.stack(temps)
        return results
    return execute   

def k_max_sum_collate(k: int):
    return k_max_aggregate_collate(k, lambda sample: sample.sum(dim=-1))

def k_max_var_collate(k: int):
    return k_max_aggregate_collate(k, lambda sample: sample.var(dim=-1))

""" Loads pregenerated esmfold outputs (sequence representations s_s) """


class ThermostabilityPregeneratedDataset(Dataset):
    def __init__(
        self,
        dsFilePath: str = "data/train.csv",
        limit: int = 1000000,
        max_seq_len: int = 700,
        representation_filepath: str = "data",
        representation_key: RepresentationKey = "s_s_avg",
    ) -> None:
        super().__init__()

        if not os.path.exists(dsFilePath):
            raise Exception(f"{dsFilePath} does not exist.")
        self.representations_dir = f"{representation_filepath}/{representation_key}"
        with open(f"{self.representations_dir}/sequences.csv", newline="\n") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=",", skipinitialspace=True)
            self.sequenceToFilename = {
                sequence: filename
                for (i, (sequence, filename)) in enumerate(spamreader)
                if i != 0
            }

        self.limit = limit
        with open(dsFilePath, newline="\n") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=",", skipinitialspace=True)
            seq_thermos = [
                (seq, float(thermo))
                for (i, (seq, thermo)) in enumerate(spamreader)
                if i != 0 and len(seq) <= max_seq_len
            ]

            self.filename_thermo_seq = [
                (self.sequenceToFilename[seq], thermo, seq)
                for (seq, thermo) in seq_thermos
                if seq in self.sequenceToFilename
            ]
            diff = len(seq_thermos) - len(self.filename_thermo_seq)
            print(
                f"""Omitted {diff} samples of {os.path.basename(dsFilePath)} because
                 their sequences have not been pregenerated"""
            )

    def norm_distr(self):
        temps = [thermo for (filename, thermo, seq) in self.filename_thermo_seq]
        return calc_norm(temps)

    def __len__(self):
        return min(len(self.filename_thermo_seq), self.limit)

    def __getitem__(self, index):
        filename, thermo, seq = self.filename_thermo_seq[index]
        with open(os.path.join(self.representations_dir, filename), "rb") as f:
            s_s = torch.load(f)

        return s_s, torch.tensor(thermo, dtype=torch.float32)
