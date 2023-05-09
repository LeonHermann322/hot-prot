from torch import nn
import torch
from thermostability.hotinfer_pregenerated import create_fc_layers


class RepresentationSummarizerAverage(nn.Module):
    def __init__(
        self,
        per_residue_summary=False,
    ):
        super().__init__()
        self.per_residue_summary = per_residue_summary
        self.per_sample_output_size = 700 if per_residue_summary else 1024

    def forward(self, s_s: torch.Tensor):
        # [-1, sequence_len, 1024]
        return s_s.mean(2 if self.per_residue_summary else 1)


class RepresentationSummarizerSingleInstance(nn.Module):
    def __init__(
        self,
        num_hidden_layers=1,
        per_residue_output_size=1,
        per_residue_summary=True,
        activation=nn.ReLU,
        p_dropout=0.0,
    ):
        super().__init__()

        # s_s shape torch.Size([1, sequence_len, 1024])
        # s_z shape torch.Size([1, sequence_len, sequence_len, 128])

        # if per_residue_summary is false we will summarize each row of the zero padded 700x1024 protein representation
        self.num_hidden_layers = num_hidden_layers
        self.per_residue_summary = per_residue_summary
        self.per_sample_output_size = per_residue_output_size * (
            700 if per_residue_summary else 1024
        )
        self.summarizer = create_fc_layers(
            num_hidden_layers,
            1024 if per_residue_summary else 700,
            per_residue_output_size,
            p_dropout=p_dropout,
            activation=activation,
        )

    def forward(self, s_s: torch.Tensor):
        # [-1, sequence_len, 1024]
        # [sequence_len, -1, 1024]
        to_summarize = (
            s_s.transpose(0, 1) if self.per_residue_summary else s_s.permute(2, 0, 1)
        )
        summaries = []
        for i, summarizable_batch in enumerate(to_summarize):
            summary = self.summarizer(summarizable_batch)
            summaries.append(summary)
        stacked = torch.stack(summaries, dim=1)
        return stacked


class RepresentationSummarizerMultiInstance(nn.Module):
    def __init__(
        self,
        num_hidden_layers=0,
        per_instance_output_size=1,
        per_residue_summary=True,
        activation=nn.ReLU,
        p_dropout=0.0,
        per_instance_input_size: int = None,
        num_instances: int = None,
    ):
        super().__init__()
        if not per_instance_input_size:
            per_instance_input_size = 1024 if per_residue_summary else 700

        if not num_instances:
            num_instances = 700 if per_residue_summary else 1024
        # s_s shape torch.Size([1, sequence_len, 1024])
        # s_z shape torch.Size([1, sequence_len, sequence_len, 128])
        self.per_residue_summary = per_residue_summary
        self.num_hidden_layers = num_hidden_layers
        self.per_sample_output_size = per_instance_output_size * num_instances
        self.summarizers = [
            create_fc_layers(
                num_hidden_layers,
                per_instance_input_size,
                per_instance_output_size,
                p_dropout=p_dropout,
                activation=activation,
            ).to("cuda:0")
            for _ in range(num_instances)
        ]

    def forward(self, s_s: torch.Tensor):
        # [-1, sequence_len, 1024]
        # [sequence_len, -1, 1024]

        to_summarize = (
            s_s.transpose(0, 1) if self.per_residue_summary else s_s.permute(2, 0, 1)
        )
        summaries = []
        for i, summarizable_batch in enumerate(to_summarize):
            summary = self.summarizers[i](summarizable_batch)
            summaries.append(summary)

        stacked = torch.stack(summaries, dim=1)
        return stacked
