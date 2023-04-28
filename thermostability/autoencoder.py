from torch import nn
import torch
from thermostability.hotinfer_pregenerated import create_fc_layers


class RepresentationAutoEncoder(nn.Module):
    def __init__(
        self,
        per_residue_input_size=1024,
        per_residue_output_size=10,
    ):
        super().__init__()

        self.encoder = Encoder(
            per_residue_output_size=per_residue_output_size,
            per_residue_input_size=per_residue_input_size,
        )
        self.decoder = Decoder(
            representation_size=per_residue_output_size,
            per_residue_output_size=per_residue_input_size,
        )

    def forward(self, s_s: torch.Tensor):
        representation, size_matrix = self.encoder(s_s)
        decoded = self.decoder(representation, size_matrix)
        return decoded


class Encoder(nn.Module):
    def __init__(
        self,
        num_hidden_layers=1,
        per_residue_input_size=1024,
        per_residue_output_size=10,
        activation=nn.ReLU,
        p_dropout=0.0,
    ):
        super().__init__()

        # s_s shape torch.Size([1, sequence_len, 1024])

        self.num_hidden_layers = num_hidden_layers

        self.summarizer = create_fc_layers(
            num_hidden_layers,
            per_residue_input_size,  # per residue vector representation len
            per_residue_output_size,
            p_dropout=p_dropout,
            activation=activation,
        )

        self.representation_square_axis_len = per_residue_output_size

    def forward(self, s_s: torch.Tensor):
        # [-1, sequence_len, 1024]
        # [sequence_len, -1, 1024]
        to_summarize = s_s.transpose(0, 1)
        summaries = []
        for i, summarizable_batch in enumerate(to_summarize):
            summary = self.summarizer(summarizable_batch)
            summaries.append(summary)
        stacked = torch.stack(summaries, dim=1)

        return torch.bmm(stacked.transpose(1, 2), stacked), stacked


class Decoder(nn.Module):
    def __init__(
        self,
        representation_size=10,
        per_residue_output_size=1024,
        activation=nn.ReLU,
        p_dropout=0.0,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(representation_size, per_residue_output_size / 2),
            activation(),
            nn.Linear(per_residue_output_size / 2, per_residue_output_size),
        )

    def forward(self, representation: torch.Tensor, size_matrix: torch.Tensor):
        sized_representation = torch.bmm(torch.ones_like(size_matrix), representation)

        reconstructions = []
        to_decode = sized_representation.transpose(0, 1)
        for i, summarizable_batch in enumerate(to_decode):
            decoded = self.layers(summarizable_batch)
            reconstructions.append(decoded)
        stacked = torch.stack(reconstructions, dim=1)
        return stacked
