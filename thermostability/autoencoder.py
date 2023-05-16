from torch import nn
import torch
from thermostability.hotinfer_pregenerated import create_fc_layers
from thermostability.repr_summarizer import RepresentationSummarizerMultiInstance


class RepresentationAutoEncoder(nn.Module):
    def __init__(
        self,
        per_residue_input_size=1024,
        per_residue_output_size=10,
        num_residues=700,
        representation_size=1024,
        decoder_layer_sizes=[64, 128, 256]
    ):
        super().__init__()
        self.encoder = Encoder(
            per_residue_output_size=per_residue_output_size,
            per_residue_input_size=per_residue_input_size,
            num_residues=num_residues,
            representation_size=representation_size,
        )

        self.decoder = Decoder(
            representation_size=representation_size,
            per_residue_output_size=per_residue_input_size,
            num_residues=num_residues,
            layer_sizes=decoder_layer_sizes
        )

    def forward(self, s_s: torch.Tensor):
        representation = self.encoder(s_s)

        decoded = self.decoder(representation)

        return decoded


class Encoder(nn.Module):
    def __init__(
        self,
        num_hidden_layers=1,
        per_residue_input_size=1024,
        per_residue_output_size=10,
        num_residues=700,
        representation_size=1024,
    ):
        super().__init__()

        # s_s shape torch.Size([1, sequence_len, 1024])
        self.summarizer = RepresentationSummarizerMultiInstance(
            per_instance_output_size=per_residue_output_size,
            per_instance_input_size=per_residue_input_size,
            num_instances=num_residues,
            num_hidden_layers=num_hidden_layers,
        )

        self.merger = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(per_residue_output_size * num_residues, representation_size),
            nn.ReLU(),
        )

    def forward(self, s_s: torch.Tensor):
        # [-1, sequence_len, 1024]
        # [sequence_len, -1, 1024]
        per_residue_summaries = self.summarizer(s_s)
        representation = self.merger(per_residue_summaries)
        return representation


class Decoder(nn.Module):
    def __init__(
        self,
        representation_size=10,
        num_residues=700,
        per_residue_output_size=1024,
        activation=nn.ReLU,
        p_dropout=0.0,
        layer_sizes=[64, 128, 256],
    ):
        super().__init__()

        output_size = num_residues * per_residue_output_size

        self.num_residues = num_residues
        self.per_residue_output_size = per_residue_output_size

        all_layer_sizes = [representation_size, *layer_sizes, output_size]
        modulePairs = [
            [nn.Linear(all_layer_sizes[i], all_layer_sizes[i + 1]), activation()]
            for i in range(len(all_layer_sizes) - 1)
        ]
        modules = [modulePairs[i // 2][i % 2] for i in range(len(modulePairs) * 2)]

        self.layers = nn.Sequential(*modules)

    def forward(self, representation: torch.Tensor):
        reconstructed = self.layers(representation)
        return reconstructed.view(-1, self.num_residues, self.per_residue_output_size)
