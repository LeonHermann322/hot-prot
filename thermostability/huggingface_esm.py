import torch
from transformers import AutoTokenizer, EsmForSequenceClassification, EsmModel
from torch import nn
from thermostability.hotinfer import CachedModel, RepresentationKeysComb
from thermostability.hotinfer_pregenerated import create_fc_layers
from typing import Literal
import os

ESMSizes = Literal["8M", "35M", "150M", "650M", "3B", "15B", "v1_1B"]

model_names = {
    "8M": "facebook/esm2_t6_8M_UR50D",
    "35M": "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
    "3B": "facebook/esm2_t36_3B_UR50D",
    "15B": "facebook/esm2_t48_15B_UR50D",
    "v1_1B": "facebook/esm1b_t33_650M_UR50S",
}

required_config_attributes = [
    "model_hidden_layers",
    "model_dropoutrate",
    "hugg_esm_layer_norm",
    "hugg_esm_freeze",
    "hugg_esm_size",
]

embedding_dims = {
    "8M": 320,
    "35M": 480,
    "150M": 640,
    "650M": 1280,
    "3B": 2560,
    "15B": 5120,
    "v1_1B": 1280,
}


class ESMForThermostability(CachedModel):
    def __init__(
        self,
        regressor_layers: int = 3,
        regressor_dropout: float = 0.3,
        regressor_activation: nn.Module = nn.LeakyReLU,
        regressor_layer_norm: bool = True,
        freeze_esm: bool = False,
        model_size: ESMSizes = "8M",
        use_batch_norm: bool = False,
        pooling: Literal["bos_token", "mean"] = "bos_token",
        cache_dir: str = None,
    ):
        super().__init__(
            f"{pooling}_{model_size}",
            caching=freeze_esm,
            enable_grad=not freeze_esm,
            cache_dir=cache_dir,
        )
        assert (
            model_size in model_names
        ), f"Invalid ESM2 model size (--hugg_esm_size): {model_size}. Must be in {model_names.keys()} "

        print("Loading ESM model with batch norm:", use_batch_norm)
        self.model_size = model_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_names[model_size], cache_dir=cache_dir
        )
        self.regression = create_fc_layers(
            num=regressor_layers,
            input_size=embedding_dims[model_size],
            p_dropout=regressor_dropout,
            activation=regressor_activation,
            use_layer_norm_before_first=regressor_layer_norm,
            output_size=1,
            use_batch_norm=use_batch_norm,
        )
        self.freeze_esm = freeze_esm
        self.esm = None
        self.pooling = pooling
        self.cache_dir = cache_dir

    def _get_esm(self):
        if self.esm is None:
            self.esm = EsmModel.from_pretrained(
                model_names[self.model_size], cache_dir=self.cache_dir
            ).to("cuda:0")
        return self.esm

    def forward(self, sequences: "list[str]"):
        batch_pooled_embeddings = self.get_cached_or_compute(sequences)
        return self.regression(batch_pooled_embeddings)

    def compute_representations(self, seqs: "list[str]", _: RepresentationKeysComb):
        assert (
            torch.is_grad_enabled() == self._enable_grad
        ), f"Grad enabled state does not match for esm bos token embeddings computation (required: {self._enable_grad}, actual: {torch.is_grad_enabled()})"
        esm = self._get_esm()
        input_ids = self.tokenizer(
            seqs, padding=True, truncation=True, return_tensors="pt"
        ).input_ids.to("cuda:0")

        outputs = esm(input_ids)
        last_hidden_state = outputs.last_hidden_state

        s_embedding = (
            last_hidden_state[:, 0, :]
            if self.pooling == "bos_token"
            else torch.mean(  # mean over sequence for each token embedding except bos token
                last_hidden_state[
                    :,
                    1:,
                ],
                dim=1,
            )
        )
        return s_embedding

    @classmethod
    def from_config(cls, config):
        for attr in required_config_attributes:
            assert (
                attr in config
            ), f"Missing required attribute {attr} in config for ESMForThermostability"

        return ESMForThermostability(
            regressor_layers=config["model_hidden_layers"],
            regressor_dropout=config["model_dropoutrate"],
            regressor_layer_norm=config["hugg_esm_layer_norm"],
            freeze_esm=config["hugg_esm_freeze"],
            model_size=config["hugg_esm_size"],
            use_batch_norm=config["hugg_esm_batch_norm"],
            pooling=config["hugg_esm_pooling"],
            cache_dir=config["hugg_esm_cache_dir"],
        ).cuda()
