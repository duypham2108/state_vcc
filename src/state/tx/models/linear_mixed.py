import logging
from typing import Dict, Optional

# pyright: reportMissingImports=false
import torch
import torch.nn as nn

from .base import PerturbationModel
from .utils import build_mlp, get_activation_class

logger = logging.getLogger(__name__)


# pyright: reportIncompatibleMethodOverride=false
class LinearMixedPerturbationModel(PerturbationModel):
    """
    A basic linear mixed-effects style model implemented with torch.

    - Fixed effects: linear maps applied to `ctrl_cell_emb` and `pert_emb`.
    - Random effect: per-batch intercept via an `nn.Embedding` (if `batch_dim` provided
      and `batch_encoder=True`).

    Notes
    - If `predict_residual=True`, the model predicts a residual on top of `ctrl_cell_emb`:
      y_hat = ctrl_cell_emb + W_pert * pert_emb + intercept + b_batch
    - Otherwise, we learn fixed effects for both inputs:
      y_hat = W_ctrl * ctrl_cell_emb + W_pert * pert_emb + intercept + b_batch
    - This avoids an additional dependency on statsmodels while giving a reasonable
      "basic LMM" baseline integrated with the existing Lightning training loop.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 0,
        output_dim: int = 0,
        pert_dim: int = 0,
        batch_dim: Optional[int] = None,
        predict_residual: bool = True,
        output_space: str = "gene",
        gene_dim: Optional[int] = None,
        **kwargs,
    ):
        effective_gene_dim = int(gene_dim) if gene_dim is not None else int(input_dim)
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            gene_dim=effective_gene_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            batch_dim=batch_dim,  # pyright: ignore[reportArgumentType]
            output_space=output_space,
            **kwargs,
        )

        self.predict_residual = predict_residual
        self.output_space = output_space
        # Sequence length hint for reshaping; fall back to dataset config key if provided
        self.cell_sentence_len = int(kwargs.get("cell_set_len", 256))

        # Follow StateTransition template hyperparams
        self.n_encoder_layers = kwargs.get("n_encoder_layers", 2)
        self.n_decoder_layers = kwargs.get("n_decoder_layers", 2)
        self.activation_class = get_activation_class(kwargs.get("activation", "gelu"))

        # Build template-like networks (no transformer)
        self._build_networks()

        # Global intercept
        self.intercept = nn.Parameter(torch.zeros(self.output_dim))

        # Optional batch encoders/effects
        self.batch_encoder_enabled = kwargs.get("batch_encoder", False) and batch_dim is not None
        if self.batch_encoder_enabled:
            # Random intercept in output space
            self.random_intercept = nn.Embedding(num_embeddings=batch_dim, embedding_dim=self.output_dim)
            # Fixed batch effect in hidden space
            enc_hidden = self.hidden_dim if self.hidden_dim and self.hidden_dim > 0 else max(64, self.input_dim)
            self.batch_encoder = nn.Embedding(num_embeddings=batch_dim, embedding_dim=enc_hidden)
        else:
            self.random_intercept = None
            self.batch_encoder = None

        # If the model outputs gene space, ensure non-negativity similar to other models
        is_gene_space = kwargs.get("embed_key") == "X_hvg" or kwargs.get("embed_key") is None
        if is_gene_space:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def _build_networks(self):
        enc_hidden = self.hidden_dim if self.hidden_dim and self.hidden_dim > 0 else max(64, self.input_dim)

        self.pert_encoder = build_mlp(
            in_dim=self.pert_dim,
            out_dim=enc_hidden,
            hidden_dim=enc_hidden,
            n_layers=self.n_encoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        self.basal_encoder = build_mlp(
            in_dim=self.input_dim,
            out_dim=enc_hidden,
            hidden_dim=enc_hidden,
            n_layers=self.n_encoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

        self.project_out = build_mlp(
            in_dim=enc_hidden,
            out_dim=self.output_dim,
            hidden_dim=enc_hidden,
            n_layers=self.n_decoder_layers,
            dropout=self.dropout,
            activation=self.activation_class,
        )

    def _extract_batch_indices(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert batch features to categorical indices if provided as one-hot.
        Expects shapes [B, S] (already reshaped) or [B, S, C].
        """
        if batch_tensor.dim() == 3:
            # One-hot -> indices
            return batch_tensor.argmax(-1)
        return batch_tensor.long()

    def forward(self, batch: dict, padded: bool = True) -> torch.Tensor:
        if padded:
            pert = batch["pert_emb"].reshape(-1, self.cell_sentence_len, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(-1, self.cell_sentence_len, self.input_dim)
            batch_idx = batch.get("batch", None)
            if batch_idx is not None:
                # Try to reshape to [B, S] or [B, S, C]
                if batch_idx.dim() == 2:
                    batch_idx = batch_idx.reshape(-1, self.cell_sentence_len)
                elif batch_idx.dim() == 3:
                    batch_idx = batch_idx.reshape(-1, self.cell_sentence_len, batch_idx.size(-1))
        else:
            pert = batch["pert_emb"].reshape(1, -1, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(1, -1, self.input_dim)
            batch_idx = batch.get("batch", None)
            if batch_idx is not None:
                if batch_idx.dim() == 1:
                    batch_idx = batch_idx.reshape(1, -1)
                elif batch_idx.dim() == 2:
                    batch_idx = batch_idx.reshape(1, -1, batch_idx.size(-1))

        # Encode inputs to hidden space
        pert_h = self.pert_encoder(pert)
        basal_h = self.basal_encoder(basal)

        # Optional fixed batch effect in hidden space
        if self.batch_encoder is not None and batch_idx is not None:
            if batch_idx.dim() == 3:
                batch_ids = batch_idx.argmax(-1)
            else:
                batch_ids = batch_idx.long()
            batch_h = self.batch_encoder(batch_ids)
            hidden_sum = pert_h + basal_h + batch_h
        else:
            hidden_sum = pert_h + basal_h

        # Project to output space
        if self.predict_residual:
            fixed_out = self.project_out(hidden_sum + basal_h)
        else:
            fixed_out = self.project_out(hidden_sum)

        # Random intercept
        if self.random_intercept is not None and batch_idx is not None:
            batch_indices = self._extract_batch_indices(batch_idx)
            rand_eff = self.random_intercept(batch_indices)
        else:
            rand_eff = 0.0

        # Combine
        # Broadcast intercept to [B, S, D]
        intercept = self.intercept.view(1, 1, -1)

        # Random intercept per sequence
        if self.random_intercept is not None and batch_idx is not None:
            if batch_idx.dim() == 3:
                seq_ids = batch_idx.argmax(-1)[:, :1]
            else:
                seq_ids = batch_idx[:, :1] if batch_idx.dim() == 2 else batch_idx.reshape(-1, 1)
            rand_seq = self.random_intercept(seq_ids.long())  # [B,1,D]
            rand_eff = rand_seq.expand(-1, hidden_sum.size(1), -1)
        else:
            rand_eff = 0.0

        y_hat = fixed_out + intercept + (rand_eff if not isinstance(rand_eff, float) else 0.0)

        # If output is gene space, apply ReLU to enforce non-negativity
        if self.relu is not None:
            y_hat = self.relu(y_hat)

        return y_hat.reshape(-1, self.output_dim)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded: bool = True) -> torch.Tensor:
        pred = self.forward(batch, padded=padded)
        target = batch["pert_cell_emb"]

        if padded:
            pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
            target = target.reshape(-1, self.cell_sentence_len, self.output_dim)
        else:
            pred = pred.reshape(1, -1, self.output_dim)
            target = target.reshape(1, -1, self.output_dim)

        loss = self.loss_fn(pred, target).nanmean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        pred = self.forward(batch)
        pred = pred.reshape(-1, self.cell_sentence_len, self.output_dim)
        target = batch["pert_cell_emb"].reshape(-1, self.cell_sentence_len, self.output_dim)
        loss = self.loss_fn(pred, target).mean()
        self.log("val_loss", loss)
        return {"loss": loss, "predictions": pred}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        pred = self.forward(batch, padded=False)
        target = batch["pert_cell_emb"]
        pred = pred.reshape(1, -1, self.output_dim)
        target = target.reshape(1, -1, self.output_dim)
        loss = self.loss_fn(pred, target).mean()
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, padded: bool = True, **kwargs):
        latent_output = self.forward(batch, padded=padded)
        output = {
            "preds": latent_output,
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
        }
        return output


