import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from .base import PerturbationModel

logger = logging.getLogger(__name__)


class PCALinearPerturbationModel(PerturbationModel):
    """
    PCA + linear additive model with inverse transform decoding to gene space.

    Workflow (per time step/cell):
    1) Fit PCA on control counts (lazy, on first batch) to K components.
    2) Transform control counts to PCA space: z_ctrl = PCA.transform(ctrl_counts)
    3) Predict additive effect in PCA space from `pert_emb`: z_delta = W_pert * pert_emb + b0 [+ b_batch]
    4) z_pred = z_ctrl + z_delta
    5) Decode to gene space: y_hat = PCA.inverse_transform(z_pred)

    Loss is computed against `pert_cell_counts`.

    Notes
    - Set `pca_dim` via kwargs (default min(256, gene_dim)).
    - Optional per-batch random intercept in PCA space when `batch_encoder=True` and `batch_dim` provided.
    - Applies ReLU to decoded counts when `embed_key` is X_hvg or None.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        batch_dim: int = None,
        output_space: str = "gene",
        gene_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            gene_dim=gene_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            batch_dim=batch_dim,
            output_space=output_space,
            **kwargs,
        )

        # We operate in gene space; output_dim should equal gene_dim for this model
        if self.output_space != "gene":
            logger.warning("PCALinearPerturbationModel is intended for output_space='gene'")

        self.cell_sentence_len = int(kwargs.get("cell_set_len", 256))

        # PCA configuration
        self.pca_dim = int(kwargs.get("pca_dim", min(256, int(self.gene_dim))))
        self._pca_fitted = False
        self._pca: Optional[PCA] = None
        self._pca_components_: Optional[np.ndarray] = None
        self._pca_mean_: Optional[np.ndarray] = None

        # Linear mapping from pert embeddings to PCA space (additive effect)
        self.pert_to_pca = nn.Linear(self.pert_dim, self.pca_dim, bias=False)
        self.pca_intercept = nn.Parameter(torch.zeros(self.pca_dim))

        # Optional random intercept per batch in PCA space
        self.batch_encoder_enabled = kwargs.get("batch_encoder", False) and batch_dim is not None
        if self.batch_encoder_enabled:
            self.batch_intercept = nn.Embedding(num_embeddings=batch_dim, embedding_dim=self.pca_dim)
        else:
            self.batch_intercept = None

        # Non-negativity for gene counts
        is_gene_space = kwargs.get("embed_key") == "X_hvg" or kwargs.get("embed_key") is None
        self.relu = nn.ReLU() if is_gene_space else None

    def _build_networks(self):
        # No deep networks to build; PCA handled outside of torch.
        return

    def _ensure_pca(self, ctrl_counts_2d: np.ndarray):
        if not self._pca_fitted:
            # Fit PCA on control counts from the first batch
            n_features = ctrl_counts_2d.shape[1]
            n_components = min(self.pca_dim, n_features)
            self._pca = PCA(n_components=n_components, svd_solver="randomized")
            self._pca.fit(ctrl_counts_2d)
            self._pca_fitted = True
            self._pca_components_ = self._pca.components_.copy()
            self._pca_mean_ = self._pca.mean_.copy()
            logger.info(f"Fitted PCA with n_components={n_components} on first batch")

    def _extract_batch_indices(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        if batch_tensor.dim() == 3:
            return batch_tensor.argmax(-1)
        return batch_tensor.long()

    def forward(self, batch: dict, padded: bool = True) -> torch.Tensor:
        # Expect counts for decoding
        if "ctrl_cell_counts" not in batch:
            raise KeyError("ctrl_cell_counts missing; PCALinearPerturbationModel requires counts in batch")

        if padded:
            ctrl_counts = batch["ctrl_cell_counts"].reshape(-1, self.cell_sentence_len, self.gene_dim)
            pert = batch["pert_emb"].reshape(-1, self.cell_sentence_len, self.pert_dim)
            batch_idx = batch.get("batch", None)
            if batch_idx is not None:
                if batch_idx.dim() == 2:
                    batch_idx = batch_idx.reshape(-1, self.cell_sentence_len)
                elif batch_idx.dim() == 3:
                    batch_idx = batch_idx.reshape(-1, self.cell_sentence_len, batch_idx.size(-1))
        else:
            ctrl_counts = batch["ctrl_cell_counts"].reshape(1, -1, self.gene_dim)
            pert = batch["pert_emb"].reshape(1, -1, self.pert_dim)
            batch_idx = batch.get("batch", None)
            if batch_idx is not None:
                if batch_idx.dim() == 1:
                    batch_idx = batch_idx.reshape(1, -1)
                elif batch_idx.dim() == 2:
                    batch_idx = batch_idx.reshape(1, -1, batch_idx.size(-1))

        B, S, G = ctrl_counts.shape
        device = ctrl_counts.device

        # Lazily fit PCA on control counts from this batch
        flat_ctrl = ctrl_counts.reshape(-1, G).detach().cpu().numpy().astype(np.float32)
        self._ensure_pca(flat_ctrl)

        # Transform control counts to PCA space
        z_ctrl = torch.from_numpy(self._pca.transform(flat_ctrl)).to(device)
        z_ctrl = z_ctrl.reshape(B, S, -1)

        # Compute additive effect in PCA space from perturbation embedding
        z_delta = self.pert_to_pca(pert)
        z_bias = self.pca_intercept.view(1, 1, -1)

        if self.batch_intercept is not None and batch_idx is not None:
            batch_indices = self._extract_batch_indices(batch_idx)
            z_batch = self.batch_intercept(batch_indices)
        else:
            z_batch = 0.0

        z_pred = z_ctrl + z_delta + z_bias + z_batch

        # Decode to gene space via inverse transform
        flat_z = z_pred.reshape(-1, z_pred.size(-1)).detach().cpu().numpy().astype(np.float32)
        y_hat = self._pca.inverse_transform(flat_z)
        y_hat = torch.from_numpy(y_hat).to(device).reshape(B, S, G)

        if self.relu is not None:
            y_hat = self.relu(y_hat)

        return y_hat.reshape(-1, G)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded: bool = True) -> torch.Tensor:
        pred = self.forward(batch, padded=padded)
        if "pert_cell_counts" not in batch:
            raise KeyError("pert_cell_counts missing; PCALinearPerturbationModel trains on counts")

        target = batch["pert_cell_counts"]
        B = pred.shape[0] // self.cell_sentence_len if padded else 1
        if padded:
            pred = pred.reshape(B, self.cell_sentence_len, self.gene_dim)
            target = target.reshape(B, self.cell_sentence_len, self.gene_dim)
        else:
            pred = pred.reshape(1, -1, self.gene_dim)
            target = target.reshape(1, -1, self.gene_dim)

        loss = self.loss_fn(pred, target).nanmean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        pred = self.forward(batch)
        target = batch["pert_cell_counts"].reshape(-1, self.cell_sentence_len, self.gene_dim)
        pred = pred.reshape(-1, self.cell_sentence_len, self.gene_dim)
        loss = self.loss_fn(pred, target).mean()
        self.log("val_loss", loss)
        return {"loss": loss, "predictions": pred}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        pred = self.forward(batch, padded=False)
        target = batch["pert_cell_counts"]
        pred = pred.reshape(1, -1, self.gene_dim)
        target = target.reshape(1, -1, self.gene_dim)
        loss = self.loss_fn(pred, target).mean()
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, padded: bool = True, **kwargs):
        y_hat = self.forward(batch, padded=padded)
        return {
            "preds": y_hat,  # gene-space predictions
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
        }


