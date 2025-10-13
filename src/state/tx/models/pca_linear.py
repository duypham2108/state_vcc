# pyright: reportMissingImports=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportIncompatibleMethodOverride=false
from typing import Dict, Optional
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
        hidden_dim: int = 0,
        output_dim: int = 0,
        pert_dim: int = 0,
        batch_dim: Optional[int] = None,
        output_space: str = "gene",
        gene_dim: Optional[int] = None,
        **kwargs,
    ):
        # Ensure an int for base class signature
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

        # Linear mapping from pert embeddings (and optional external features) to PCA space
        self._pert_feat_dim = None  # will be set if external features are provided
        self.pert_to_pca: nn.Linear
        self._init_pert_mapping(input_dim=self.pert_dim, kwargs=kwargs)
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

        # Optional pretrained gene embeddings for decoding correction
        self._use_gene_embeddings = False
        self._gene_emb_matrix: Optional[torch.Tensor] = None  # [G, E]
        self._gene_emb_proj: Optional[nn.Linear] = None  # maps PCA delta -> gene-emb space or counts
        self._init_gene_embeddings(kwargs)

    def _build_networks(self):
        # No deep networks to build; PCA handled outside of torch.
        return

    def _init_pert_mapping(self, input_dim: int, kwargs: dict):
        """
        Initialize mapping from perturbation inputs to PCA space, optionally
        concatenating external perturbation features loaded from
        `perturbation_features_file`.
        The file is expected to be a torch.load()-able mapping from perturbation key
        to feature vector, aligned with batch["pert_name"] if provided at training.
        """
        pert_feats_file = kwargs.get("perturbation_features_file", None)
        if pert_feats_file is None:
            self._pert_feat_dim = 0
            self.pert_to_pca = nn.Linear(input_dim, self.pca_dim, bias=False)
            self._pert_external_features = None
            return

        try:
            loaded = torch.load(pert_feats_file)
        except Exception as e:
            logger.warning(f"Failed to load perturbation features from {pert_feats_file}: {e}")
            self._pert_feat_dim = 0
            self.pert_to_pca = nn.Linear(input_dim, self.pca_dim, bias=False)
            self._pert_external_features = None
            return

        # Expect a dict-like with string keys
        if isinstance(loaded, dict):
            # Convert all vectors to tensors
            ext = {str(k): (v if isinstance(v, torch.Tensor) else torch.tensor(v)) for k, v in loaded.items()}
            # Infer feature dim from first entry
            first_key = next(iter(ext.keys())) if len(ext) > 0 else None
            feat_dim = int(ext[first_key].numel()) if first_key is not None else 0
            self._pert_feat_dim = feat_dim
            self._pert_external_features = ext
            self.pert_to_pca = nn.Linear(input_dim + feat_dim, self.pca_dim, bias=False)
            logger.info(f"Loaded perturbation features with dim={feat_dim} from {pert_feats_file}")
        else:
            logger.warning(
                f"Unexpected format in {pert_feats_file}; expected dict mapping to vectors. Proceeding without extra features."
            )
            self._pert_feat_dim = 0
            self._pert_external_features = None
            self.pert_to_pca = nn.Linear(input_dim, self.pca_dim, bias=False)

    def _init_gene_embeddings(self, kwargs: dict):
        """
        Initialize optional gene embeddings for decoding refinement.
        If `gene_embeddings_path` is provided and loadable via torch.load, we create
        a projection from PCA delta into gene-embedding space and back to counts.
        """
        path = kwargs.get("gene_embeddings_path", None)
        if path is None:
            return
        try:
            loaded = torch.load(path)
        except Exception as e:
            logger.warning(f"Failed to load gene embeddings from {path}: {e}")
            return

        if isinstance(loaded, dict):
            # Expect a dict of {gene_name: embedding_tensor}
            # Build matrix in dataset gene order if `gene_names` are available
            gene_names = self.hparams.get("gene_names", None)
            if gene_names is None:
                # Fallback: try to stack in arbitrary order
                try:
                    mat = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in loaded.values()])
                except Exception as e:
                    logger.warning(f"Could not stack gene embeddings: {e}")
                    return
            else:
                rows = []
                missing = 0
                for g in gene_names:
                    vec = loaded.get(str(g), None)
                    if vec is None:
                        missing += 1
                        # If missing, use zeros
                        if len(rows) == 0:
                            continue
                        rows.append(torch.zeros_like(rows[0]))
                    else:
                        rows.append(vec if isinstance(vec, torch.Tensor) else torch.tensor(vec))
                if len(rows) == 0:
                    logger.warning("Gene embeddings missing or empty; skipping embedding refinement")
                    return
                if missing > 0:
                    logger.info(f"Gene embeddings missing for {missing} genes; filled with zeros")
                mat = torch.stack(rows, dim=0)

            # Cache on module (moved to device during forward)
            self._gene_emb_matrix = mat  # [G, E]
            # A small projection: PCA delta -> gene counts via embeddings
            self._gene_emb_proj = nn.Linear(self.pca_dim, mat.shape[1], bias=False)
            self._use_gene_embeddings = True
            logger.info(f"Loaded gene embeddings with shape {mat.shape} from {path}")
        else:
            logger.warning(
                f"Unexpected gene embeddings format at {path}; expected dict mapping gene->embedding tensor"
            )

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

    def _get_ctrl_counts_tensor(self, batch: dict) -> torch.Tensor:
        # Prefer explicit counts; else fall back to embeddings if provided
        if "ctrl_cell_counts" in batch and batch["ctrl_cell_counts"] is not None:
            return batch["ctrl_cell_counts"]
        if "ctrl_cell_emb" in batch and batch["ctrl_cell_emb"] is not None:
            return batch["ctrl_cell_emb"]
        raise KeyError("ctrl_cell_counts missing; PCALinearPerturbationModel requires counts in batch")

    def _get_pert_counts_tensor(self, batch: dict) -> torch.Tensor:
        # Prefer explicit counts; else fall back to embeddings if provided
        if "pert_cell_counts" in batch and batch["pert_cell_counts"] is not None:
            return batch["pert_cell_counts"]
        if "pert_cell_emb" in batch and batch["pert_cell_emb"] is not None:
            return batch["pert_cell_emb"]
        raise KeyError("pert_cell_counts missing; PCALinearPerturbationModel requires counts in batch")

    def forward(self, batch: dict, padded: bool = True) -> torch.Tensor:
        # Expect counts (or fallback embeddings) for decoding
        ctrl_source = self._get_ctrl_counts_tensor(batch)
        feature_dim = ctrl_source.shape[-1]

        if padded:
            ctrl_counts = ctrl_source.reshape(-1, self.cell_sentence_len, feature_dim)
            pert = batch["pert_emb"].reshape(-1, self.cell_sentence_len, self.pert_dim)
            batch_idx = batch.get("batch", None)
            if batch_idx is not None:
                if batch_idx.dim() == 2:
                    batch_idx = batch_idx.reshape(-1, self.cell_sentence_len)
                elif batch_idx.dim() == 3:
                    batch_idx = batch_idx.reshape(-1, self.cell_sentence_len, batch_idx.size(-1))
        else:
            ctrl_counts = ctrl_source.reshape(1, -1, feature_dim)
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
        pca = self._pca
        assert pca is not None
        z_ctrl = torch.from_numpy(pca.transform(flat_ctrl)).to(device)
        z_ctrl = z_ctrl.reshape(B, S, -1)

        # Compute additive effect in PCA space from perturbation embedding (+ optional external features)
        if self._pert_feat_dim and self._pert_external_features is not None and "pert_name" in batch:
            # Build feature tensor aligned to batch["pert_name"] (B,S)
            pert_names = batch["pert_name"]  # list/array-like of strings per element
            if isinstance(pert_names, (list, tuple)):
                # Expect flattened alignment; reshape to [B,S]
                if len(pert_names) == B * S:
                    pert_names = np.array(pert_names).reshape(B, S)
                else:
                    pert_names = np.array(pert_names)
            # Gather features
            feat_rows = []
            for i in range(B):
                row = []
                for j in range(S):
                    key = str(pert_names[i][j]) if pert_names.ndim == 2 else str(pert_names[j])
                    vec = self._pert_external_features.get(key, None)
                    if vec is None:
                        if len(feat_rows) == 0 and len(row) == 0:
                            # create a zero vector placeholder if needed
                            zero = torch.zeros(self._pert_feat_dim)
                        else:
                            zero = torch.zeros_like(feat_rows[0][0])
                        row.append(zero)
                    else:
                        row.append(vec if isinstance(vec, torch.Tensor) else torch.tensor(vec))
                feat_rows.append(torch.stack(row, dim=0))
            ext_feats = torch.stack(feat_rows, dim=0).to(pert.device).to(pert.dtype)
            pert_input = torch.cat([pert, ext_feats], dim=-1)
        else:
            pert_input = pert

        z_delta = self.pert_to_pca(pert_input)
        z_bias = self.pca_intercept.view(1, 1, -1)

        if self.batch_intercept is not None and batch_idx is not None:
            batch_indices = self._extract_batch_indices(batch_idx)
            z_batch = self.batch_intercept(batch_indices)
        else:
            z_batch = 0.0

        z_pred = z_ctrl + z_delta + z_bias + z_batch

        # Decode to gene space via inverse transform
        flat_z = z_pred.reshape(-1, z_pred.size(-1)).detach().cpu().numpy().astype(np.float32)
        pca = self._pca
        assert pca is not None
        y_hat = pca.inverse_transform(flat_z)
        y_hat = torch.from_numpy(y_hat).to(device).reshape(B, S, G)

        # Optional refinement using gene embeddings as a corrective term
        if self._use_gene_embeddings and self._gene_emb_matrix is not None and self._gene_emb_proj is not None:
            # Map PCA delta to gene-embedding space, then back to counts via a learned linear comb (through embeddings)
            ge = self._gene_emb_matrix.to(device)
            z_delta_flat = z_delta.reshape(-1, z_delta.size(-1))
            emb_coeffs = self._gene_emb_proj(z_delta_flat)  # [B*S, E]
            # Project to counts by multiplying by embedding matrix transpose: [B*S, E] @ [E, G]
            correction = emb_coeffs @ ge.T  # [B*S, G]
            correction = correction.reshape(B, S, G)
            y_hat = y_hat + correction

        if self.relu is not None:
            y_hat = self.relu(y_hat)

        return y_hat.reshape(-1, G)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded: bool = True) -> torch.Tensor:
        pred = self.forward(batch, padded=padded)
        target = self._get_pert_counts_tensor(batch)
        G = pred.shape[-1]
        B = pred.shape[0] // self.cell_sentence_len if padded else 1
        if padded:
            pred = pred.reshape(B, self.cell_sentence_len, G)
            target = target.reshape(B, self.cell_sentence_len, G)
        else:
            pred = pred.reshape(1, -1, G)
            target = target.reshape(1, -1, G)

        loss = self.loss_fn(pred, target).nanmean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        pred = self.forward(batch)
        G = pred.shape[-1]
        target = self._get_pert_counts_tensor(batch).reshape(-1, self.cell_sentence_len, G)
        pred = pred.reshape(-1, self.cell_sentence_len, G)
        loss = self.loss_fn(pred, target).mean()
        self.log("val_loss", loss)
        return None

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


