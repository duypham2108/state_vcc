import logging

import torch
import torch.nn as nn
from typing import Optional
from omegaconf import OmegaConf

from ...emb.finetune_decoder import Finetune

logger = logging.getLogger(__name__)


class FinetuneVCICountsDecoder(nn.Module):
    def __init__(
        self,
        genes,
        # model_loc="/large_storage/ctc/userspace/aadduri/vci/checkpoint/rda_tabular_counts_2048_new/step=950000.ckpt",
        # config="/large_storage/ctc/userspace/aadduri/vci/checkpoint/rda_tabular_counts_2048_new/tahoe_config.yaml",
        model_loc="/home/aadduri/vci_pretrain/vci_1.4.2.ckpt",
        config="/large_storage/ctc/userspace/aadduri/vci/checkpoint/large_1e-4_rda_tabular_counts_2048/crossds_config.yaml",
        read_depth=1200,
        latent_dim=1024,  # dimension of pretrained vci model
        hidden_dims=[512, 512, 512],  # hidden dimensions of the decoder
        dropout=0.1,
        basal_residual=False,
    ):
        super().__init__()
        self.genes = genes
        self.model_loc = model_loc
        self.config = config
        self.finetune = Finetune(OmegaConf.load(self.config))
        self.finetune.load_model(self.model_loc)
        self.read_depth = nn.Parameter(torch.tensor(read_depth, dtype=torch.float), requires_grad=False)
        self.basal_residual = basal_residual

        # layers = [
        #     nn.Linear(latent_dim, hidden_dims[0]),
        # ]

        # self.gene_lora = nn.Sequential(*layers)

        self.latent_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], len(self.genes)),
            nn.ReLU(),
        )

        self.gene_decoder_proj = nn.Sequential(
            nn.Linear(len(self.genes), 128),
            nn.Linear(128, len(self.genes)),
        )

        self.binary_decoder = self.finetune.model.binary_decoder
        for param in self.binary_decoder.parameters():
            param.requires_grad = False

    def gene_dim(self):
        return len(self.genes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, S, latent_dim].
        if len(x.shape) != 3:
            x = x.unsqueeze(0)
        batch_size, seq_len, latent_dim = x.shape
        x = x.view(batch_size * seq_len, latent_dim)

        # Get gene embeddings
        gene_embeds = self.finetune.get_gene_embedding(self.genes)

        # Handle RDA task counts
        use_rda = getattr(self.finetune.model.cfg.model, "rda", False)
        # Define your sub-batch size (tweak this based on your available memory)
        sub_batch_size = 16
        logprob_chunks = []  # to store outputs of each sub-batch

        for i in range(0, x.shape[0], sub_batch_size):
            # Get the sub-batch of latent vectors
            x_sub = x[i : i + sub_batch_size]

            # Create task_counts for the sub-batch if needed
            if use_rda:
                # task_counts_sub = torch.full(
                #     (x_sub.shape[0],), self.read_depth, device=x.device
                # )
                task_counts_sub = torch.ones((x_sub.shape[0],), device=x.device) * self.read_depth
            else:
                task_counts_sub = None

            # Compute merged embeddings for the sub-batch
            merged_embs_sub = self.finetune.model.resize_batch(x_sub, gene_embeds, task_counts_sub)

            # Run the binary decoder on the sub-batch
            logprobs_sub = self.binary_decoder(merged_embs_sub)

            # Squeeze the singleton dimension if needed
            if logprobs_sub.dim() == 3 and logprobs_sub.size(-1) == 1:
                logprobs_sub = logprobs_sub.squeeze(-1)

            # Collect the results
            logprob_chunks.append(logprobs_sub)

        # Concatenate the sub-batches back together
        logprobs = torch.cat(logprob_chunks, dim=0)

        # Reshape back to [B, S, gene_dim]
        decoded_gene = logprobs.view(batch_size, seq_len, len(self.genes))
        decoded_gene = decoded_gene + self.gene_decoder_proj(decoded_gene)
        # decoded_gene = torch.nn.functional.relu(decoded_gene)

        # # normalize the sum of decoded_gene to be read depth
        # decoded_gene = decoded_gene / decoded_gene.sum(dim=2, keepdim=True) * self.read_depth

        # decoded_gene = self.gene_lora(decoded_gene)
        # TODO: fix this to work with basal counts

        # add logic for basal_residual:
        decoded_x = self.latent_decoder(x)
        decoded_x = decoded_x.view(batch_size, seq_len, len(self.genes))

        # Pass through the additional decoder layers
        return decoded_gene + decoded_x


class GeneCrossAttentionDecoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        gene_embeddings: Optional[torch.Tensor] = None,
        gene_embeddings_path: Optional[str] = None,
        num_heads: int = 4,
        dropout: float = 0.1,
        freeze_embeddings: bool = True,
    ):
        super().__init__()

        if gene_embeddings is None and gene_embeddings_path is None:
            raise ValueError("Provide either gene_embeddings tensor or gene_embeddings_path")

        if gene_embeddings is None:
            loaded = torch.load(gene_embeddings_path)
            if isinstance(loaded, dict) and "embeddings" in loaded:
                gene_embeddings = loaded["embeddings"]
            else:
                gene_embeddings = loaded
        if not isinstance(gene_embeddings, torch.Tensor):
            raise TypeError("gene_embeddings must be a torch.Tensor")

        self.register_buffer("_gene_embeddings_buffer", gene_embeddings.float())
        self._freeze_embeddings = freeze_embeddings

        in_dim = self._gene_embeddings_buffer.shape[1]
        self.project_gene = nn.Linear(in_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_dim, 1)

    def gene_dim(self) -> int:
        return int(self._gene_embeddings_buffer.shape[0])

    def forward(self, cell_latents: torch.Tensor) -> torch.Tensor:
        # cell_latents: [B, S, H]
        B = cell_latents.size(0)
        gene_emb = self._gene_embeddings_buffer
        if not self._freeze_embeddings and gene_emb.requires_grad is False:
            gene_emb = gene_emb.clone().detach().requires_grad_(True)

        gene_q = self.project_gene(gene_emb)  # [G, H]
        gene_q = gene_q.unsqueeze(0).expand(B, -1, -1)  # [B, G, H]

        attn_out, _ = self.attn(query=gene_q, key=cell_latents, value=cell_latents)
        logits = self.out(attn_out).squeeze(-1)  # [B, G]

        # Expand back to per-cell if needed: replicate per sequence cell
        # Here we treat gene prediction per cell; broadcast to sequence length
        S = cell_latents.size(1)
        logits = logits.unsqueeze(1).expand(B, S, -1)  # [B, S, G]
        return logits
