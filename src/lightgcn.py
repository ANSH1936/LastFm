"""LightGCN recommender (PyTorch)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from torch import Tensor


def _sparse_to_torch(sp_mat: csr_matrix, device: torch.device) -> Tensor:
    coo = sp_mat.tocoo()
    indices = torch.from_numpy(
        np.vstack([coo.row, coo.col]).astype(np.int64)
    )
    values = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device).coalesce()


class LightGCN(nn.Module):
    def __init__(self, n_nodes: int, embed_dim: int, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self.embedding = nn.Embedding(n_nodes, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def propagate(self, adj: Tensor, ego: Tensor) -> Tensor:
        embs = [ego]
        x = ego
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            embs.append(x)
        return torch.mean(torch.stack(embs, dim=0), dim=0)

    def forward(self, adj: Tensor) -> Tensor:
        ego = self.embedding.weight
        return self.propagate(adj, ego)

@dataclass
class TrainConfig:
    embed_dim: int = 64
    n_layers: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 2048
    epochs: int = 30
    device: str = "cpu"
    seed: int = 42


def _sample_bpr_batch(
    train: csr_matrix, n_items: int, batch_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    users, pos_items, _ = sp.find(train)
    idx = rng.integers(0, len(users), size=batch_size)
    batch_u = users[idx]
    batch_pos = pos_items[idx]
    batch_neg = rng.integers(0, n_items, size=batch_size)
    for k in range(batch_size):
        u = batch_u[k]
        while train[u, batch_neg[k]] != 0:
            batch_neg[k] = rng.integers(0, n_items)
    return batch_u, batch_pos, batch_neg


def train_lightgcn(
    train: csr_matrix,
    adj_norm: csr_matrix,
    n_users: int,
    n_items: int,
    config: TrainConfig | None = None,
) -> tuple[LightGCN, dict]:
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    n_nodes = n_users + n_items
    model = LightGCN(n_nodes, config.embed_dim, config.n_layers).to(device)
    adj = _sparse_to_torch(adj_norm, device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed)
    history: dict[str, list[float]] = {"loss": []}

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        n_steps = max(1, train.nnz // config.batch_size)
        for _ in range(n_steps):
            bu, bp, bn = _sample_bpr_batch(train, n_items, config.batch_size, rng)
            users_t = torch.from_numpy(bu).long().to(device)
            pos_t = torch.from_numpy(bp + n_users).long().to(device)
            neg_t = torch.from_numpy(bn + n_users).long().to(device)

            all_emb = model(adj)
            u_emb = all_emb[users_t]
            pos_emb = all_emb[pos_t]
            neg_emb = all_emb[neg_t]
            pos_scores = (u_emb * pos_emb).sum(dim=1)
            neg_scores = (u_emb * neg_emb).sum(dim=1)
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        history["loss"].append(epoch_loss / n_steps)

    model.eval()
    meta = {"config": config, "n_users": n_users, "n_items": n_items, "history": history}
    return model, meta


@torch.no_grad()
def recommend(
    model: LightGCN,
    adj_norm: csr_matrix,
    train: csr_matrix,
    user_id: int,
    n_users: int,
    top_k: int = 10,
    device: str = "cpu",
) -> list[tuple[int, float]]:
    dev = torch.device(device)
    adj = _sparse_to_torch(adj_norm, dev)
    model = model.to(dev)
    all_emb = model(adj)
    u_emb = all_emb[user_id]
    item_emb = all_emb[n_users:]
    scores = (item_emb @ u_emb).cpu().numpy()
    seen = train[user_id].nonzero()[1]
    scores[seen] = -np.inf
    top = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
    top = top[np.argsort(-scores[top])]
    return [(int(i), float(scores[i])) for i in top]
