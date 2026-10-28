"""Ranking metrics for implicit feedback."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from src.lightgcn import LightGCN, recommend


def recall_at_k(
    model: LightGCN,
    adj_norm,
    train: csr_matrix,
    test: csr_matrix,
    n_users: int,
    k: int = 10,
    device: str = "cpu",
    max_users: int | None = 200,
) -> float:
    hits = 0
    total = 0
    users = range(n_users if max_users is None else min(n_users, max_users))
    for u in users:
        test_items = test[u].nonzero()[1]
        if len(test_items) == 0:
            continue
        recs = [i for i, _ in recommend(model, adj_norm, train, u, n_users, top_k=k, device=device)]
        hits += len(set(recs) & set(test_items.tolist()))
        total += len(test_items)
    return hits / total if total else 0.0


def ndcg_at_k(
    model: LightGCN,
    adj_norm,
    train: csr_matrix,
    test: csr_matrix,
    n_users: int,
    k: int = 10,
    device: str = "cpu",
    max_users: int | None = 200,
) -> float:
    dcg_sum = 0.0
    users = range(n_users if max_users is None else min(n_users, max_users))
    n_eval = 0
    for u in users:
        test_items = set(test[u].nonzero()[1].tolist())
        if not test_items:
            continue
        recs = [i for i, _ in recommend(model, adj_norm, train, u, n_users, top_k=k, device=device)]
        dcg = 0.0
        for rank, item in enumerate(recs, start=1):
            if item in test_items:
                dcg += 1.0 / np.log2(rank + 1)
        idcg = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(test_items), k) + 1))
        if idcg > 0:
            dcg_sum += dcg / idcg
            n_eval += 1
    return dcg_sum / n_eval if n_eval else 0.0
