"""Train LightGCN and save checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import build_bipartite_adj, load_lastfm
from src.lightgcn import TrainConfig, train_lightgcn
from src.metrics import ndcg_at_k, recall_at_k
from src.paths import DEFAULT_CHECKPOINT


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--quick", action="store_true", help="Fast demo: 8 epochs")
    args = p.parse_args()

    data = load_lastfm(cache=False)
    adj = build_bipartite_adj(data.train)
    cfg = TrainConfig(
        embed_dim=args.embed_dim,
        n_layers=args.layers,
        epochs=8 if args.quick else args.epochs,
        device=args.device,
    )
    print(f"Training LightGCN: users={data.n_users}, items={data.n_items}, epochs={cfg.epochs}")
    model, meta = train_lightgcn(data.train, adj, data.n_users, data.n_items, cfg)

    rec = recall_at_k(model, adj, data.train, data.test, data.n_users, k=10, device=cfg.device)
    ndcg = ndcg_at_k(model, adj, data.train, data.test, data.n_users, k=10, device=cfg.device)
    print(f"Recall@10 (sampled users): {rec:.4f}")
    print(f"NDCG@10 (sampled users): {ndcg:.4f}")

    payload = {
        "state_dict": model.state_dict(),
        "meta": meta,
        "metrics": {"recall@10": rec, "ndcg@10": ndcg},
    }
    DEFAULT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, DEFAULT_CHECKPOINT)
    print(f"Saved checkpoint: {DEFAULT_CHECKPOINT}")


if __name__ == "__main__":
    main()
