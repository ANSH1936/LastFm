"""Print sample model outputs to the terminal."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.data import build_bipartite_adj, load_lastfm
from src.lightgcn import LightGCN, recommend
from src.paths import DEFAULT_CHECKPOINT
from src.ppr import HetGraphPPR


def main() -> None:
    print("=" * 60)
    print("LAST.FM LightGCN + PPR — demo output")
    print("=" * 60)

    data = load_lastfm()
    print(f"\n[Data] users={data.n_users}, artists={data.n_items}, train_edges={data.train.nnz}")

    adj = build_bipartite_adj(data.train)
    ckpt = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = ckpt["meta"]["config"]
    print(f"\n[Checkpoint] {DEFAULT_CHECKPOINT.name}")
    print(f"  embed_dim={cfg.embed_dim}, layers={cfg.n_layers}, epochs_trained={cfg.epochs}")
    print(f"  metrics: {ckpt.get('metrics', {})}")

    model = LightGCN(data.n_users + data.n_items, cfg.embed_dim, cfg.n_layers)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    user_id = 0
    raw_uid = data.inv_user_map[user_id]
    print(f"\n[User] internal id={user_id}, Last.fm userID={raw_uid}")

    recs = recommend(model, adj, data.train, user_id, data.n_users, top_k=10, device="cpu")
    print("\n[LightGCN Top-10 recommendations]")
    print(f"{'Rank':<6}{'Artist ID':<12}{'Score':<10}Artist")
    print("-" * 60)
    for rank, (aid, score) in enumerate(recs, 1):
        name = data.artist_names.get(aid, f"Artist {aid}")
        raw_aid = data.inv_item_map.get(aid, aid)
        print(f"{rank:<6}{raw_aid:<12}{score:<10.4f}{name}")

    ppr = HetGraphPPR(data)
    artist_id = recs[0][0]
    ex = ppr.explain(user_id, artist_id)
    print(f"\n[PPR explain] top pick: {ex.target_name}")
    print(f"  PPR score: {ex.ppr_score:.6e}")
    print(f"  PPR rank among artists: {ex.rank_among_artists}")
    print(f"  LightGCN score: {recs[0][1]:.4f}")
    print("  Paths:")
    for p in ex.paths:
        print(f"    - {p}")
    print("  Top contributors (by PPR mass):")
    for c in ex.contributors[:5]:
        print(f"    - {c['label']} ({c['type']}): ppr={c['ppr']:.6e}")

    hist = ckpt["meta"].get("history", {}).get("loss", [])
    if hist:
        print(f"\n[Training loss] first={hist[0]:.4f}, last={hist[-1]:.4f}, steps={len(hist)}")
    print("\nDashboard: streamlit run app/streamlit_app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
