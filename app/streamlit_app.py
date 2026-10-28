"""Streamlit dashboard: LightGCN recommendations + PPR explanations."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import LastFmData, build_bipartite_adj, load_lastfm
from src.lightgcn import LightGCN, TrainConfig, recommend, train_lightgcn
from src.metrics import ndcg_at_k, recall_at_k
from src.paths import DEFAULT_CHECKPOINT
from src.ppr import HetGraphPPR, PPRConfig

st.set_page_config(
    page_title="Last.fm LightGCN + PPR",
    page_icon="🎧",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading HetRec Last.fm data …")
def get_data() -> LastFmData:
    return load_lastfm()


@st.cache_resource(show_spinner="Building graph for PageRank …")
def get_ppr() -> HetGraphPPR:
    return HetGraphPPR(get_data(), PPRConfig())


def load_model(data: LastFmData, device: str):
    adj = build_bipartite_adj(data.train)
    if DEFAULT_CHECKPOINT.is_file():
        ckpt = torch.load(DEFAULT_CHECKPOINT, map_location=device, weights_only=False)
        meta = ckpt["meta"]
        cfg = meta["config"]
        model = LightGCN(data.n_users + data.n_items, cfg.embed_dim, cfg.n_layers)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        metrics = ckpt.get("metrics", {})
        return model, {"history": meta.get("history", {}), "metrics": metrics, "source": "checkpoint"}, adj

    with st.spinner("No checkpoint found — training a quick demo model (~2–5 min on CPU) …"):
        model, meta = train_lightgcn(
            data.train,
            adj,
            data.n_users,
            data.n_items,
            TrainConfig(epochs=8, device=device),
        )
        rec = recall_at_k(model, adj, data.train, data.test, data.n_users, k=10, device=device)
        ndcg = ndcg_at_k(model, adj, data.train, data.test, data.n_users, k=10, device=device)
        return model, {
            "history": meta.get("history", {}),
            "metrics": {"recall@10": rec, "ndcg@10": ndcg},
            "source": "demo-train",
        }, adj


def ppr_subgraph_figure(ppr_engine: HetGraphPPR, data: LastFmData, user_id: int, artist_id: int, pi: np.ndarray):
    G = nx.Graph()
    user_label = f"U{data.inv_user_map.get(user_id, user_id)}"
    artist_name = data.artist_names.get(artist_id, f"A{artist_id}")
    G.add_node(user_label, ntype="user", ppr=float(pi[user_id]))
    G.add_node(artist_name, ntype="artist", ppr=float(pi[ppr_engine.n_users + artist_id]))

    artist_node = ppr_engine.n_users + artist_id
    row_u = ppr_engine.adj.getrow(user_id)
    for n in row_u.indices[:12]:
        ntype, label = ppr_engine._node_label(n)
        if ntype == "user":
            label = f"U{data.inv_user_map.get(n, n)}"
        G.add_node(label, ntype=ntype, ppr=float(pi[n]))
        G.add_edge(user_label, label, weight=float(row_u[0, n]))

    row_a = ppr_engine.adj.getrow(artist_node)
    for n in row_a.indices[:12]:
        if n == user_id:
            continue
        ntype, label = ppr_engine._node_label(n)
        if ntype == "user":
            label = f"U{data.inv_user_map.get(n, n)}"
        G.add_node(label, ntype=ntype, ppr=float(pi[n]))
        G.add_edge(artist_name, label, weight=float(row_a[0, n]))

    pos = nx.spring_layout(G, seed=42, k=0.9)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, colors, sizes, labels, hovers = [], [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        ntype = G.nodes[node].get("ntype", "artist")
        colors.append("#4C78A8" if ntype == "user" else "#F58518")
        sizes.append(14 + 40 * G.nodes[node].get("ppr", 0) / (max(pi.max(), 1e-9)))
        labels.append(node)
        hovers.append(f"{node}<br>PPR={G.nodes[node].get('ppr', 0):.2e}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#aaa"), hoverinfo="none")
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=sizes, color=colors, line=dict(width=1, color="white")),
            hovertext=hovers,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title="PPR neighborhood (size ∝ PageRank mass)",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=420,
    )
    return fig


def main() -> None:
    st.title("Last.fm (HetRec 2011): LightGCN + Personalized PageRank")
    st.caption(
        "Graph neural collaborative filtering with post-hoc explanations on a heterogeneous "
        "user · friend · artist graph."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = get_data()
    ppr_engine = get_ppr()
    model, info, adj = load_model(data, device)

    with st.sidebar:
        st.header("Settings")
        raw_users = sorted(data.user_id_map.keys())
        selected_raw = st.selectbox("Last.fm user ID", raw_users, index=0)
        user_id = data.user_id_map[selected_raw]
        top_k = st.slider("Top-K recommendations", 5, 30, 10)
        st.divider()
        st.markdown("**Model**")
        st.write(f"Source: `{info.get('source', '?')}`")
        m = info.get("metrics", {})
        if m:
            st.metric("Recall@10", f"{m.get('recall@10', 0):.3f}")
            st.metric("NDCG@10", f"{m.get('ndcg@10', 0):.3f}")
        st.markdown(
            f"Users **{data.n_users}** · Artists **{data.n_items}** · "
            f"Train edges **{data.train.nnz}**"
        )

    tab_rec, tab_expl, tab_metrics = st.tabs(["Recommendations", "PPR explainer", "Training"])

    recs = recommend(model, adj, data.train, user_id, data.n_users, top_k=top_k, device=device)

    with tab_rec:
        st.subheader(f"Top-{top_k} artist recommendations (LightGCN)")
        rows = []
        for rank, (aid, score) in enumerate(recs, start=1):
            rows.append(
                {
                    "rank": rank,
                    "artist_id": data.inv_item_map.get(aid, aid),
                    "artist": data.artist_names.get(aid, f"Artist {aid}"),
                    "score": round(score, 4),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

    with tab_expl:
        st.subheader("Explain a recommendation with Personalized PageRank")
        artist_labels = [f"{data.artist_names.get(a, a)} ({data.inv_item_map.get(a, a)})" for a, _ in recs]
        choice = st.selectbox("Recommended artist", range(len(recs)), format_func=lambda i: artist_labels[i])
        artist_id = recs[choice][0]
        pi = ppr_engine.run(user_id)
        explanation = ppr_engine.explain(user_id, artist_id)
        c1, c2, c3 = st.columns(3)
        c1.metric("PPR score (artist node)", f"{explanation.ppr_score:.2e}")
        c2.metric("PPR rank among artists", explanation.rank_among_artists)
        c3.metric("LightGCN score", f"{recs[choice][1]:.4f}")

        st.markdown("**Why this artist?**")
        for p in explanation.paths:
            st.markdown(f"- {p}")

        st.markdown("**Top graph neighbors by PPR mass**")
        st.dataframe(explanation.contributors, use_container_width=True, hide_index=True)
        st.plotly_chart(ppr_subgraph_figure(ppr_engine, data, user_id, artist_id, pi), use_container_width=True)

    with tab_metrics:
        st.subheader("Training loss")
        hist = info.get("history", {}).get("loss", [])
        if hist:
            st.line_chart({"BPR loss": hist})
        else:
            st.info("Train with `python scripts/train.py` to persist loss curves in the checkpoint.")
        st.markdown(
            """
            **Pipeline**
            1. **LightGCN** learns user/artist embeddings on the bipartite listening graph (BPR).
            2. **Recommendations** = dot product between user embedding and unseen artist embeddings.
            3. **PPR explainer** runs on an extended graph (listen · friend · co-listen · tags) from the
               selected user to describe *graph pathways* that support the suggestion.
            """
        )


if __name__ == "__main__":
    main()
