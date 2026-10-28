"""Personalized PageRank explainer on a heterogeneous Last.fm graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from src.data import LastFmData


@dataclass
class PPRConfig:
    alpha: float = 0.15
    max_iter: int = 50
    tol: float = 1e-6
    friend_weight: float = 1.0
    listen_weight: float = 1.0
    colisten_weight: float = 0.35
    tag_weight: float = 0.25


@dataclass
class ExplainResult:
    target_artist: int
    target_name: str
    ppr_score: float
    rank_among_artists: int
    contributors: list[dict]
    paths: list[str]


class HetGraphPPR:
    """User–artist–friend graph with optional artist–artist co-listening edges."""

    def __init__(self, data: LastFmData, config: PPRConfig | None = None):
        self.data = data
        self.config = config or PPRConfig()
        self.n_users = data.n_users
        self.n_items = data.n_items
        self.n_nodes = self.n_users + self.n_items
        self.adj = self._build_adj()
        self.transition = self._normalize_outgoing(self.adj)

    def _build_adj(self) -> csr_matrix:
        cfg = self.config
        n = self.n_nodes
        rows, cols, vals = [], [], []

        ui, uj = self.data.train.nonzero()
        for u, i in zip(ui, uj):
            w = cfg.listen_weight
            artist_node = self.n_users + i
            rows.extend([u, artist_node])
            cols.extend([artist_node, u])
            vals.extend([w, w])

        for u, v in self.data.friends:
            if u != v and u < self.n_users and v < self.n_users:
                rows.extend([u, v])
                cols.extend([v, u])
                vals.extend([cfg.friend_weight, cfg.friend_weight])

        # Artist–artist: same user listened to both (top co-occurrence only)
        user_items: dict[int, list[int]] = {}
        for u, i in zip(ui, uj):
            user_items.setdefault(int(u), []).append(int(i))
        pair_count: dict[tuple[int, int], int] = {}
        for items in user_items.values():
            if len(items) < 2:
                continue
            uniq = sorted(set(items))
            for a in range(len(uniq)):
                for b in range(a + 1, len(uniq)):
                    key = (uniq[a], uniq[b])
                    pair_count[key] = pair_count.get(key, 0) + 1
        for (a, b), c in sorted(pair_count.items(), key=lambda x: -x[1])[:8000]:
            na, nb = self.n_users + a, self.n_users + b
            w = cfg.colisten_weight * np.log1p(c)
            rows.extend([na, nb])
            cols.extend([nb, na])
            vals.extend([w, w])

        if not self.data.user_tags.empty:
            tag_groups = self.data.user_tags.groupby(["artist", "tag"]).size()
            artist_tags: dict[int, set[int]] = {}
            for (artist, tag), _ in tag_groups.items():
                artist_tags.setdefault(int(artist), set()).add(int(tag))
            tag_pairs: dict[tuple[int, int], int] = {}
            for tags in artist_tags.values():
                tlist = sorted(tags)
                for a in range(len(tlist)):
                    for b in range(a + 1, len(tlist)):
                        key = (tlist[a], tlist[b])
                        tag_pairs[key] = tag_pairs.get(key, 0) + 1
            artists_by_tag: dict[int, list[int]] = {}
            for artist, tags in artist_tags.items():
                for t in tags:
                    artists_by_tag.setdefault(t, []).append(artist)
            for t, alist in artists_by_tag.items():
                if len(alist) < 2:
                    continue
                for a in range(min(len(alist), 40)):
                    for b in range(a + 1, min(len(alist), 40)):
                        i, j = alist[a], alist[b]
                        if i > j:
                            i, j = j, i
                        key = (i, j)
                        tag_pairs[key] = tag_pairs.get(key, 0) + 1
            for (a, b), c in sorted(tag_pairs.items(), key=lambda x: -x[1])[:6000]:
                na, nb = self.n_users + a, self.n_users + b
                w = cfg.tag_weight * np.log1p(c)
                rows.extend([na, nb])
                cols.extend([nb, na])
                vals.extend([w, w])

        return csr_matrix((vals, (rows, cols)), shape=(n, n))

    @staticmethod
    def _normalize_outgoing(adj: csr_matrix) -> csr_matrix:
        out_deg = np.array(adj.sum(axis=1)).flatten()
        out_deg[out_deg == 0] = 1.0
        inv = sp.diags(1.0 / out_deg)
        return inv @ adj

    def run(self, user_id: int) -> np.ndarray:
        cfg = self.config
        pi = np.zeros(self.n_nodes, dtype=np.float64)
        pi[user_id] = 1.0
        teleport = np.zeros(self.n_nodes, dtype=np.float64)
        teleport[user_id] = 1.0
        for _ in range(cfg.max_iter):
            new_pi = (1 - cfg.alpha) * (self.transition.T @ pi) + cfg.alpha * teleport
            if np.linalg.norm(new_pi - pi, 1) < cfg.tol:
                pi = new_pi
                break
            pi = new_pi
        return pi

    def _node_label(self, node: int) -> tuple[str, str]:
        if node < self.n_users:
            raw = self.data.inv_user_map.get(node, node)
            return "user", f"User {raw}"
        item = node - self.n_users
        name = self.data.artist_names.get(item, f"Artist {item}")
        return "artist", name

    def explain(self, user_id: int, artist_id: int, top_contrib: int = 8) -> ExplainResult:
        pi = self.run(user_id)
        artist_node = self.n_users + artist_id
        artist_slice = pi[self.n_users :]
        rank = int(np.argsort(-artist_slice).tolist().index(artist_id)) + 1

        # Neighbors of target artist with highest PPR mass (excluding target user direct if trivial)
        row = self.adj.getrow(artist_node)
        idx = row.indices
        contrib = []
        for n in idx:
            if n == user_id and n == artist_node:
                continue
            ntype, label = self._node_label(n)
            contrib.append(
                {
                    "node": int(n),
                    "type": ntype,
                    "label": label,
                    "ppr": float(pi[n]),
                    "edge_weight": float(row[0, n]),
                }
            )
        contrib.sort(key=lambda x: -x["ppr"])
        contrib = contrib[:top_contrib]

        paths = self._extract_paths(user_id, artist_id, pi)
        name = self.data.artist_names.get(artist_id, f"Artist {artist_id}")
        return ExplainResult(
            target_artist=artist_id,
            target_name=name,
            ppr_score=float(pi[artist_node]),
            rank_among_artists=rank,
            contributors=contrib,
            paths=paths,
        )

    def _extract_paths(self, user_id: int, artist_id: int, pi: np.ndarray) -> list[str]:
        """Greedy 2-hop path descriptions from user toward recommended artist."""
        user_label = f"User {self.data.inv_user_map.get(user_id, user_id)}"
        target = self.data.artist_names.get(artist_id, f"Artist {artist_id}")
        paths: list[str] = []

        artist_node = self.n_users + artist_id
        if self.data.train[user_id, artist_id] != 0:
            paths.append(f"{user_label} already listens to {target} (training history).")

        # Direct neighbor artists the user listens to that link to target via co-listen
        listened = self.data.train[user_id].nonzero()[1]
        target_neighbors = set(self.adj.getrow(artist_node).indices)
        bridges = []
        for i in listened:
            node_i = self.n_users + int(i)
            if node_i in target_neighbors:
                bridges.append(self.data.artist_names.get(int(i), f"Artist {i}"))
        if bridges:
            b = bridges[0]
            paths.append(
                f"{user_label} → your artist «{b}» → «{target}» via similar listening / co-occurrence."
            )

        # Friend path: user → friend → artist
        user_neighbors = self.adj.getrow(user_id).indices
        for n in sorted(user_neighbors, key=lambda x: -pi[x]):
            if n >= self.n_users:
                continue
            friend_label = f"User {self.data.inv_user_map.get(n, n)}"
            if artist_node in self.adj.getrow(n).indices:
                paths.append(f"{user_label} → friend {friend_label} → «{target}».")
                break

        if not paths:
            paths.append(
                f"PPR ranks «{target}» #{self._artist_rank(artist_id, pi)} among artists "
                f"reachable from {user_label} in the social + listening graph."
            )
        return paths[:5]

    def _artist_rank(self, artist_id: int, pi: np.ndarray) -> int:
        sl = pi[self.n_users :]
        return int(np.argsort(-sl).tolist().index(artist_id)) + 1
