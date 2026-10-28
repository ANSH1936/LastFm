"""Load and preprocess HetRec 2011 Last.fm data."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from src.paths import DATA_PROCESSED, DATA_RAW


@dataclass
class LastFmData:
    """Mapped user–artist interactions and side metadata."""

    n_users: int
    n_items: int
    train: csr_matrix
    test: csr_matrix
    user_id_map: dict[int, int]
    item_id_map: dict[int, int]
    inv_user_map: dict[int, int]
    inv_item_map: dict[int, int]
    artist_names: dict[int, str]
    tag_names: dict[int, str]
    friends: list[tuple[int, int]]
    user_tags: pd.DataFrame

    @property
    def train_pairs(self) -> np.ndarray:
        rows, cols = self.train.nonzero()
        return np.stack([rows, cols], axis=1)


def _is_header(*fields: str) -> bool:
    if not fields:
        return True
    return any(ch.isalpha() for ch in fields[0])


def _read_artists(raw_dir: Path) -> dict[int, str]:
    path = raw_dir / "artists.dat"
    names: dict[int, str] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and not _is_header(parts[0]):
                names[int(parts[0])] = parts[1]
    return names


def _read_tags(raw_dir: Path) -> dict[int, str]:
    path = raw_dir / "tags.dat"
    names: dict[int, str] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and not _is_header(parts[0]):
                names[int(parts[0])] = parts[1]
    return names


def _read_friends(raw_dir: Path, user_map: dict[int, int]) -> list[tuple[int, int]]:
    path = raw_dir / "user_friends.dat"
    edges: list[tuple[int, int]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and not _is_header(parts[0]):
                u, v = int(parts[0]), int(parts[1])
                if u in user_map and v in user_map:
                    edges.append((user_map[u], user_map[v]))
    return edges


def _read_user_tags(raw_dir: Path, user_map: dict[int, int], item_map: dict[int, int]) -> pd.DataFrame:
    path = raw_dir / "user_taggedartists.dat"
    rows = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and not _is_header(parts[0]):
                u, a, t = int(parts[0]), int(parts[1]), int(parts[2])
                if u in user_map and a in item_map:
                    rows.append({"user": user_map[u], "artist": item_map[a], "tag": t})
    return pd.DataFrame(rows)


def _read_interactions(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "user_artists.dat"
    rows = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and not _is_header(parts[0]):
                rows.append({"user": int(parts[0]), "artist": int(parts[1]), "weight": float(parts[2])})
    return pd.DataFrame(rows)


def load_lastfm(
    raw_dir: Path | None = None,
    test_ratio: float = 0.2,
    min_user_interactions: int = 5,
    seed: int = 42,
    cache: bool = True,
) -> LastFmData:
    """Load HetRec Last.fm; leave-one-out style holdout per user for test."""
    raw_dir = raw_dir or DATA_RAW
    cache_path = DATA_PROCESSED / "lastfm_processed.pkl"
    if cache and cache_path.is_file():
        with cache_path.open("rb") as f:
            return pickle.load(f)

    if not (raw_dir / "user_artists.dat").is_file():
        raise FileNotFoundError(
            f"Dataset not found under {raw_dir}. Run: python scripts/download_data.py"
        )

    inter = _read_interactions(raw_dir)
    counts = inter.groupby("user").size()
    keep_users = set(counts[counts >= min_user_interactions].index)
    inter = inter[inter["user"].isin(keep_users)]

    unique_users = sorted(inter["user"].unique())
    unique_items = sorted(inter["artist"].unique())
    user_id_map = {uid: i for i, uid in enumerate(unique_users)}
    item_id_map = {iid: i for i, iid in enumerate(unique_items)}
    inv_user_map = {v: k for k, v in user_id_map.items()}
    inv_item_map = {v: k for k, v in item_id_map.items()}

    rng = np.random.default_rng(seed)
    train_rows, train_cols, test_rows, test_cols = [], [], [], []

    for uid, grp in inter.groupby("user"):
        u = user_id_map[uid]
        items = grp["artist"].tolist()
        weights = grp["weight"].tolist()
        order = rng.permutation(len(items))
        n_test = max(1, int(len(items) * test_ratio))
        test_idx = set(order[:n_test])
        for j, (artist, w) in enumerate(zip(items, weights)):
            i = item_id_map[artist]
            if j in test_idx:
                test_rows.append(u)
                test_cols.append(i)
            else:
                train_rows.append(u)
                train_cols.append(i)

    n_users = len(unique_users)
    n_items = len(unique_items)
    train = csr_matrix(
        (np.ones(len(train_rows), dtype=np.float32), (train_rows, train_cols)),
        shape=(n_users, n_items),
    )
    test = csr_matrix(
        (np.ones(len(test_rows), dtype=np.float32), (test_rows, test_cols)),
        shape=(n_users, n_items),
    )

    artist_raw = _read_artists(raw_dir)
    artist_names = {item_id_map[k]: v for k, v in artist_raw.items() if k in item_id_map}
    tag_names = _read_tags(raw_dir)
    friends = _read_friends(raw_dir, user_id_map)
    user_tags = _read_user_tags(raw_dir, user_id_map, item_id_map)

    data = LastFmData(
        n_users=n_users,
        n_items=n_items,
        train=train,
        test=test,
        user_id_map=user_id_map,
        item_id_map=item_id_map,
        inv_user_map=inv_user_map,
        inv_item_map=inv_item_map,
        artist_names=artist_names,
        tag_names=tag_names,
        friends=friends,
        user_tags=user_tags,
    )
    if cache:
        with cache_path.open("wb") as f:
            pickle.dump(data, f)
    return data


def build_bipartite_adj(train: csr_matrix) -> csr_matrix:
    """Symmetric normalized adjacency for LightGCN (user + item nodes)."""
    n_users, n_items = train.shape
    ui = train.astype(np.float32)
    upper = sp.hstack([csr_matrix((n_users, n_users)), ui], format="csr")
    lower = sp.hstack([ui.T, csr_matrix((n_items, n_items))], format="csr")
    adj = sp.vstack([upper, lower], format="csr")
    rowsum = np.array(adj.sum(axis=1)).flatten()
    rowsum[rowsum == 0] = 1.0
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_mat = sp.diags(d_inv_sqrt)
    return d_mat @ adj @ d_mat
