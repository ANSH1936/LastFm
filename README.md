# Last.fm HetRec 2011 — LightGCN + PPR Explainer

Collaborative filtering with **LightGCN** on user–artist listening data, plus **Personalized PageRank (PPR)** explanations on a heterogeneous graph (friends, co-listening, tags). Dashboard built with **Streamlit**.

## What you get

| Piece | Role |
|--------|------|
| `src/lightgcn.py` | LightGCN training (BPR) and top-K inference |
| `src/ppr.py` | PPR explainer on user / friend / artist graph |
| `app/streamlit_app.py` | Interactive dashboard |
| `scripts/download_data.py` | Fetches HetRec 2011 Last.fm 2K from GroupLens |
| `scripts/train.py` | Trains and saves `checkpoints/lightgcn_lastfm.pt` |

Dataset: [HetRec 2011 Last.fm](https://grouplens.org/datasets/hetrec-2011/) (non-commercial).

---

## Quick start (VS Code)

### 1. Open the project

1. Install [VS Code](https://code.visualstudio.com/) and the **Python** extension.
2. **File → Open Folder** → select `LastFm`.
3. **Terminal → New Terminal** (PowerShell).

### 2. Create a virtual environment

```powershell
cd C:\Users\anshs\LastFm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Select the interpreter: **Ctrl+Shift+P** → `Python: Select Interpreter` → `.venv`.

### 3. Download data and train

```powershell
python scripts/download_data.py
python scripts/train.py --quick
```

Full training (better quality, ~10–20 min CPU):

```powershell
python scripts/train.py --epochs 30
```

### 4. Run the dashboard

```powershell
streamlit run app/streamlit_app.py
```

Or:

```powershell
streamlit run lastfm_lightgcn_ppr_dashboard.py
```

Browser opens at `http://localhost:8501`.

---

## GitHub setup

### 1. Initialize git (once)

```powershell
cd C:\Users\anshs\LastFm
git init
git add .
git commit -m "Add LightGCN + PPR Streamlit dashboard for HetRec Last.fm"
```

### 2. Create a GitHub repo

1. [github.com/new](https://github.com/new) → name e.g. `lastfm-lightgcn-ppr`.
2. Do **not** add a README if you already have one locally.

```powershell
git remote add origin https://github.com/YOUR_USERNAME/lastfm-lightgcn-ppr.git
git branch -M main
git push -u origin main
```

`data/raw/` and `checkpoints/` are gitignored (download/train locally). Optionally use [Git LFS](https://git-lfs.com/) for checkpoints if you want to share weights.

### 3. VS Code + GitHub

- Install **GitHub Pull Requests and Issues**.
- **Source Control** (Ctrl+Shift+G) → **Publish Branch** or sync after `remote add`.
- `.vscode/settings.json` (optional): `"python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe"`

---

## Project layout

```
LastFm/
├── app/streamlit_app.py      # Dashboard
├── scripts/
│   ├── download_data.py
│   └── train.py
├── src/
│   ├── data.py               # HetRec loading & splits
│   ├── lightgcn.py           # Model
│   ├── ppr.py                # Explainer
│   └── metrics.py
├── data/raw/                 # Created by download (not in git)
├── checkpoints/              # Trained weights (not in git)
├── requirements.txt
└── README.md
```

---

## How it works (short)

1. **Data**: `user_artists.dat` → sparse user–artist matrix; 20% interactions held out per user for evaluation.
2. **LightGCN**: Normalized bipartite adjacency; layer-wise embedding propagation; mean pooling; BPR loss.
3. **Recommendations**: Score = ⟨user, artist⟩; mask artists already in training history.
4. **PPR explainer**: Random-walk restart from the selected user on edges for listening, friendships, co-listened artists, and tag overlap. Surfaces paths and high-PPR neighbors for a recommended artist.

---

## Troubleshooting

| Issue | Fix |
|--------|-----|
| `Dataset not found` | Run `python scripts/download_data.py` |
| Slow first dashboard load | No checkpoint → app runs 8-epoch demo train; run `scripts/train.py` first |
| CUDA | `python scripts/train.py --device cuda` if PyTorch sees a GPU |
| Streamlit port in use | `streamlit run app/streamlit_app.py --server.port 8502` |

---

## License note

HetRec Last.fm data is for **non-commercial** use. See the [GroupLens README](https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-readme.txt).
