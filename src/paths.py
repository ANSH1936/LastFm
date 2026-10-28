from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CHECKPOINTS = ROOT / "checkpoints"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

DEFAULT_CHECKPOINT = CHECKPOINTS / "lightgcn_lastfm.pt"
