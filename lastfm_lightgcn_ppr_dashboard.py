"""Entry point: streamlit run lastfm_lightgcn_ppr_dashboard.py"""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "app" / "streamlit_app.py"), run_name="__main__")
