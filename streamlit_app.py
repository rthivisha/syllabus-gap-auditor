"""Streamlit Cloud entrypoint proxying to ui/app.py."""
import runpy
from pathlib import Path

_target = Path(__file__).resolve().parent / "ui" / "app.py"
runpy.run_path(str(_target), run_name="__main__")
