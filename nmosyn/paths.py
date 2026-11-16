"""Locations of the inputs, the cached results, and the figures."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
EXTERNAL = DATA / 'external'        # public releases, fetched by scripts/download_data.py
RESULTS = ROOT / 'results'          # cached results of compute/, read by the figures and numbers.py
FIGURES = ROOT / 'output'           # PDFs of the figures, under the file names the paper includes
