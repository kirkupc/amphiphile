"""Load the OpenADMET logD benchmark set.

OpenADMET publishes curated ADMET benchmarks; we use their logD set as the
external test target. Source URL is pinned below; caching to data/ for
reproducibility.

NOTE: this module is intentionally permissive about the file format. The
benchmark has shifted column names across releases; we normalise at load time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from logd.features import canonicalise
from logd.utils import data_dir, get_logger

LOG = get_logger(__name__)

CACHE_FILE: Final[str] = "openadmet_logd.parquet"

# Primary source; update if OpenADMET re-organises the repo.
# Kept as a constant so the value is the one thing to change.
DEFAULT_URL = (
    "https://raw.githubusercontent.com/OpenADMET/openadmet-benchmarks/main/"
    "datasets/logd/openadmet_logd.csv"
)

# Candidate column names seen across OpenADMET releases.
_SMILES_CANDIDATES = ("SMILES", "smiles", "canonical_smiles", "Canonical_SMILES")
_LOGD_CANDIDATES = ("logD", "LogD", "logd", "logD7.4", "LogD7.4", "Y")


def _pick(cols: list[str], candidates: tuple[str, ...]) -> str:
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"No column from {candidates} found; got {cols}")


def fetch(url: str = DEFAULT_URL, cache: Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Download the OpenADMET logD CSV, normalise columns, cache as parquet."""
    cache = cache or (data_dir() / CACHE_FILE)
    if cache.exists() and not refresh:
        LOG.info("Loading cached OpenADMET logD from %s", cache)
        return pd.read_parquet(cache)

    LOG.info("Fetching OpenADMET logD from %s", url)
    df = pd.read_csv(url)
    smi_col = _pick(list(df.columns), _SMILES_CANDIDATES)
    y_col = _pick(list(df.columns), _LOGD_CANDIDATES)
    df = df[[smi_col, y_col]].rename(columns={smi_col: "smiles", y_col: "logd"})
    df["smiles"] = df["smiles"].map(canonicalise)
    df = df.dropna(subset=["smiles", "logd"]).reset_index(drop=True)
    df["logd"] = pd.to_numeric(df["logd"], errors="coerce")
    df = df.dropna(subset=["logd"]).reset_index(drop=True)
    df.to_parquet(cache, index=False)
    LOG.info("Wrote OpenADMET logD (%d compounds) to %s", len(df), cache)
    return df


def load(refresh: bool = False) -> pd.DataFrame:
    return fetch(refresh=refresh)
