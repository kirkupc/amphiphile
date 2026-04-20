"""Load the OpenADMET ExpansionRx challenge training set as our external benchmark.

This dataset is 5,039 LogD-labelled compounds from real drug-discovery
campaigns run by Expansion Therapeutics on RNA-mediated diseases. Published by
OpenADMET as the public training split of the ExpansionRx-OpenADMET Blind
Challenge. Held out from ChEMBL — a legitimate generalization test.

We treat this as an external eval only; it is never mixed into training.

Pinned to a specific Hugging Face dataset revision so re-runs are byte-identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from logd.features import canonicalise
from logd.utils import data_dir, get_logger

LOG = get_logger(__name__)

# Pinned to huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-train-data
# as of 2026-04-20. Update this SHA to pull newer data.
HF_DATASET_SHA: Final[str] = "f40a23ea75a56560e9a1ce6ddc1586dd6a1f25d9"

URL: Final[str] = (
    "https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-train-data/"
    f"resolve/{HF_DATASET_SHA}/expansion_data_train.csv"
)

CACHE_FILE: Final[str] = "expansionrx_logd.parquet"
SMILES_COL: Final[str] = "SMILES"
LOGD_COL: Final[str] = "LogD"


def fetch(cache: Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Download the ExpansionRx challenge CSV, filter to compounds with LogD labels.

    Returns DataFrame with columns [smiles, logd, inchikey].
    """
    cache = cache or (data_dir() / CACHE_FILE)
    if cache.exists() and not refresh:
        LOG.info("Loading cached ExpansionRx LogD from %s", cache)
        return pd.read_parquet(cache)

    LOG.info("Fetching ExpansionRx challenge data from %s", URL)
    df = pd.read_csv(URL)
    LOG.info("Fetched %d rows, %d with LogD labels", len(df), df[LOGD_COL].notna().sum())

    df = df[[SMILES_COL, LOGD_COL]].rename(columns={SMILES_COL: "smiles", LOGD_COL: "logd"})
    df = df.dropna(subset=["smiles", "logd"]).reset_index(drop=True)
    df["logd"] = pd.to_numeric(df["logd"], errors="coerce")
    df = df.dropna(subset=["logd"]).reset_index(drop=True)

    df["smiles"] = df["smiles"].map(canonicalise)
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)

    from rdkit import Chem

    df["inchikey"] = df["smiles"].map(
        lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s)) if s else None
    )
    df = df.dropna(subset=["inchikey"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["inchikey"]).reset_index(drop=True)

    df = df[["smiles", "logd", "inchikey"]]
    df.to_parquet(cache, index=False)
    LOG.info("Wrote %d unique LogD-labelled compounds to %s", len(df), cache)
    return df


def load(refresh: bool = False) -> pd.DataFrame:
    return fetch(refresh=refresh)
