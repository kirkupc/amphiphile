"""Load OpenADMET's curated ChEMBL35 LogD aggregated dataset.

OpenADMET publishes intake-catalog-managed parquets of curated ADMET data. Their
`ChEMBL_LogD` catalog provides both a raw per-activity parquet and an aggregated
per-compound parquet. We use the aggregated one as our primary training source.

Pinned to a specific commit SHA of the OpenADMET/data-catalogs repo so reviewers
re-running the code get identical training data.

Why this over a fresh `chembl_downloader` pull:
  - 1.1 MB download vs ~4 GB ChEMBL sqlite dump. Minutes vs hours to reproduce.
  - Curation is done by OpenADMET using their toolkit; it's more conservative
    than a naive SQL pull would be (handles assay heterogeneity, units, etc.).
  - Single pinned URL → reproducibility is trivial to audit.

Column names are normalised at load time because the catalog has shifted names
across revisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from logd.features import canonicalise
from logd.utils import data_dir, get_logger

LOG = get_logger(__name__)

# Pinned to OpenADMET/data-catalogs @ main as of 2026-04-20.
# Change this SHA to update to a newer curation; training must be re-run.
DATA_CATALOGS_SHA: Final[str] = "e1a9494f491339d902b3dfeaaa9ac83a3e0cf0d9"

AGGREGATED_URL: Final[str] = (
    f"https://raw.githubusercontent.com/OpenADMET/data-catalogs/{DATA_CATALOGS_SHA}/"
    "catalogs/activities/ChEMBL_LogD/ChEMBL35_LogD/ChEMBL_LogD_LogD_aggregated.parquet"
)
RAW_URL: Final[str] = (
    f"https://raw.githubusercontent.com/OpenADMET/data-catalogs/{DATA_CATALOGS_SHA}/"
    "catalogs/activities/ChEMBL_LogD/ChEMBL35_LogD/ChEMBL_LogD_LogD_raw.parquet"
)

CACHE_FILE: Final[str] = "openadmet_chembl_logd.parquet"
MIN_LOGD: Final[float] = -5.0
MAX_LOGD: Final[float] = 10.0

# Column names observed; normalise at load.
_SMILES_CANDIDATES = ("SMILES", "smiles", "canonical_smiles", "Canonical_SMILES")
_LOGD_CANDIDATES = ("LogD", "logD", "logd", "LOGD", "value", "standard_value", "logD7.4")


def _pick(cols: list[str], candidates: tuple[str, ...], label: str) -> str:
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"No {label} column from {candidates} found; got {cols}")


def fetch(cache: Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Download + clean OpenADMET ChEMBL35 LogD aggregated parquet.

    Returns a DataFrame with columns [smiles, logd, inchikey].
    Cached locally in parquet for repeat runs.
    """
    cache = cache or (data_dir() / CACHE_FILE)
    if cache.exists() and not refresh:
        LOG.info("Loading cached OpenADMET ChEMBL LogD from %s", cache)
        return pd.read_parquet(cache)

    LOG.info("Fetching OpenADMET ChEMBL LogD from %s", AGGREGATED_URL)
    df = pd.read_parquet(AGGREGATED_URL)
    LOG.info("Fetched %d raw rows with columns: %s", len(df), list(df.columns))

    smi_col = _pick(list(df.columns), _SMILES_CANDIDATES, "SMILES")
    y_col = _pick(list(df.columns), _LOGD_CANDIDATES, "LogD")

    df = df[[smi_col, y_col]].rename(columns={smi_col: "smiles", y_col: "logd"})
    df["logd"] = pd.to_numeric(df["logd"], errors="coerce")
    df = df.dropna(subset=["smiles", "logd"])
    df = df[(df["logd"] >= MIN_LOGD) & (df["logd"] <= MAX_LOGD)]

    df["smiles"] = df["smiles"].map(canonicalise)
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)

    # Compute InChIKey for cross-dataset dedup (vs ExpansionRx).
    from rdkit import Chem

    df["inchikey"] = df["smiles"].map(
        lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s)) if s else None
    )
    df = df.dropna(subset=["inchikey"])

    # One-compound-per-row after canonicalisation (aggregated is already
    # deduplicated by OpenADMET, but enforce it defensively).
    df = df.groupby("inchikey", as_index=False).agg(
        smiles=("smiles", "first"), logd=("logd", "median")
    )
    df = df[["smiles", "logd", "inchikey"]].reset_index(drop=True)

    df.to_parquet(cache, index=False)
    LOG.info("Wrote %d unique compounds to %s", len(df), cache)
    return df


def load(refresh: bool = False) -> pd.DataFrame:
    return fetch(refresh=refresh)
