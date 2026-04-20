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

# Actual columns in OpenADMET's aggregated parquet (verified 2026-04-20):
#   OPENADMET_CANONICAL_SMILES, OPENADMET_INCHIKEY
#   assay_id_count
#   standard_value_mean, standard_value_median, standard_value_std
#   pchembl_value_mean, pchembl_value_median, pchembl_value_std
# We use standard_value_median as the target (robust to assay outliers) and
# standard_value_std as the per-compound noise estimate (used later in the
# data-quality audit — no extra computation needed).
SMILES_COL: Final[str] = "OPENADMET_CANONICAL_SMILES"
INCHIKEY_COL: Final[str] = "OPENADMET_INCHIKEY"
LOGD_COL: Final[str] = "standard_value_median"
LOGD_STD_COL: Final[str] = "standard_value_std"
N_ASSAYS_COL: Final[str] = "assay_id_count"


def fetch(cache: Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Download + clean OpenADMET ChEMBL35 LogD aggregated parquet.

    Returns a DataFrame with columns:
      - smiles    — canonical SMILES
      - logd      — per-compound median across assays
      - logd_std  — per-compound std across assays (noise floor estimate)
      - n_assays  — number of assay observations
      - inchikey  — canonical InChIKey (used for cross-dataset dedup)

    Cached locally in parquet for repeat runs.
    """
    cache = cache or (data_dir() / CACHE_FILE)
    if cache.exists() and not refresh:
        LOG.info("Loading cached OpenADMET ChEMBL LogD from %s", cache)
        return pd.read_parquet(cache)

    LOG.info("Fetching OpenADMET ChEMBL LogD from %s", AGGREGATED_URL)
    df = pd.read_parquet(AGGREGATED_URL)
    LOG.info("Fetched %d raw rows with columns: %s", len(df), list(df.columns))

    required = [SMILES_COL, LOGD_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns missing: {missing}; got {list(df.columns)}")

    df = df.rename(
        columns={
            SMILES_COL: "smiles",
            LOGD_COL: "logd",
            LOGD_STD_COL: "logd_std",
            N_ASSAYS_COL: "n_assays",
            INCHIKEY_COL: "inchikey",
        }
    )
    df["logd"] = pd.to_numeric(df["logd"], errors="coerce")
    df = df.dropna(subset=["smiles", "logd"])
    df = df[(df["logd"] >= MIN_LOGD) & (df["logd"] <= MAX_LOGD)]

    # Canonicalise + desalt via our own RDKit pipeline; OpenADMET's canonical
    # form is fine, but we want consistency with ExpansionRx handling.
    df["smiles"] = df["smiles"].map(canonicalise)
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)

    # Recompute InChIKey for safety (OpenADMET precomputed one exists but we
    # canonicalised with desalting, which can change the key).
    from rdkit import Chem

    df["inchikey"] = df["smiles"].map(
        lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s)) if s else None
    )
    df = df.dropna(subset=["inchikey"])

    # Enforce one-compound-per-row after canonicalisation.
    df = df.groupby("inchikey", as_index=False).agg(
        smiles=("smiles", "first"),
        logd=("logd", "median"),
        logd_std=("logd_std", "median") if "logd_std" in df.columns else ("logd", "size"),
        n_assays=("n_assays", "sum") if "n_assays" in df.columns else ("logd", "size"),
    )
    keep_cols = ["smiles", "logd", "inchikey"]
    if "logd_std" in df.columns:
        keep_cols.append("logd_std")
    if "n_assays" in df.columns:
        keep_cols.append("n_assays")
    df = df[keep_cols].reset_index(drop=True)

    df.to_parquet(cache, index=False)
    LOG.info("Wrote %d unique compounds to %s", len(df), cache)
    return df


def load(refresh: bool = False) -> pd.DataFrame:
    return fetch(refresh=refresh)
