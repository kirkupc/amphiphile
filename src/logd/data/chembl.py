"""Assemble a clean logD7.4 training set from ChEMBL.

Approach:
  - Query ChEMBL activities for standard_type IN ('LogD', 'LogD7.4').
  - Drop rows with missing SMILES or standard_value.
  - Canonicalise + desalt SMILES (features.canonicalise).
  - Drop outliers: logD outside [-5, 10] (clearly suspect).
  - Aggregate duplicates by InChIKey → median logD (handles intra-lab and inter-lab replicates).
  - Cache resulting parquet to data/chembl_logd.parquet.

Why ChEMBL: it's the standard source for this class of ADME data. The rationale for
preferring LogD7.4 over generic LogD is that the test target is physiological
partitioning; mixing assay pHs introduces ~0.5 log unit noise.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from logd.features import canonicalise
from logd.utils import data_dir, get_logger

LOG = get_logger(__name__)

CACHE_FILE = "chembl_logd.parquet"
MIN_LOGD = -5.0
MAX_LOGD = 10.0

# SQL for ChEMBL's public schema. Uses chembl_downloader to get a local sqlite
# copy; this gives reproducible results (ChEMBL version can be pinned).
_SQL = """
SELECT
    cs.canonical_smiles AS smiles,
    act.standard_value AS logd,
    act.standard_type AS assay_type,
    act.assay_id AS assay_id,
    md.chembl_id AS chembl_id
FROM activities act
JOIN assays a ON a.assay_id = act.assay_id
JOIN molecule_dictionary md ON md.molregno = act.molregno
JOIN compound_structures cs ON cs.molregno = act.molregno
WHERE act.standard_type IN ('LogD', 'LogD7.4')
  AND act.standard_value IS NOT NULL
  AND act.standard_relation = '='
  AND cs.canonical_smiles IS NOT NULL
"""


def fetch_raw(version: str | None = None) -> pd.DataFrame:
    """
    Pull raw logD rows from a locally-downloaded ChEMBL snapshot.

    `version` pins the ChEMBL release (e.g. "34"). None → latest.
    """
    try:
        import chembl_downloader
    except ImportError as e:
        raise RuntimeError(
            "chembl_downloader not installed. Run `uv sync` to install project deps."
        ) from e

    LOG.info("Querying ChEMBL (version=%s) for LogD activities", version or "latest")
    df = chembl_downloader.query(_SQL, version=version)
    LOG.info("Fetched %d raw rows", len(df))
    return df


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise, desalt, dedup, drop outliers. Returns columns [smiles, logd, inchikey]."""
    from rdkit import Chem

    df = raw.copy()
    df["logd"] = pd.to_numeric(df["logd"], errors="coerce")
    df = df.dropna(subset=["smiles", "logd"])

    df["canonical_smiles"] = df["smiles"].map(canonicalise)
    df = df.dropna(subset=["canonical_smiles"])

    df["inchikey"] = df["canonical_smiles"].map(
        lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s)) if s else None
    )
    df = df.dropna(subset=["inchikey"])

    df = df[(df["logd"] >= MIN_LOGD) & (df["logd"] <= MAX_LOGD)]

    # One row per compound — median over replicates.
    agg = (
        df.groupby("inchikey", as_index=False)
        .agg(smiles=("canonical_smiles", "first"), logd=("logd", "median"), n_obs=("logd", "size"))
        .reset_index(drop=True)
    )
    LOG.info("Cleaned to %d unique compounds", len(agg))
    return agg[["smiles", "logd", "inchikey", "n_obs"]]


def load(cache: Path | None = None, refresh: bool = False, version: str | None = None) -> pd.DataFrame:
    """Load cleaned ChEMBL logD data; fetch + clean if cache missing."""
    cache = cache or (data_dir() / CACHE_FILE)
    if cache.exists() and not refresh:
        LOG.info("Loading cached ChEMBL logD from %s", cache)
        return pd.read_parquet(cache)
    raw = fetch_raw(version=version)
    cleaned = clean(raw)
    cleaned.to_parquet(cache, index=False)
    LOG.info("Wrote cleaned ChEMBL logD to %s", cache)
    return cleaned
