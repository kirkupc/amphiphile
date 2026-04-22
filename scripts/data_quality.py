"""Data quality audit — the noise floor analysis.

OpenADMET's aggregated parquet gives us `logd_std` per compound (intra-compound
standard deviation across multiple ChEMBL assays). This is the empirical
noise floor — the best RMSE any model could achieve is bounded by how much
the reference data itself disagrees with itself.

This script:
  1. Loads the aggregated training set.
  2. Subsets to compounds measured in >= 2 assays (std is only meaningful there).
  3. Reports the distribution of intra-compound std.
  4. Estimates the noise floor as the median / mean intra-compound std.
  5. Saves a histogram + summary json to reports/.

Interpretation: if our scaffold-test RMSE is close to the noise floor, we're
near the ceiling of what any model can do with this data. If we're well above
it, there's modelling slack left.

Usage:
    uv run python scripts/data_quality.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from logd.data import openadmet_chembl
from logd.utils import get_logger, reports_dir

LOG = get_logger(__name__)


def run() -> None:
    LOG.info("Loading OpenADMET ChEMBL LogD aggregated data")
    df = openadmet_chembl.load()

    if "logd_std" not in df.columns or "n_assays" not in df.columns:
        LOG.warning("logd_std / n_assays columns not available; skipping audit")
        return

    # Compounds with replicate measurements
    multi = df[df["n_assays"] >= 2].copy()
    multi["logd_std"] = pd.to_numeric(multi["logd_std"], errors="coerce")
    multi = multi.dropna(subset=["logd_std"])

    LOG.info(
        "Total compounds: %d; with >=2 assay observations: %d (%.1f%%)",
        len(df),
        len(multi),
        100 * len(multi) / len(df),
    )

    std_stats = {
        "median": float(multi["logd_std"].median()),
        "mean": float(multi["logd_std"].mean()),
        "p90": float(multi["logd_std"].quantile(0.9)),
        "max": float(multi["logd_std"].max()),
    }
    stats: dict[str, object] = {
        "n_total": len(df),
        "n_replicate": len(multi),
        "intra_compound_std_log_units": std_stats,
        "implied_noise_floor_rmse_log_units": float(multi["logd_std"].mean()),
    }
    LOG.info(
        "Noise floor: median intra-compound std = %.3f log units (mean = %.3f)",
        std_stats["median"],
        std_stats["mean"],
    )

    # Histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        multi["logd_std"].clip(upper=2.0), bins=50, color="steelblue", edgecolor="black", alpha=0.8
    )
    ax.axvline(
        std_stats["median"],
        color="crimson",
        linestyle="--",
        label=f"median = {std_stats['median']:.3f}",
    )
    ax.set_xlabel("Intra-compound std across ChEMBL assays (log units)")
    ax.set_ylabel("Number of compounds")
    ax.set_title(f"LogD assay noise floor — {len(multi):,} compounds with ≥2 observations")
    ax.legend()
    fig.tight_layout()

    out_dir = reports_dir()
    fig.savefig(out_dir / "noise_floor.png", dpi=120)
    plt.close(fig)

    (out_dir / "noise_floor.json").write_text(json.dumps(stats, indent=2))
    LOG.info("Wrote noise_floor.png + noise_floor.json to %s", out_dir)


if __name__ == "__main__":
    run()
