"""Profile the end-to-end prediction pipeline across batch sizes.

Measures:
  - wall time (perf_counter)
  - throughput (mol / s)
  - peak RSS (psutil)
  - per-stage breakdown:
      parse+canonicalise  (RDKit MolFromSmiles + SaltRemover)
      featurise           (descriptors + Morgan)
      model forward       (ensemble predict)
      uncertainty+flag    (AD Tanimoto + conformal + reliability)

Input compounds: ExpansionRx training data, drawn down to each batch size. This
is a realistic drug-like distribution held out from training, so no cache hits.

Usage:
    uv run python scripts/profile_inference.py
    uv run python scripts/profile_inference.py --batch-sizes 1 100 1000 10000
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import psutil

from logd.data import expansionrx
from logd.features import FeatureSpec, featurise_batch, mol_from_smiles, morgan_fp
from logd.inference import LoadedModel, load_model
from logd.utils import get_logger, reports_dir

LOG = get_logger(__name__)


@dataclass
class StageTiming:
    parse_canon_s: float
    featurise_s: float
    model_forward_s: float
    uncertainty_s: float

    @property
    def total_s(self) -> float:
        return self.parse_canon_s + self.featurise_s + self.model_forward_s + self.uncertainty_s


@dataclass
class BatchResult:
    batch_size: int
    n_valid: int
    wall_time_s: float
    throughput_mol_per_s: float
    peak_rss_mb: float
    stages: StageTiming


def _time(fn: Callable[[], object]) -> tuple[float, object]:
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def _profile_batch(smiles: list[str], model: LoadedModel) -> BatchResult:
    proc = psutil.Process()
    peak_rss = proc.memory_info().rss

    # Stage 1: parse + canonicalise (implicit inside featurise_batch, but
    # we isolate it by calling mol_from_smiles separately to attribute time).
    t_parse, _mols = _time(lambda: [mol_from_smiles(s) for s in smiles])
    peak_rss = max(peak_rss, proc.memory_info().rss)

    # Stage 2: featurise — descriptors + Morgan
    t_feat, (X, mask) = _time(lambda: featurise_batch(smiles, model.feature_spec))
    peak_rss = max(peak_rss, proc.memory_info().rss)

    # Stage 3: model forward
    t_model, (y_pred, y_std) = _time(lambda: model.baseline.predict(X))
    peak_rss = max(peak_rss, proc.memory_info().rss)

    # Stage 4: uncertainty + reliability
    def _unc() -> np.ndarray:
        if X.shape[0] == 0:
            return np.zeros(0)
        valid = [s for s, m in zip(smiles, mask) if m]
        fps = np.stack([morgan_fp(mol_from_smiles(s)) for s in valid], axis=0)
        nn = model.reliability.ad.nearest_similarity(fps)
        flags = model.reliability.flag(y_std, nn)
        return flags

    t_unc, _ = _time(_unc)
    peak_rss = max(peak_rss, proc.memory_info().rss)

    stages = StageTiming(
        parse_canon_s=t_parse,
        featurise_s=t_feat,
        model_forward_s=t_model,
        uncertainty_s=t_unc,
    )
    total = stages.total_s
    n_valid = int(mask.sum())
    return BatchResult(
        batch_size=len(smiles),
        n_valid=n_valid,
        wall_time_s=total,
        throughput_mol_per_s=len(smiles) / total if total > 0 else 0.0,
        peak_rss_mb=peak_rss / (1024 * 1024),
        stages=stages,
    )


def run(batch_sizes: list[int], output: Path | None = None) -> None:
    output = output or (reports_dir() / "profiling.json")
    LOG.info("Loading model")
    model = load_model()
    LOG.info("Loading ExpansionRx compounds for realistic distribution")
    df = expansionrx.load()
    pool = df["smiles"].tolist()
    rng = np.random.default_rng(0)

    results: list[BatchResult] = []
    for bs in batch_sizes:
        if bs > len(pool):
            LOG.warning(
                "Requested batch %d > available pool %d; sampling with replacement", bs, len(pool)
            )
            idx = rng.integers(0, len(pool), size=bs)
        else:
            idx = rng.choice(len(pool), size=bs, replace=False)
        batch = [pool[i] for i in idx]

        # Warmup on first batch only (LightGBM booster init, imports)
        if bs == batch_sizes[0]:
            LOG.info("Warmup pass (batch=%d)", min(bs, 10))
            _profile_batch(batch[: min(bs, 10)], model)

        gc.collect()
        LOG.info("Profiling batch=%d", bs)
        r = _profile_batch(batch, model)
        results.append(r)
        LOG.info(
            "  total=%.3fs, throughput=%.1f mol/s, peak_rss=%.1f MB",
            r.wall_time_s,
            r.throughput_mol_per_s,
            r.peak_rss_mb,
        )
        LOG.info(
            "  stages: parse=%.3f featurise=%.3f model=%.3f uncertainty=%.3f",
            r.stages.parse_canon_s,
            r.stages.featurise_s,
            r.stages.model_forward_s,
            r.stages.uncertainty_s,
        )

    # Markdown table for README
    lines = [
        "| Batch | Total (s) | Throughput (mol/s) | Parse (s) | Featurise (s) | Model (s) | Uncertainty (s) | Peak RSS (MB) |",
        "|------:|----------:|-------------------:|----------:|--------------:|----------:|----------------:|--------------:|",
    ]
    for r in results:
        lines.append(
            f"| {r.batch_size} | {r.wall_time_s:.3f} | {r.throughput_mol_per_s:.1f} | "
            f"{r.stages.parse_canon_s:.3f} | {r.stages.featurise_s:.3f} | "
            f"{r.stages.model_forward_s:.3f} | {r.stages.uncertainty_s:.3f} | "
            f"{r.peak_rss_mb:.1f} |"
        )
    table_md = "\n".join(lines)

    out = {
        "batch_results": [
            {**asdict(r), "stages": asdict(r.stages)} for r in results
        ],
        "table_md": table_md,
    }
    output.write_text(json.dumps(out, indent=2))
    LOG.info("Wrote %s", output)
    print("\n" + table_md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 100, 1000, 10000]
    )
    args = parser.parse_args()
    run(batch_sizes=args.batch_sizes)
