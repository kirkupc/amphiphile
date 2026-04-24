"""CLI entry point.

Subcommands:
  prepare-data  — fetch + cache ChEMBL + OpenADMET
  train         — train baseline ensemble and calibrate reliability
  predict       — inference from SMILES strings or a file
  profile       — run the batch-size profiler
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from logd.utils import get_logger

LOG = get_logger(__name__)

app = typer.Typer(add_completion=False, no_args_is_help=True, help="logD prediction CLI.")


def _ensure_scripts_importable() -> None:
    """Add project root to sys.path so `from scripts.X import ...` works."""
    src_dir = Path(__file__).resolve().parent.parent  # src/
    project_root = src_dir.parent
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


@app.command("prepare-data")
def prepare_data(refresh: bool = typer.Option(False, help="Re-fetch even if cached")) -> None:
    """Fetch + clean OpenADMET ChEMBL35 LogD + ExpansionRx benchmark."""
    from logd.data import expansionrx, openadmet_chembl

    typer.echo("Fetching OpenADMET ChEMBL35 LogD...")
    df = openadmet_chembl.load(refresh=refresh)
    typer.echo(f"  ✓ {len(df)} compounds")
    typer.echo("Fetching ExpansionRx benchmark...")
    oa = expansionrx.load(refresh=refresh)
    typer.echo(f"  ✓ {len(oa)} compounds")


@app.command("train")
def train(
    model: str = typer.Option("baseline", help="baseline | chemprop"),
    seed: int = typer.Option(0),
    k: int = typer.Option(5, help="Ensemble size"),
    tune: bool = typer.Option(True, help="Run hyperparameter search (baseline only)"),
) -> None:
    """Train a model. baseline is Day-1 ready; chemprop lands Day 2."""
    if model == "baseline":
        from logd.training import train_baseline

        typer.echo(f"Training baseline ensemble (k={k}, seed={seed}, tune={tune})...")
        metrics = train_baseline(seed=seed, k=k, tune=tune)
        typer.echo(json.dumps(metrics, indent=2))
    elif model == "chemprop":
        from logd.training import train_chemprop

        typer.echo(f"Training Chemprop D-MPNN ensemble (k={k}, seed={seed})...")
        metrics = train_chemprop(seed=seed, k=k)
        typer.echo(json.dumps(metrics, indent=2))
    else:
        raise typer.BadParameter(f"Unknown model: {model}")


@app.command("predict")
def predict_cmd(
    smiles: list[str] = typer.Option(None, "--smiles", help="One or more SMILES (repeatable)"),
    input_file: Path = typer.Option(None, "--input-file", help="File with one SMILES per line"),
    model: str = typer.Option("baseline", help="baseline | chemprop"),
) -> None:
    """Predict logD + uncertainty + reliability. Outputs JSON lines."""
    from logd.inference import load_model
    from logd.inference import predict as run_predict

    strs: list[str] = list(smiles or [])
    if input_file:
        if not input_file.exists():
            raise typer.BadParameter(f"File not found: {input_file}")
        strs.extend([line.strip() for line in input_file.read_text().splitlines() if line.strip()])
    if not strs:
        raise typer.BadParameter("Provide --smiles or --input-file")

    loaded = load_model(model_type=model)
    for pred in run_predict(strs, model=loaded):
        typer.echo(json.dumps(pred.as_dict()))


@app.command("profile")
def profile_cmd(
    batch_sizes: list[int] = typer.Option([1, 100, 1000, 10000], "--batch-size"),
    output: Path = typer.Option(Path("reports/profiling.json")),
) -> None:
    """Profile inference pipeline across batch sizes."""
    _ensure_scripts_importable()
    from scripts.profile_inference import run

    run(batch_sizes=batch_sizes, output=output)


@app.command("error-analysis")
def error_analysis_cmd(top: int = typer.Option(10)) -> None:
    """Worst-N compounds on ExpansionRx with structures + rationales."""
    _ensure_scripts_importable()
    from scripts.error_analysis import run

    run(top=top)


@app.command("diagnostics")
def diagnostics_cmd() -> None:
    """Model diagnostics: bias analysis, confidence curves, feature importance."""
    _ensure_scripts_importable()
    from scripts.model_diagnostics import run

    run()


@app.command("data-quality")
def data_quality_cmd() -> None:
    """Audit intra-compound std in ChEMBL to estimate the noise floor."""
    _ensure_scripts_importable()
    from scripts.data_quality import run

    run()


if __name__ == "__main__":
    app()
