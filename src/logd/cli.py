"""CLI entry point.

Subcommands:
  prepare-data  — fetch + cache ChEMBL + OpenADMET
  train         — train baseline ensemble and calibrate reliability
  predict       — inference from SMILES strings or a file
  profile       — run the batch-size profiler
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from logd.utils import get_logger

LOG = get_logger(__name__)

app = typer.Typer(add_completion=False, no_args_is_help=True, help="logD prediction CLI.")


@app.command("prepare-data")
def prepare_data(refresh: bool = typer.Option(False, help="Re-fetch even if cached")) -> None:
    """Fetch + clean ChEMBL logD and OpenADMET benchmark."""
    from logd.data import chembl, openadmet

    df = chembl.load(refresh=refresh)
    typer.echo(f"ChEMBL: {len(df)} compounds")
    oa = openadmet.load(refresh=refresh)
    typer.echo(f"OpenADMET: {len(oa)} compounds")


@app.command("train")
def train(
    model: str = typer.Option("baseline", help="baseline | chemprop"),
    seed: int = typer.Option(0),
    k: int = typer.Option(5, help="Ensemble size"),
) -> None:
    """Train a model. baseline is Day-1 ready; chemprop lands Day 2."""
    if model == "baseline":
        from logd.training import train_baseline

        metrics = train_baseline(seed=seed, k=k)
        typer.echo(json.dumps(metrics, indent=2))
    elif model == "chemprop":
        from logd.training import train_chemprop

        metrics = train_chemprop(seed=seed, k=k)
        typer.echo(json.dumps(metrics, indent=2))
    else:
        raise typer.BadParameter(f"Unknown model: {model}")


@app.command("predict")
def predict_cmd(
    smiles: list[str] = typer.Option(None, "--smiles", help="One or more SMILES (repeatable)"),
    input_file: Path = typer.Option(None, "--input-file", help="File with one SMILES per line"),
) -> None:
    """Predict logD + uncertainty + reliability. Outputs JSON lines."""
    from logd.inference import load_model, predict as run_predict

    strs: list[str] = list(smiles or [])
    if input_file:
        strs.extend([line.strip() for line in input_file.read_text().splitlines() if line.strip()])
    if not strs:
        raise typer.BadParameter("Provide --smiles or --input-file")

    model = load_model()
    for pred in run_predict(strs, model=model):
        typer.echo(json.dumps(pred.as_dict()))


@app.command("profile")
def profile_cmd(
    batch_sizes: list[int] = typer.Option([1, 100, 1000, 10000], "--batch-size"),
    output: Path = typer.Option(Path("reports/profiling.json")),
) -> None:
    """Run the inference profiler. Day-4 deliverable; stub here."""
    typer.echo("Profiler lands Day 4 — see PLAN.md §4.")
    typer.echo(f"(would profile batch sizes: {batch_sizes} → {output})")


if __name__ == "__main__":
    app()
