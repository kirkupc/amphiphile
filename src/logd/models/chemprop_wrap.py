"""Chemprop v2 D-MPNN wrapper.

Trained as a k-member deep ensemble. Exposes the same (mean, std) prediction
surface as BaselineModel so inference.py can stay model-agnostic.

Chemprop handles its own featurisation from SMILES directly (molecular graph),
so we do NOT pass pre-computed descriptors here. The baseline's descriptor +
Morgan block is a distinct feature set used only by the LightGBM ensemble.

Serialisation: a k-member model is stored as a directory with k `model_{i}.pt`
checkpoints plus a `config.json` describing ensemble size. This matches
Chemprop's single-model save format (`chemprop.models.save_model`) repeated k
times, so each checkpoint is independently loadable with Chemprop's own API.

Note on collation: Chemprop's BatchMolGraph.__post_init__ uses
torch.from_numpy (zero-copy). With NumPy >= 2.0 copy-on-write semantics this
can cause SIGSEGV on macOS ARM. We provide _safe_collate_batch that deep-copies
arrays before conversion, used in _build_loader for both training and inference.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from logd.utils import get_logger, set_seed

LOG = get_logger(__name__)

DEFAULT_MAX_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 0


def _build_model() -> Any:
    from chemprop import models, nn
    from chemprop.nn.metrics import MAE, RMSE

    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    ffn = nn.RegressionFFN()
    return models.MPNN(mp, agg, ffn, batch_norm=True, metrics=[RMSE(), MAE()])


def _build_datapoints(smiles: list[str], y: np.ndarray | None = None) -> list[Any]:
    from chemprop import data

    if y is None:
        return [data.MoleculeDatapoint.from_smi(s, [float("nan")]) for s in smiles]
    return [
        data.MoleculeDatapoint.from_smi(s, [float(yi)]) for s, yi in zip(smiles, y, strict=True)
    ]


def _safe_collate_batch(batch: list[Any]) -> Any:
    """Collate with owned copies to avoid SIGSEGV from NumPy CoW + torch.from_numpy.

    Deep-copies every MolGraph numpy array, then converts via torch.tensor
    (which always copies) instead of torch.from_numpy (zero-copy).
    """
    from chemprop.data.collate import BatchMolGraph, TrainingBatch

    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch, strict=True)

    Vs, Es, edge_indexes, rev_edge_indexes, batch_indexes = [], [], [], [], []
    num_nodes = 0
    num_edges = 0

    for i, mg in enumerate(mgs):
        Vs.append(np.array(mg.V, dtype=np.float32, copy=True))
        Es.append(np.array(mg.E, dtype=np.float32, copy=True))
        edge_indexes.append(np.array(mg.edge_index, dtype=np.int64, copy=True) + num_nodes)
        rev_edge_indexes.append(np.array(mg.rev_edge_index, dtype=np.int64, copy=True) + num_edges)
        batch_indexes.extend([i] * mg.V.shape[0])
        num_nodes += mg.V.shape[0]
        num_edges += mg.edge_index.shape[1]

    bmg = object.__new__(BatchMolGraph)
    object.__setattr__(bmg, "_BatchMolGraph__size", len(mgs))
    object.__setattr__(bmg, "V", torch.tensor(np.concatenate(Vs)))
    object.__setattr__(bmg, "E", torch.tensor(np.concatenate(Es)))
    object.__setattr__(bmg, "edge_index", torch.tensor(np.hstack(edge_indexes)))
    object.__setattr__(bmg, "rev_edge_index", torch.tensor(np.concatenate(rev_edge_indexes)))
    object.__setattr__(bmg, "batch", torch.tensor(np.array(batch_indexes, dtype=np.int64)))

    return TrainingBatch(
        bmg,
        None if V_ds[0] is None else torch.tensor(np.concatenate(V_ds), dtype=torch.float),
        None if x_ds[0] is None else torch.tensor(np.array(x_ds), dtype=torch.float),
        None if ys[0] is None else torch.tensor(np.array(ys), dtype=torch.float),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_masks[0] is None else torch.tensor(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.tensor(np.array(gt_masks)),
    )


def _build_loader(smiles: list[str], y: np.ndarray | None, batch_size: int, shuffle: bool) -> Any:
    from chemprop import data, featurizers
    from torch.utils.data import DataLoader

    datapoints = _build_datapoints(smiles, y)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dataset = data.MoleculeDataset(datapoints, featurizer)

    drop_last = len(dataset) % batch_size == 1
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=DEFAULT_NUM_WORKERS,
        collate_fn=_safe_collate_batch,
        drop_last=drop_last,
    )


@dataclass
class ChempropModel:
    """k-member Chemprop ensemble, loaded lazily per prediction call."""

    checkpoint_dir: Path
    k: int
    _models: list | None = None

    @property
    def models(self) -> list:
        if self._models is None:
            self._models = self._load_models()
        return self._models

    # ------------------------------------------------------------------
    # Training (designed for Colab GPU; uses Lightning Trainer)
    # ------------------------------------------------------------------
    def train(
        self,
        train_smiles: list[str],
        train_y: np.ndarray,
        val_smiles: list[str],
        val_y: np.ndarray,
        k: int = 5,
        max_epochs: int = DEFAULT_MAX_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        base_seed: int = 0,
        accelerator: str | None = None,
    ) -> None:
        """Train k ensemble members. Checkpoints saved to self.checkpoint_dir."""
        import lightning.pytorch as pl

        if accelerator is None:
            accelerator = os.environ.get("LOGD_CHEMPROP_ACCELERATOR", "auto")
        LOG.info("Chemprop accelerator: %s", accelerator)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.k = k

        for i in range(k):
            seed = base_seed + i
            set_seed(seed)
            pl.seed_everything(seed, workers=True)
            LOG.info("Training Chemprop member %d/%d (seed=%d)", i + 1, k, seed)

            model = _build_model()
            train_loader = _build_loader(train_smiles, train_y, batch_size, shuffle=True)
            val_loader = _build_loader(val_smiles, val_y, batch_size, shuffle=False)

            trainer = pl.Trainer(
                max_epochs=max_epochs,
                accelerator=accelerator,
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=True,
                enable_model_summary=(i == 0),
                gradient_clip_val=1.0,
            )
            trainer.fit(model, train_loader, val_loader)

            ckpt_path = self.checkpoint_dir / f"model_{i}.pt"
            self._save_member(model, ckpt_path)

        (self.checkpoint_dir / "config.json").write_text(
            json.dumps({"k": k, "model_type": "chemprop_v2_dmpnn"})
        )
        LOG.info("Saved %d Chemprop checkpoints to %s", k, self.checkpoint_dir)

    # ------------------------------------------------------------------
    # Prediction (works locally — safe collation avoids macOS SIGSEGV)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_smiles(self, smiles: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict (mean, std, valid_mask) across the ensemble."""
        from rdkit import Chem

        valid_mask = np.array(
            [Chem.MolFromSmiles(s) is not None if isinstance(s, str) else False for s in smiles],
            dtype=bool,
        )
        valid_smiles = [s for s, m in zip(smiles, valid_mask, strict=True) if m]

        if not valid_smiles:
            n = len(smiles)
            return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32), valid_mask

        preds_per_member = np.stack(
            [self._predict_member(m, valid_smiles) for m in self.models], axis=0
        )
        mean = preds_per_member.mean(axis=0)
        std = preds_per_member.std(axis=0)
        return mean.astype(np.float32), std.astype(np.float32), valid_mask

    @torch.no_grad()
    def _predict_member(self, model: Any, smiles: list[str]) -> np.ndarray:
        model.eval()
        loader = _build_loader(smiles, None, DEFAULT_BATCH_SIZE, shuffle=False)
        all_preds = []
        for batch in loader:
            preds = model(batch.bmg, batch.V_d, batch.X_d)
            all_preds.append(preds.cpu())
        return torch.cat(all_preds, dim=0).squeeze(-1).numpy()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def _save_member(self, model: Any, path: Path) -> None:
        from chemprop.models import save_model

        save_model(path, model)

    def _load_models(self) -> list[Any]:
        from chemprop.models import load_model

        config = json.loads((self.checkpoint_dir / "config.json").read_text())
        k = config["k"]
        return [load_model(self.checkpoint_dir / f"model_{i}.pt") for i in range(k)]

    @classmethod
    def load(cls, checkpoint_dir: Path) -> ChempropModel:
        checkpoint_dir = Path(checkpoint_dir)
        config = json.loads((checkpoint_dir / "config.json").read_text())
        return cls(checkpoint_dir=checkpoint_dir, k=int(config["k"]))
