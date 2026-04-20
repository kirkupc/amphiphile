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
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from logd.utils import get_logger, set_seed

LOG = get_logger(__name__)

DEFAULT_MAX_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 0  # Chemprop's default; raises on macOS if we set >0 here


def _build_model() -> "chemprop.models.MPNN":  # noqa: F821
    """Fresh Chemprop D-MPNN with conservative defaults.

    - BondMessagePassing: the standard D-MPNN edge update (Yang et al. 2019).
    - MeanAggregation: graph-level pooling; robust across molecule sizes.
    - RegressionFFN: MSE head for scalar logD.
    - batch_norm: standard in Chemprop's published recipes.
    """
    from chemprop import models, nn
    from chemprop.nn.metrics import MAE, RMSE

    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    ffn = nn.RegressionFFN()
    return models.MPNN(mp, agg, ffn, batch_norm=True, metrics=[RMSE(), MAE()])


def _build_datapoints(smiles: list[str], y: np.ndarray | None = None):
    """Map (smiles, y) → Chemprop MoleculeDatapoints. Invalid SMILES are filtered upstream."""
    from chemprop import data

    if y is None:
        return [data.MoleculeDatapoint.from_smi(s, [float("nan")]) for s in smiles]
    return [data.MoleculeDatapoint.from_smi(s, [float(yi)]) for s, yi in zip(smiles, y)]


def _build_loader(smiles: list[str], y: np.ndarray | None, batch_size: int, shuffle: bool):
    from chemprop import data, featurizers

    datapoints = _build_datapoints(smiles, y)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dataset = data.MoleculeDataset(datapoints, featurizer)
    return data.build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=DEFAULT_NUM_WORKERS,
        shuffle=shuffle,
    )


@dataclass
class ChempropModel:
    """k-member Chemprop ensemble, loaded lazily per prediction call.

    The Lightning models are held in a list; on save we write k checkpoints to
    a directory. Prediction runs all k forward passes and returns (mean, std).
    """

    checkpoint_dir: Path
    k: int
    _models: list | None = None  # populated on first predict() after load()

    @property
    def models(self) -> list:
        if self._models is None:
            self._models = self._load_models()
        return self._models

    # ------------------------------------------------------------------
    # Training
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
    ) -> None:
        """Train k ensemble members. Checkpoints saved to self.checkpoint_dir."""
        import pytorch_lightning as pl

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
                accelerator="auto",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
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
    # Prediction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_smiles(self, smiles: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict (mean, std, valid_mask) across the ensemble.

        Invalid SMILES are identified up front (Chemprop raises on parse failure);
        we filter those, predict on the remainder, and return an aligned mask.
        """
        from rdkit import Chem

        valid_mask = np.array(
            [Chem.MolFromSmiles(s) is not None if isinstance(s, str) else False for s in smiles],
            dtype=bool,
        )
        valid_smiles = [s for s, m in zip(smiles, valid_mask) if m]

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
    def _predict_member(self, model, smiles: list[str]) -> np.ndarray:
        import pytorch_lightning as pl

        loader = _build_loader(smiles, None, DEFAULT_BATCH_SIZE, shuffle=False)
        trainer = pl.Trainer(
            accelerator="auto", devices=1, logger=False, enable_progress_bar=False
        )
        # trainer.predict returns a list of batch-sized prediction tensors.
        batches = trainer.predict(model, loader)
        return torch.cat(batches, dim=0).squeeze(-1).cpu().numpy()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def _save_member(self, model, path: Path) -> None:
        from chemprop.models import save_model

        save_model(path, model)

    def _load_models(self) -> list:
        from chemprop.models import load_model

        config = json.loads((self.checkpoint_dir / "config.json").read_text())
        k = config["k"]
        return [load_model(self.checkpoint_dir / f"model_{i}.pt") for i in range(k)]

    @classmethod
    def load(cls, checkpoint_dir: Path) -> "ChempropModel":
        checkpoint_dir = Path(checkpoint_dir)
        config = json.loads((checkpoint_dir / "config.json").read_text())
        return cls(checkpoint_dir=checkpoint_dir, k=int(config["k"]))
