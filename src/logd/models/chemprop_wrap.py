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
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from logd.utils import get_logger, set_seed

LOG = get_logger(__name__)

DEFAULT_MAX_EPOCHS = 50
# Batch size 32: avoids a segfault in Chemprop's BatchMolGraph collation that
# occurs when large drug-like molecules (avg ~30 heavy atoms) are batched at
# size 64+.  The crash is a numpy/PyTorch memory corruption triggered by
# torch.from_numpy on the concatenated graph arrays.  Size 32 stays well under
# the threshold while keeping training speed reasonable.
DEFAULT_BATCH_SIZE = 32
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


def _safe_collate_batch(batch):
    """Collate with explicit numpy copies to avoid SIGSEGV from CoW + torch.from_numpy.

    Chemprop's ``BatchMolGraph.__post_init__`` calls ``torch.from_numpy`` on
    arrays concatenated from per-molecule ``MolGraph`` numpy buffers.  With
    NumPy >= 2.0 copy-on-write semantics, those buffers may share a backing
    store; ``torch.from_numpy`` creates a zero-copy tensor view.  If the CoW
    mechanism later reallocates the numpy side, the tensor is left pointing at
    freed memory, causing a SIGSEGV inside Lightning's training loop.

    We work around this by copying each ``MolGraph``'s arrays before
    ``BatchMolGraph`` is constructed so every tensor owns its own memory.
    """
    from chemprop.data.collate import BatchMolGraph, TrainingBatch

    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch)

    # Deep-copy the numpy arrays inside each MolGraph so torch.from_numpy
    # gets a stable, owned buffer instead of a CoW view.
    safe_mgs = []
    for mg in mgs:
        mg.V = np.array(mg.V, copy=True)
        mg.E = np.array(mg.E, copy=True)
        mg.edge_index = np.array(mg.edge_index, copy=True)
        mg.rev_edge_index = np.array(mg.rev_edge_index, copy=True)
        safe_mgs.append(mg)

    return TrainingBatch(
        BatchMolGraph(safe_mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        None if ys[0] is None else torch.from_numpy(np.array(ys)).float(),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )


def _build_loader(smiles: list[str], y: np.ndarray | None, batch_size: int, shuffle: bool):
    from torch.utils.data import DataLoader

    from chemprop import data, featurizers

    datapoints = _build_datapoints(smiles, y)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dataset = data.MoleculeDataset(datapoints, featurizer)

    # We bypass ``chemprop.data.build_dataloader`` so we can inject our
    # ``_safe_collate_batch`` (see docstring above).  ``build_dataloader``
    # hard-codes ``collate_fn`` and doesn't accept an override.
    drop_last = len(dataset) % batch_size == 1  # matches Chemprop's auto logic
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
        accelerator: str | None = None,
    ) -> None:
        """Train k ensemble members. Checkpoints saved to self.checkpoint_dir."""
        import lightning.pytorch as pl

        # Default accelerator: honour LOGD_CHEMPROP_ACCELERATOR env var.
        # Default to "cpu" rather than "auto" because Chemprop's batched
        # molecular-graph tensors trigger segfaults / silent hangs on MPS
        # (Apple Silicon).  The root cause is a numpy/PyTorch memory
        # corruption in BatchMolGraph collation when large drug-like
        # molecules are moved to the MPS device.  CPU is only ~5x slower
        # for D-MPNN training and avoids the issue entirely.
        if accelerator is None:
            accelerator = os.environ.get("LOGD_CHEMPROP_ACCELERATOR", "cpu")
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
        import lightning.pytorch as pl

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
