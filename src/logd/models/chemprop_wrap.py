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

macOS ARM fix
-------------
Chemprop's ``BatchMolGraph.__post_init__`` uses ``torch.from_numpy`` for
zero-copy tensor creation.  With NumPy >= 2.0 copy-on-write semantics on
macOS ARM (Apple Silicon), those shared buffers cause use-after-free
(SIGSEGV / exit 139) once the numpy side reallocates.

We fix this with two changes:

1. **Safe collation** – ``_safe_collate_batch`` replaces the default
   ``collate_batch``.  It deep-copies every ``MolGraph`` numpy array with
   ``np.array(..., copy=True)`` then converts via ``torch.tensor`` (which
   always copies) instead of ``torch.from_numpy``.

2. **Manual training loop** – ``_train_member_manual`` replaces
   ``lightning.Trainer.fit()``.  Lightning's internal bookkeeping
   (``setup_data``, ``configure_optimizers`` → ``estimated_stepping_batches``)
   iterates the DataLoader an extra time before training even starts,
   doubling the number of numpy→torch conversions and triggering the
   SIGSEGV at scale (>~250 molecules).  A plain PyTorch loop avoids this.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from logd.utils import get_logger, set_seed

LOG = get_logger(__name__)

DEFAULT_MAX_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 0  # Chemprop's default; raises on macOS if we set >0 here
DEFAULT_LR = 1e-3


def _build_model() -> "chemprop.models.MPNN":  # noqa: F821
    """Fresh Chemprop D-MPNN with conservative defaults."""
    from chemprop import models, nn
    from chemprop.nn.metrics import MAE, RMSE

    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    ffn = nn.RegressionFFN()
    return models.MPNN(mp, agg, ffn, batch_norm=True, metrics=[RMSE(), MAE()])


def _build_datapoints(smiles: list[str], y: np.ndarray | None = None):
    """Map (smiles, y) -> Chemprop MoleculeDatapoints."""
    from chemprop import data

    if y is None:
        return [data.MoleculeDatapoint.from_smi(s, [float("nan")]) for s in smiles]
    return [data.MoleculeDatapoint.from_smi(s, [float(yi)]) for s, yi in zip(smiles, y)]


def _np_to_tensor(arr, dtype=None):
    """Convert a numpy array to a torch tensor via ``bytes`` (memoryview copy).

    This completely avoids the ``torch.from_numpy`` / ``torch.tensor(ndarray)``
    code paths that trigger SIGSEGV on macOS ARM with NumPy >= 2.0 due to
    copy-on-write memory aliasing.  Going through ``torch.frombuffer`` on a
    ``bytes`` object guarantees a fully owned copy with no numpy interop.
    """
    import numpy as _np

    arr = _np.ascontiguousarray(arr)
    if dtype is None:
        torch_dtype = {
            _np.dtype("float32"): torch.float32,
            _np.dtype("float64"): torch.float64,
            _np.dtype("int64"): torch.int64,
            _np.dtype("int32"): torch.int32,
        }[arr.dtype]
    else:
        torch_dtype = dtype
        arr = arr.astype(
            {torch.float32: _np.float32, torch.float64: _np.float64,
             torch.int64: _np.int64, torch.int32: _np.int32}[torch_dtype]
        )
    raw = arr.tobytes()
    t = torch.frombuffer(bytearray(raw), dtype=torch_dtype)
    return t.reshape(arr.shape)


def _safe_collate_batch(batch):
    """Collate with full memory isolation to avoid SIGSEGV on macOS ARM.

    Chemprop's ``BatchMolGraph.__post_init__`` uses ``torch.from_numpy`` which
    creates zero-copy tensor views of numpy arrays.  With NumPy >= 2.0
    copy-on-write semantics, those shared buffers cause use-after-free
    (SIGSEGV) when the numpy side reallocates.

    We work around this by:
    1. Deep-copying every per-molecule numpy array (``copy=True``).
    2. Converting numpy -> torch via ``tobytes()`` + ``torch.frombuffer`` to
       completely bypass the numpy-torch C-level interop that triggers the bug.
    """
    import numpy as _np

    from chemprop.data.collate import BatchMolGraph, TrainingBatch

    mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch)

    # ---- Build batched molecular graph ----
    Vs, Es, edge_indexes, rev_edge_indexes, batch_indexes = [], [], [], [], []
    num_nodes = 0
    num_edges = 0

    for i, mg in enumerate(mgs):
        Vs.append(_np.array(mg.V, dtype=_np.float32, copy=True, order="C"))
        Es.append(_np.array(mg.E, dtype=_np.float32, copy=True, order="C"))
        edge_indexes.append(
            _np.array(mg.edge_index, dtype=_np.int64, copy=True, order="C") + num_nodes
        )
        rev_edge_indexes.append(
            _np.array(mg.rev_edge_index, dtype=_np.int64, copy=True, order="C") + num_edges
        )
        batch_indexes.extend([i] * mg.V.shape[0])
        num_nodes += mg.V.shape[0]
        num_edges += mg.edge_index.shape[1]

    V_np = _np.concatenate(Vs)
    E_np = _np.concatenate(Es)
    ei_np = _np.hstack(edge_indexes)
    rei_np = _np.concatenate(rev_edge_indexes)
    bi_np = _np.array(batch_indexes, dtype=_np.int64)

    del Vs, Es, edge_indexes, rev_edge_indexes, batch_indexes

    bmg = object.__new__(BatchMolGraph)
    object.__setattr__(bmg, "_BatchMolGraph__size", len(mgs))
    object.__setattr__(bmg, "V", _np_to_tensor(V_np))
    object.__setattr__(bmg, "E", _np_to_tensor(E_np))
    object.__setattr__(bmg, "edge_index", _np_to_tensor(ei_np))
    object.__setattr__(bmg, "rev_edge_index", _np_to_tensor(rei_np))
    object.__setattr__(bmg, "batch", _np_to_tensor(bi_np))

    del V_np, E_np, ei_np, rei_np, bi_np

    # ---- remaining fields ----
    return TrainingBatch(
        bmg,
        None if V_ds[0] is None else _np_to_tensor(
            _np.concatenate(V_ds), dtype=torch.float
        ),
        None if x_ds[0] is None else _np_to_tensor(
            _np.array(x_ds), dtype=torch.float
        ),
        None if ys[0] is None else _np_to_tensor(
            _np.array(ys), dtype=torch.float
        ),
        _np_to_tensor(_np.array(weights, dtype=_np.float32)).unsqueeze(1),
        None if lt_masks[0] is None else _np_to_tensor(_np.array(lt_masks)),
        None if gt_masks[0] is None else _np_to_tensor(_np.array(gt_masks)),
    )


def _build_loader(smiles: list[str], y: np.ndarray | None, batch_size: int, shuffle: bool):
    from torch.utils.data import DataLoader

    from chemprop import data, featurizers

    datapoints = _build_datapoints(smiles, y)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dataset = data.MoleculeDataset(datapoints, featurizer)

    drop_last = len(dataset) % batch_size == 1  # matches Chemprop's auto logic
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=DEFAULT_NUM_WORKERS,
        collate_fn=_safe_collate_batch,
        drop_last=drop_last,
    )


# ---------------------------------------------------------------------------
# Manual training loop – avoids Lightning Trainer SIGSEGV on macOS ARM
# ---------------------------------------------------------------------------

def _train_member_manual(
    model,
    train_loader,
    val_loader,
    max_epochs: int,
    gradient_clip_val: float = 1.0,
    lr: float = DEFAULT_LR,
) -> None:
    """Train a single MPNN member with a plain PyTorch loop.

    Lightning's Trainer.fit() triggers an extra DataLoader iteration in
    ``configure_optimizers`` -> ``estimated_stepping_batches`` -> ``setup_data``,
    which doubles numpy->torch conversions and causes SIGSEGV on macOS ARM
    with NumPy >= 2.0. This manual loop avoids that entirely.
    """
    from tqdm import tqdm

    device = torch.device("cpu")
    model = model.to(device)
    model.train()

    # Chemprop MPNN uses Adam by default
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_train_batches = len(train_loader)
    total_steps = n_train_batches * max_epochs

    # Warmup + cosine schedule matching Chemprop defaults
    warmup_epochs = min(2, max_epochs)
    warmup_steps = warmup_epochs * n_train_batches

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{max_epochs}",
            leave=False,
            disable=False,
        )
        for batch in pbar:
            # Forward pass through the MPNN
            preds = model(batch.bmg, batch.V_d, batch.X_d)

            # MSE loss (Chemprop default for regression)
            targets = batch.Y.to(device)
            mask = ~torch.isnan(targets)
            if mask.any():
                loss = torch.nn.functional.mse_loss(preds[mask], targets[mask])
            else:
                loss = torch.tensor(0.0, device=device)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                preds = model(batch.bmg, batch.V_d, batch.X_d)
                targets = batch.Y.to(device)
                mask = ~torch.isnan(targets)
                if mask.any():
                    val_loss += torch.nn.functional.mse_loss(preds[mask], targets[mask]).item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        val_rmse = avg_val_loss ** 0.5

        LOG.info(
            "Epoch %d/%d  train_loss=%.4f  val_rmse=%.4f  lr=%.2e",
            epoch + 1,
            max_epochs,
            avg_loss,
            val_rmse,
            optimizer.param_groups[0]["lr"],
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
        accelerator: str | None = None,  # unused; kept for API compat
    ) -> None:
        """Train k ensemble members. Checkpoints saved to self.checkpoint_dir."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.k = k

        for i in range(k):
            seed = base_seed + i
            set_seed(seed)
            LOG.info("Training Chemprop member %d/%d (seed=%d)", i + 1, k, seed)

            model = _build_model()
            train_loader = _build_loader(train_smiles, train_y, batch_size, shuffle=True)
            val_loader = _build_loader(val_smiles, val_y, batch_size, shuffle=False)

            _train_member_manual(
                model,
                train_loader,
                val_loader,
                max_epochs=max_epochs,
                gradient_clip_val=1.0,
            )

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
        """Run inference for a single ensemble member."""
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
