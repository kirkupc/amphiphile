"""Bemis-Murcko scaffold splitting.

Why scaffold splits: drug-discovery datasets cluster by chemical series. A random
split lets the model memorise series membership and over-reports performance. A
scaffold split asks "how does this do on chemical matter it hasn't seen?" which
is the only question a deployed model should be held to.

Implementation: compute the Bemis-Murcko scaffold SMILES for each molecule, bucket
by scaffold, then greedy-pack buckets into train/val/test in size order. The
largest scaffolds go to train; the smallest to val/test. This minimises the
"freak huge scaffold lands in test" failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def __post_init__(self) -> None:
        sets = [set(self.train.tolist()), set(self.val.tolist()), set(self.test.tolist())]
        if sum(len(s) for s in sets) != len(set().union(*sets)):
            raise ValueError("Split indices are not disjoint.")


def _scaffold_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True) if scaffold is not None else ""
    except (ValueError, RuntimeError):
        return ""


def scaffold_split(
    smiles: pd.Series,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 0,
) -> SplitIndices:
    """Bucket-greedy scaffold split.

    Singleton scaffolds (single-compound buckets) are shuffled and assigned to
    val/test preferentially; this behaves like a conventional scaffold split used
    in Chemprop / MoleculeNet.
    """
    if not np.isclose(frac_train + frac_val + frac_test, 1.0):
        raise ValueError("Split fractions must sum to 1.0")

    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"smiles": smiles.reset_index(drop=True)})
    df["scaffold"] = df["smiles"].map(_scaffold_smiles)

    buckets: dict[str, list[int]] = {}
    for i, scaf in enumerate(df["scaffold"].tolist()):
        buckets.setdefault(scaf, []).append(i)

    singletons = [v for v in buckets.values() if len(v) == 1]
    multis = sorted((v for v in buckets.values() if len(v) > 1), key=len, reverse=True)

    rng.shuffle(singletons)

    n = len(df)
    n_train_target = int(frac_train * n)
    n_val_target = int(frac_val * n)

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []

    # Big buckets first → train. Greedy fill.
    for bucket in multis:
        if len(train) + len(bucket) <= n_train_target:
            train.extend(bucket)
        elif len(val) + len(bucket) <= n_val_target:
            val.extend(bucket)
        else:
            test.extend(bucket)

    # Singletons split proportionally between val/test, topping up train if we undershot.
    for bucket in singletons:
        if len(train) < n_train_target:
            train.extend(bucket)
        elif len(val) < n_val_target:
            val.extend(bucket)
        else:
            test.extend(bucket)

    return SplitIndices(
        train=np.array(sorted(train), dtype=np.int64),
        val=np.array(sorted(val), dtype=np.int64),
        test=np.array(sorted(test), dtype=np.int64),
    )


def random_split(
    n: int,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 0,
) -> SplitIndices:
    if not np.isclose(frac_train + frac_val + frac_test, 1.0):
        raise ValueError("Split fractions must sum to 1.0")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(frac_train * n)
    n_val = int(frac_val * n)
    return SplitIndices(
        train=np.sort(idx[:n_train]),
        val=np.sort(idx[n_train : n_train + n_val]),
        test=np.sort(idx[n_train + n_val :]),
    )
