"""Tests for scaffold splitting — the non-obvious part is leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from logd.data.splits import random_split, scaffold_split


def _scaffold(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    sc = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(sc, canonical=True) if sc else ""


SMILES = [
    "c1ccccc1",
    "c1ccc(C)cc1",
    "c1ccc(CC)cc1",
    "c1ccc(CCC)cc1",
    "CCO",
    "CCCO",
    "CCCCO",
    "CCCCCO",
    "c1ccncc1",
    "c1ccncc1C",
    "c1ccncc1CC",
    "c1ccncc1CCC",
    "C1CCCCC1",
    "C1CCCCC1C",
    "C1CCCCC1CC",
    "C1CCCCC1CCC",
    "C1=CC=CN=C1",
    "C1=CC=CN=C1C",
    "C1=CC=CN=C1CC",
    "C1=CC=CN=C1CCC",
]


def test_scaffold_split_no_overlap() -> None:
    s = pd.Series(SMILES)
    split = scaffold_split(s, seed=1)
    all_ids = np.concatenate([split.train, split.val, split.test])
    assert len(all_ids) == len(SMILES)
    assert len(set(all_ids.tolist())) == len(SMILES)


def test_scaffold_split_no_scaffold_leakage() -> None:
    s = pd.Series(SMILES)
    split = scaffold_split(s, seed=1)
    train_scaffolds = {_scaffold(SMILES[i]) for i in split.train}
    test_scaffolds = {_scaffold(SMILES[i]) for i in split.test}
    # Minus the empty scaffold (alkanes have none)
    overlap = (train_scaffolds & test_scaffolds) - {""}
    assert overlap == set(), f"Scaffold leakage between train and test: {overlap}"


def test_random_split_proportions() -> None:
    split = random_split(100, 0.8, 0.1, 0.1, seed=0)
    assert len(split.train) == 80
    assert len(split.val) == 10
    assert len(split.test) == 10
    all_ids = np.concatenate([split.train, split.val, split.test])
    assert set(all_ids.tolist()) == set(range(100))


def test_random_split_bad_fractions_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        random_split(10, 0.5, 0.5, 0.5)
