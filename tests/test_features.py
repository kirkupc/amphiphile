"""Tests for featurisation. These should pass without training data."""

from __future__ import annotations

import numpy as np

from logd.features import (
    _IONISABLE_PATTERNS,
    DESCRIPTOR_NAMES,
    PKA_FEATURE_NAMES,
    FeatureSpec,
    canonicalise,
    featurise_batch,
    featurise_one,
    ionisable_counts,
    mol_from_smiles,
    morgan_fp,
    pka_corrections,
)


def test_canonicalise_valid() -> None:
    assert canonicalise("CCO") == canonicalise("OCC")
    assert canonicalise("c1ccccc1") is not None


def test_canonicalise_invalid() -> None:
    assert canonicalise("") is None
    assert canonicalise("not_a_smiles") is None
    assert canonicalise(None) is None  # type: ignore[arg-type]
    assert canonicalise(123) is None  # type: ignore[arg-type]


def test_canonicalise_strips_salts() -> None:
    # Sodium acetate → acetate
    cano = canonicalise("CC(=O)[O-].[Na+]")
    assert cano is not None
    assert "Na" not in cano


def test_morgan_fp_shape_and_binary() -> None:
    fp = morgan_fp(mol_from_smiles("CCO"))  # type: ignore[arg-type]
    assert fp.shape == (2048,)
    assert set(np.unique(fp).tolist()).issubset({0, 1})


def test_featurise_one_dim() -> None:
    spec = FeatureSpec()
    feat = featurise_one("CCO", spec)
    assert feat is not None
    expected = len(DESCRIPTOR_NAMES) + len(_IONISABLE_PATTERNS) + len(PKA_FEATURE_NAMES) + 2048
    assert feat.shape == (expected,)
    assert feat.dtype == np.float32


def test_featurise_one_invalid_returns_none() -> None:
    assert featurise_one("not_a_smiles") is None


def test_featurise_batch_mask_order() -> None:
    spec = FeatureSpec()
    X, mask = featurise_batch(["CCO", "not_a_smiles", "c1ccccc1"], spec)
    assert mask.tolist() == [True, False, True]
    assert X.shape == (2, spec.dim)


def test_featurise_batch_all_invalid() -> None:
    spec = FeatureSpec()
    X, mask = featurise_batch(["foo", "bar"], spec)
    assert mask.sum() == 0
    assert X.shape == (0, spec.dim)


def test_canonical_equivalence_produces_same_features() -> None:
    a = featurise_one("CCO")
    b = featurise_one("OCC")
    assert a is not None and b is not None
    np.testing.assert_array_equal(a, b)


def test_ionisable_counts_basic_amine() -> None:
    mol = mol_from_smiles("CCN")
    assert mol is not None
    counts = ionisable_counts(mol)
    assert counts.shape == (len(_IONISABLE_PATTERNS),)
    assert counts[0] > 0  # n_basic_amine


def test_ionisable_counts_carboxylic_acid() -> None:
    mol = mol_from_smiles("CC(=O)O")
    assert mol is not None
    counts = ionisable_counts(mol)
    assert counts[4] > 0  # n_acidic_oh


def test_pka_corrections_basic_amine() -> None:
    mol = mol_from_smiles("CCN")  # ethylamine: 1 basic amine, pKa ~9.5
    assert mol is not None
    pka = pka_corrections(mol)
    assert pka.shape == (2,)
    # Henderson-Hasselbalch: shift = -log10(1 + 10^(9.5-7.4)) ≈ -2.12
    assert -2.2 < pka[0] < -2.0, f"expected ~-2.1, got {pka[0]}"
    # Fraction protonated: 10^2.1 / (1 + 10^2.1) ≈ 0.992
    assert 0.98 < pka[1] < 1.0, f"expected ~0.99, got {pka[1]}"


def test_pka_corrections_carboxylic_acid() -> None:
    mol = mol_from_smiles("CC(=O)O")  # acetic acid, pKa ~4.5
    assert mol is not None
    pka = pka_corrections(mol)
    # shift = -log10(1 + 10^(7.4-4.5)) ≈ -2.90
    assert -3.0 < pka[0] < -2.8, f"expected ~-2.9, got {pka[0]}"
    # Fraction deprotonated: 10^2.9 / (1 + 10^2.9) ≈ 0.999 → charge ≈ -0.999
    assert -1.0 < pka[1] < -0.99, f"expected ~-0.999, got {pka[1]}"


def test_pka_corrections_neutral() -> None:
    mol = mol_from_smiles("CCCCCC")  # hexane: no ionisable groups
    assert mol is not None
    pka = pka_corrections(mol)
    assert pka[0] == 0.0  # no shift
    assert pka[1] == 0.0  # no charge


def test_feature_spec_dim_matches_output() -> None:
    spec_full = FeatureSpec()
    spec_desc_only = FeatureSpec(use_morgan=False)
    spec_morgan_only = FeatureSpec(use_descriptors=False)
    spec_no_ion = FeatureSpec(use_ionisable=False)
    spec_no_pka = FeatureSpec(use_pka=False)
    assert featurise_one("CCO", spec_full).shape == (spec_full.dim,)  # type: ignore[union-attr]
    assert featurise_one("CCO", spec_desc_only).shape == (spec_desc_only.dim,)  # type: ignore[union-attr]
    assert featurise_one("CCO", spec_morgan_only).shape == (spec_morgan_only.dim,)  # type: ignore[union-attr]
    assert featurise_one("CCO", spec_no_ion).shape == (spec_no_ion.dim,)  # type: ignore[union-attr]
    assert featurise_one("CCO", spec_no_pka).shape == (spec_no_pka.dim,)  # type: ignore[union-attr]
