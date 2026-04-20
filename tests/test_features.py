"""Tests for featurisation. These should pass without training data."""

from __future__ import annotations

import numpy as np

from logd.features import (
    DESCRIPTOR_NAMES,
    FeatureSpec,
    canonicalise,
    featurise_batch,
    featurise_one,
    morgan_fp,
    mol_from_smiles,
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
    assert feat.shape == (len(DESCRIPTOR_NAMES) + 2048,)
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


def test_feature_spec_dim_matches_output() -> None:
    spec_full = FeatureSpec()
    spec_desc_only = FeatureSpec(use_morgan=False)
    spec_morgan_only = FeatureSpec(use_descriptors=False)
    assert featurise_one("CCO", spec_full).shape == (spec_full.dim,)  # type: ignore[union-attr]
    assert featurise_one("CCO", spec_desc_only).shape == (spec_desc_only.dim,)  # type: ignore[union-attr]
    assert featurise_one("CCO", spec_morgan_only).shape == (spec_morgan_only.dim,)  # type: ignore[union-attr]
