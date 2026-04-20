"""Featurisation: RDKit 2D descriptors + Morgan fingerprints.

Both featurisers return numpy arrays suitable for tree / linear / MLP models.
Invalid SMILES return None; callers are responsible for masking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.SaltRemover import SaltRemover

# Silence RDKit's per-molecule complaints about bad SMILES; we handle them.
RDLogger.DisableLog("rdApp.*")

MORGAN_RADIUS = 2
MORGAN_BITS = 2048

# Descriptor list is a stable subset of RDKit's full descriptor set,
# chosen for relevance to logD and to stay numerically well-behaved.
# Listed explicitly (not Descriptors._descList) so features are stable
# across RDKit versions.
DESCRIPTOR_NAMES: tuple[str, ...] = (
    "MolWt",
    "HeavyAtomMolWt",
    "ExactMolWt",
    "MolLogP",
    "MolMR",
    "TPSA",
    "LabuteASA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumHeteroatoms",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "NOCount",
    "NHOHCount",
    "NumValenceElectrons",
    "BalabanJ",
    "BertzCT",
    "Chi0",
    "Chi1",
    "Chi2n",
    "Chi3n",
    "HallKierAlpha",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "qed",
)

_DESCRIPTOR_FNS = [(name, getattr(Descriptors, name)) for name in DESCRIPTOR_NAMES]

_SALT_REMOVER = SaltRemover()


@dataclass(frozen=True)
class FeatureSpec:
    """Describes a feature block; serialized with the model so inference matches training."""

    use_descriptors: bool = True
    use_morgan: bool = True
    morgan_radius: int = MORGAN_RADIUS
    morgan_bits: int = MORGAN_BITS

    @property
    def dim(self) -> int:
        d = 0
        if self.use_descriptors:
            d += len(DESCRIPTOR_NAMES)
        if self.use_morgan:
            d += self.morgan_bits
        return d


def canonicalise(smiles: str) -> str | None:
    """Parse SMILES, strip salts, return canonical SMILES string. None if invalid."""
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = _SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    """Parse a SMILES into a salt-stripped RDKit Mol. None if invalid."""
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = _SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return mol


def descriptors(mol: Chem.Mol) -> np.ndarray:
    """Compute the fixed descriptor block. NaN-cleaned."""
    out = np.empty(len(_DESCRIPTOR_FNS), dtype=np.float32)
    for i, (_, fn) in enumerate(_DESCRIPTOR_FNS):
        try:
            v = fn(mol)
        except Exception:
            v = 0.0
        if v is None or not np.isfinite(v):
            v = 0.0
        out[i] = float(v)
    return out


def morgan_fp(mol: Chem.Mol, radius: int = MORGAN_RADIUS, bits: int = MORGAN_BITS) -> np.ndarray:
    """Morgan (ECFP-like) fingerprint as a dense uint8 array."""
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=bits)
    fp = gen.GetFingerprintAsNumPy(mol)
    return fp.astype(np.uint8)


def featurise_one(smiles: str, spec: FeatureSpec = FeatureSpec()) -> np.ndarray | None:
    """Featurise a single SMILES. None if invalid."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    parts: list[np.ndarray] = []
    if spec.use_descriptors:
        parts.append(descriptors(mol))
    if spec.use_morgan:
        parts.append(morgan_fp(mol, radius=spec.morgan_radius, bits=spec.morgan_bits))
    return np.concatenate(parts).astype(np.float32)


def featurise_batch(
    smiles_iter: Iterable[str], spec: FeatureSpec = FeatureSpec()
) -> tuple[np.ndarray, np.ndarray]:
    """
    Featurise a batch. Returns (features [n_valid, dim], valid_mask [n_input]).

    Invalid SMILES are dropped from the feature matrix; use the mask to align
    back to input order.
    """
    smiles_list = list(smiles_iter)
    rows: list[np.ndarray] = []
    mask = np.zeros(len(smiles_list), dtype=bool)
    for i, smi in enumerate(smiles_list):
        feat = featurise_one(smi, spec)
        if feat is None:
            continue
        rows.append(feat)
        mask[i] = True
    if not rows:
        return np.zeros((0, spec.dim), dtype=np.float32), mask
    return np.stack(rows, axis=0), mask
