"""Featurisation: RDKit 2D descriptors + Morgan fingerprints.

Both featurisers return numpy arrays suitable for tree / linear / MLP models.
Invalid SMILES return None; callers are responsible for masking.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.SaltRemover import SaltRemover

# Silence RDKit's per-molecule complaints about bad SMILES; we handle them.
RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

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

# SMARTS for ionisable groups relevant to pH 7.4 logD prediction.
# These count features let the model learn charge-state effects that
# the neutral-parent descriptors and Morgan fingerprints miss.
_IONISABLE_SMARTS: list[tuple[str, str]] = [
    ("n_basic_amine", "[NX3;H2,H1,H0;!$(NC=O);!$(N=*);!$([nR])]"),
    ("n_amidine", "[NX3][CX3]=[NX2]"),
    ("n_guanidine", "[NX3][CX3](=[NX2])[NX3]"),
    ("n_quat_n", "[N+;!$([N+]-[O-])]"),
    ("n_acidic_oh", "[OX2H][CX3]=O"),
    ("n_pyridine_n", "[nR1;H0]"),
]
_IONISABLE_PATTERNS = [(name, Chem.MolFromSmarts(sma)) for name, sma in _IONISABLE_SMARTS]

# Henderson-Hasselbalch logD correction per ionisable group.
# logD_shift_i = -log10(1 + 10^(pKa - pH)) for bases, -log10(1 + 10^(pH - pKa)) for acids.
# Using group-average literature pKa values at pH 7.4.
_TARGET_PH = 7.4
_GROUP_PKA: dict[str, tuple[float, str]] = {
    "n_basic_amine": (9.5, "base"),
    "n_amidine": (10.5, "base"),
    "n_guanidine": (12.5, "base"),
    "n_quat_n": (14.0, "base"),  # permanently charged; large pKa → shift ≈ -6.6
    "n_acidic_oh": (4.5, "acid"),
    "n_pyridine_n": (4.0, "base"),
}
_GROUP_SHIFT: dict[str, float] = {}
for _name, (_pka, _kind) in _GROUP_PKA.items():
    if _kind == "base":
        _GROUP_SHIFT[_name] = -np.log10(1.0 + 10.0 ** (_pka - _TARGET_PH))
    else:
        _GROUP_SHIFT[_name] = -np.log10(1.0 + 10.0 ** (_TARGET_PH - _pka))
PKA_FEATURE_NAMES: tuple[str, ...] = ("estimated_logd_shift", "net_charge_pH7_4")


@dataclass(frozen=True)
class FeatureSpec:
    """Describes a feature block; serialized with the model so inference matches training."""

    use_descriptors: bool = True
    use_morgan: bool = True
    use_ionisable: bool = True
    use_pka: bool = True
    morgan_radius: int = MORGAN_RADIUS
    morgan_bits: int = MORGAN_BITS

    @property
    def dim(self) -> int:
        d = 0
        if self.use_descriptors:
            d += len(DESCRIPTOR_NAMES)
        if self.use_ionisable:
            d += len(_IONISABLE_PATTERNS)
        if self.use_pka:
            d += len(PKA_FEATURE_NAMES)
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
    except (ValueError, RuntimeError):
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
        except (ValueError, RuntimeError, ZeroDivisionError):
            v = 0.0
        if v is None or not np.isfinite(v):
            v = 0.0
        out[i] = float(v)
    return out


def ionisable_counts(mol: Chem.Mol) -> np.ndarray:
    """Count ionisable functional groups relevant to pH 7.4 partitioning."""
    out = np.zeros(len(_IONISABLE_PATTERNS), dtype=np.float32)
    for i, (_, pat) in enumerate(_IONISABLE_PATTERNS):
        if pat is not None:
            out[i] = float(len(mol.GetSubstructMatches(pat)))
    return out


def pka_corrections(mol: Chem.Mol) -> np.ndarray:
    """Henderson-Hasselbalch logD correction features from ionisable-group counts.

    Returns [estimated_logd_shift, net_charge_pH7_4].
    """
    logd_shift = 0.0
    net_charge = 0.0
    for name, pat in _IONISABLE_PATTERNS:
        if pat is None:
            continue
        n = len(mol.GetSubstructMatches(pat))
        if n == 0:
            continue
        _pka, kind = _GROUP_PKA[name]
        logd_shift += n * _GROUP_SHIFT[name]
        if kind == "base":
            frac = 10.0 ** (_pka - _TARGET_PH) / (1.0 + 10.0 ** (_pka - _TARGET_PH))
            net_charge += n * frac
        else:
            frac = 10.0 ** (_TARGET_PH - _pka) / (1.0 + 10.0 ** (_TARGET_PH - _pka))
            net_charge -= n * frac
    return np.array([logd_shift, net_charge], dtype=np.float32)


def morgan_fp(mol: Chem.Mol, radius: int = MORGAN_RADIUS, bits: int = MORGAN_BITS) -> np.ndarray:
    """Morgan (ECFP-like) fingerprint as a dense uint8 array."""
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=bits)  # type: ignore[attr-defined]
    fp = gen.GetFingerprintAsNumPy(mol)
    return fp.astype(np.uint8)


def featurise_one(smiles: str, spec: FeatureSpec | None = None) -> np.ndarray | None:
    """Featurise a single SMILES. None if invalid."""
    if spec is None:
        spec = FeatureSpec()
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    parts: list[np.ndarray] = []
    if spec.use_descriptors:
        parts.append(descriptors(mol))
    if spec.use_ionisable:
        parts.append(ionisable_counts(mol))
    if spec.use_pka:
        parts.append(pka_corrections(mol))
    if spec.use_morgan:
        parts.append(morgan_fp(mol, radius=spec.morgan_radius, bits=spec.morgan_bits))
    return np.concatenate(parts).astype(np.float32)


def featurise_batch(
    smiles_iter: Iterable[str], spec: FeatureSpec | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Featurise a batch. Returns (features [n_valid, dim], valid_mask [n_input]).

    Invalid SMILES are dropped from the feature matrix; use the mask to align
    back to input order.
    """
    if spec is None:
        spec = FeatureSpec()
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
