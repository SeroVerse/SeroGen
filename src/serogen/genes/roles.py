"""Canonical biosynthetic gene roles and metadata specifications."""

from dataclasses import dataclass
from enum import IntEnum, StrEnum

import numpy as np


class RoleCategory(StrEnum):
    """Functional category of a biosynthetic gene role."""

    TRANSPORT = "Transport"
    POLYMER_ASSEMBLY = "PolymerAssembly"
    MODIFIER = "Modifier"
    PRECURSOR_SYNTHESIS = "PrecursorSynthesis"
    MOBILE_ELEMENT = "MobileElement"
    BACKGROUND = "Background"


@dataclass(frozen=True, slots=True)
class RoleSpec:
    """Immutable metadata specification for a canonical gene role."""

    name: str
    abbreviation: str
    category: RoleCategory
    description: str
    aliases: tuple[str, ...] = ()


class GeneRole(IntEnum):
    """Canonical biosynthetic gene roles (0..29) for surface polysaccharide loci.

    Subclasses `IntEnum` for zero-overhead integer semantics in bitfields (`IntNode`),
    32-bit presence masks (`1 << role`), Numba kernels, and SciPy CSR matrices.
    """

    FLIPPASE = 0
    POLYMERASE = 1
    COPOLYMERASE = 2
    KINASE = 3
    PHOSPHATASE = 4
    ABC_PERMEASE = 5
    ABC_ATPASE = 6
    LIGASE = 7
    TRANSLOCON = 8
    CAPSULE_ANCHOR = 9
    PYRUVYLTRANSFERASE = 10
    METHYLTRANSFERASE = 11
    ACETYLTRANSFERASE = 12
    PHOSPHOTRANSFERASE = 13
    FORMYLTRANSFERASE = 14
    ALANINE_TRANSFERASE = 15
    AMINOTRANSFERASE = 16
    ALDOLASE = 17
    DEACETYLASE = 18
    EPIMERASE = 19
    OXIDOREDUCTASE = 20
    TRANSMEMBRANE_SYNTHASE = 21
    NUCLEOTIDYLTRANSFERASE = 22
    NUCLEOTIDASE = 23
    INITIATOR = 24
    GLYCOSYLHYDROLASE = 25
    GLYCOSYLTRANSFERASE = 26
    TRANSPOSASE = 27
    INTEGRASE = 28
    UNKNOWN = 29

    # ------------------------------------------------------------------------------------------------------------------
    # Instance Metadata Properties
    # ------------------------------------------------------------------------------------------------------------------
    @property
    def spec(self) -> RoleSpec:
        """Return the immutable `RoleSpec` metadata for this role."""
        return _ROLE_SPECS[self]

    @property
    def abbreviation(self) -> str:
        """Short display abbreviation (e.g. `'Wzx'`, `'GT'`)."""
        return _ROLE_SPECS[self].abbreviation

    @property
    def category(self) -> RoleCategory:
        """Functional `RoleCategory` of this gene role."""
        return _ROLE_SPECS[self].category

    @property
    def description(self) -> str:
        """Human-readable biochemical description."""
        return _ROLE_SPECS[self].description

    @property
    def aliases(self) -> tuple[str, ...]:
        """Known gene symbol aliases for this role."""
        return _ROLE_SPECS[self].aliases

    @property
    def bitmask(self) -> int:
        """32-bit integer bitmask (`1 << int(self)`) for archetype presence masks."""
        return 1 << int(self)

    @property
    def is_mobile_element(self) -> bool:
        """True if this role is a mobile genetic element (`TRANSPOSASE`, `INTEGRASE`)."""
        return _ROLE_SPECS[self].category == RoleCategory.MOBILE_ELEMENT

    @property
    def is_biosynthetic(self) -> bool:
        """True if this role represents active locus biosynthesis, modification, or transport."""
        return _ROLE_SPECS[self].category not in (RoleCategory.MOBILE_ELEMENT, RoleCategory.BACKGROUND)

    # ------------------------------------------------------------------------------------------------------------------
    # Lookups & Vectorized Class Views
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def from_name(cls, key: str | bytes | int | GeneRole) -> GeneRole:
        """Resolve a role from an integer index, canonical name, abbreviation, or alias.

        Args:
            key: Role identifier (`GeneRole`, `int`, `str`, or `bytes`).

        Returns:
            GeneRole: Resolved canonical role enum member.

        Raises:
            KeyError: If a string/bytes key does not match any known role name, abbreviation, or alias.
            ValueError: If an integer index is out of range.
        """
        if isinstance(key, cls):
            return key
        if isinstance(key, (int, np.integer)):
            return cls(int(key))
        s = key.decode("utf-8", errors="replace") if isinstance(key, (bytes, bytearray)) else str(key)
        s_clean = s.strip()
        if s_clean in _LOOKUP_MAP:
            return _LOOKUP_MAP[s_clean]
        s_lower = s_clean.lower()
        if s_lower in _LOOKUP_MAP:
            return _LOOKUP_MAP[s_lower]
        raise KeyError(f"Unknown GeneRole identifier: {key!r}")

    @classmethod
    def resolve(cls, key: str | bytes | int | GeneRole, default: GeneRole | None = None) -> GeneRole:
        """Resolve a role identifier, falling back to `default` (or `GeneRole.UNKNOWN`) if not found."""
        fallback = cls.UNKNOWN if default is None else default
        try:
            return cls.from_name(key)
        except KeyError, ValueError:
            return fallback

    @classmethod
    def names_array(cls) -> np.ndarray[tuple[int], np.dtype[np.str_]]:
        """Contiguous 1D NumPy string array of canonical role names."""
        return _NAMES_ARRAY

    @classmethod
    def abbreviations_array(cls) -> np.ndarray[tuple[int], np.dtype[np.str_]]:
        """Contiguous 1D NumPy string array of role abbreviations."""
        return _ABBREVIATIONS_ARRAY

    @classmethod
    def categories_array(cls) -> np.ndarray[tuple[int], np.dtype[np.str_]]:
        """Contiguous 1D NumPy string array of role categories."""
        return _CATEGORIES_ARRAY

    @classmethod
    def feature_matrix(cls) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        """Return precomputed `(30, D)` float32 one-hot category feature matrix for embeddings."""
        return _FEATURE_MATRIX


# ----------------------------------------------------------------------------------------------------------------------
# Canonical Gene Role Specifications (Ordered 0..29 matching GeneRole)
# ----------------------------------------------------------------------------------------------------------------------

_ROLE_SPECS: tuple[RoleSpec, ...] = (
    # 0: FLIPPASE
    RoleSpec(
        name="FLIPPASE",
        abbreviation="Wzx",
        category=RoleCategory.TRANSPORT,
        description="Wzx flippase; translocates completed lipid-linked repeat units across inner membrane.",
        aliases=("wzx", "gmla"),
    ),
    # 1: POLYMERASE
    RoleSpec(
        name="POLYMERASE",
        abbreviation="Wzy",
        category=RoleCategory.POLYMER_ASSEMBLY,
        description="Wzy polymerase; forms the polymer repeat unit boundary glycosidic linkage.",
        aliases=("wzy",),
    ),
    # 2: COPOLYMERASE
    RoleSpec(
        name="COPOLYMERASE",
        abbreviation="Wzz",
        category=RoleCategory.TRANSPORT,
        description="Wzz polysaccharide co-polymerase; modal chain length regulator.",
        aliases=("wzz", "cld"),
    ),
    # 3: KINASE
    RoleSpec(
        name="KINASE",
        abbreviation="Kin",
        category=RoleCategory.TRANSPORT,
        description="Wzc / BY-kinase; tyrosine autokinase coordinating polymer export.",
        aliases=("wzc", "ptk"),
    ),
    # 4: PHOSPHATASE
    RoleSpec(
        name="PHOSPHATASE",
        abbreviation="Wzb",
        category=RoleCategory.TRANSPORT,
        description="Wzb / PTP / CpsACP; low-molecular-weight phosphotyrosine and acid phosphatases.",
        aliases=("wzb", "ptp", "cpsacp", "cpsb"),
    ),
    # 5: ABC_PERMEASE
    RoleSpec(
        name="ABC_PERMEASE",
        abbreviation="Wzm",
        category=RoleCategory.TRANSPORT,
        description="Wzm / KpsM ABC transporter permease; transmembrane translocation channel.",
        aliases=("wzm", "kpsm"),
    ),
    # 6: ABC_ATPASE
    RoleSpec(
        name="ABC_ATPASE",
        abbreviation="Wzt",
        category=RoleCategory.TRANSPORT,
        description="Wzt / KpsT ABC transporter ATPase; nucleotide-binding domain powering export.",
        aliases=("wzt", "kpst"),
    ),
    # 7: LIGASE
    RoleSpec(
        name="LIGASE",
        abbreviation="WaaL",
        category=RoleCategory.POLYMER_ASSEMBLY,
        description="WaaL ligase; attaches periplasmic polysaccharide to Lipid A core.",
        aliases=("waal", "rfal"),
    ),
    # 8: TRANSLOCON
    RoleSpec(
        name="TRANSLOCON",
        abbreviation="Wza",
        category=RoleCategory.TRANSPORT,
        description="Wza / KpsD / KpsE outer membrane and periplasmic translocon complex for capsular polysaccharide surface export.",
        aliases=("wza", "gfcd", "gfcb", "gfcc", "kpsd", "kpse"),
    ),
    # 9: CAPSULE_ANCHOR
    RoleSpec(
        name="CAPSULE_ANCHOR",
        abbreviation="Wzi",
        category=RoleCategory.TRANSPORT,
        description="Wzi outer membrane lectin / KpsC lipid anchor / Wzg (CpsA) LytR-CpsA-Psr peptidoglycan capsule anchor transferase.",
        aliases=("wzi", "anchor", "capsule_anchor", "kpsc", "wzg", "cpsa", "cps2a", "capa"),
    ),
    # 10: PYRUVYLTRANSFERASE
    RoleSpec(
        name="PYRUVYLTRANSFERASE",
        abbreviation="PT",
        category=RoleCategory.MODIFIER,
        description="Pyruvyltransferase; adds pyruvate acetal (Pyr) modifications.",
        aliases=("wcak", "ptr", "wbmx"),
    ),
    # 11: METHYLTRANSFERASE
    RoleSpec(
        name="METHYLTRANSFERASE",
        abbreviation="MetT",
        category=RoleCategory.MODIFIER,
        description="Methyltransferase; adds O-methyl (OMe) groups.",
    ),
    # 12: ACETYLTRANSFERASE
    RoleSpec(
        name="ACETYLTRANSFERASE",
        abbreviation="AT",
        category=RoleCategory.MODIFIER,
        description="Acetyltransferase / Acyltransferase; adds O-acetyl (OAc) or hydroxyacyl (3HOBut, 3HOMyr) groups.",
        aliases=("oata", "wecd", "atr", "qdte", "wcio"),
    ),
    # 13: PHOSPHOTRANSFERASE
    RoleSpec(
        name="PHOSPHOTRANSFERASE",
        abbreviation="PhoT",
        category=RoleCategory.MODIFIER,
        description="Phosphotransferase; adds phosphate (P), phosphoethanolamine (PEtn), or phosphocholine (ChoP) groups.",
        aliases=("licd", "pgt"),
    ),
    # 14: FORMYLTRANSFERASE
    RoleSpec(
        name="FORMYLTRANSFERASE",
        abbreviation="ForT",
        category=RoleCategory.MODIFIER,
        description="Formyltransferase; adds formyl (Fo) modifications.",
        aliases=("qdtf", "fdtd", "vioa"),
    ),
    # 15: ALANINE_TRANSFERASE
    RoleSpec(
        name="ALANINE_TRANSFERASE",
        abbreviation="Alt",
        category=RoleCategory.MODIFIER,
        description="Alt1 alanine transferase; transfers D-alanine to the carboxyl group of GalA/GalNAcA forming an amide.",
        aliases=("alt", "alt1", "ala_transferase"),
    ),
    # 16: AMINOTRANSFERASE
    RoleSpec(
        name="AMINOTRANSFERASE",
        abbreviation="ATn",
        category=RoleCategory.PRECURSOR_SYNTHESIS,
        description="Aminotransferase / Transaminase (e.g. WecE, RhnB, RhnC); converts keto-intermediates to amino sugars.",
        aliases=("wece", "rhnb", "rhnc"),
    ),
    # 17: ALDOLASE
    RoleSpec(
        name="ALDOLASE",
        abbreviation="Ald",
        category=RoleCategory.PRECURSOR_SYNTHESIS,
        description="PEP-condensing aldolase/synthase (e.g. NeuB, PseI, LegI, KdsA); synthesizes nonulosonic and ulosonic acids.",
        aliases=("neub", "psei", "legi", "kdsa"),
    ),
    # 18: DEACETYLASE
    RoleSpec(
        name="DEACETYLASE",
        abbreviation="DAc",
        category=RoleCategory.MODIFIER,
        description="Deacetylase; generates free amino sugar derivatives.",
        aliases=("pgab", "pda1", "pda2", "pda2a", "ahy"),
    ),
    # 19: EPIMERASE
    RoleSpec(
        name="EPIMERASE",
        abbreviation="Epi",
        category=RoleCategory.PRECURSOR_SYNTHESIS,
        description="Epimerase / Isomerase / Mutase; interconverts UDP-sugar stereoisomers and hexose-phosphate intermediates.",
        aliases=("gale", "rmlc", "neuc", "gne", "rfbc", "kpsf", "manb", "pgm", "gpi", "glf"),
    ),
    # 20: OXIDOREDUCTASE
    RoleSpec(
        name="OXIDOREDUCTASE",
        abbreviation="OxR",
        category=RoleCategory.PRECURSOR_SYNTHESIS,
        description="Reductase / Dehydratase / Dehydrogenase (e.g. RmlB, RmlD, Ugd, Gnd, Gmd, Fcl); synthesizes deoxyhexoses, dideoxyhexoses, and uronic acids.",
        aliases=("rmlb", "rmld", "rhna", "epab", "epad", "rfbb", "rfbd", "ugd", "gnd", "gmd", "fcl"),
    ),
    # 21: TRANSMEMBRANE_SYNTHASE
    RoleSpec(
        name="TRANSMEMBRANE_SYNTHASE",
        abbreviation="TMS",
        category=RoleCategory.POLYMER_ASSEMBLY,
        description="Processive transmembrane glycosyltransferase (e.g. HasA, PgaC, BcsA); polymerizes and extrudes glycan chains.",
        aliases=("hasa", "pgac", "bcsa"),
    ),
    # 22: NUCLEOTIDYLTRANSFERASE
    RoleSpec(
        name="NUCLEOTIDYLTRANSFERASE",
        abbreviation="NT",
        category=RoleCategory.PRECURSOR_SYNTHESIS,
        description="Nucleotide-sugar precursor synthase (e.g. GalF, GalU, RmlA, ManC, NeuA, KdsB); acts on base monosaccharide pathways.",
        aliases=("rmla", "galu", "galf", "manc", "neua", "kdsb", "kpsu", "epaa", "epac"),
    ),
    # 23: NUCLEOTIDASE
    RoleSpec(
        name="NUCLEOTIDASE",
        abbreviation="Nuc",
        category=RoleCategory.PRECURSOR_SYNTHESIS,
        description="Sugar-nucleotide nucleotidase / hydrolase (e.g. PsaD, PsaG, PsaI, LgaF, Gmm, WcaH); cleaves sugar-nucleotide intermediates.",
        aliases=("psad", "psag", "psai", "lgaf", "gmm", "wcah", "sugar_phosphatase"),
    ),
    # 24: INITIATOR
    RoleSpec(
        name="INITIATOR",
        abbreviation="Init",
        category=RoleCategory.POLYMER_ASSEMBLY,
        description="Initial sugar transferase (e.g. WbaP, WcaJ, WchA, ItrA, WecA); attaches first non-modifier sugar onto undecaprenyl phosphate.",
        aliases=("wbap", "wcaj", "wcha", "itra", "itrb", "weca", "initiating_gt"),
    ),
    # 25: GLYCOSYLHYDROLASE
    RoleSpec(
        name="GLYCOSYLHYDROLASE",
        abbreviation="GH",
        category=RoleCategory.MODIFIER,
        description="Glycosylhydrolase / lyase; processes, depolymerizes, or cleaves glycosidic linkages.",
        aliases=("gnlb", "ghl", "orf1(ghy)", "ghy"),
    ),
    # 26: GLYCOSYLTRANSFERASE
    RoleSpec(
        name="GLYCOSYLTRANSFERASE",
        abbreviation="GT",
        category=RoleCategory.POLYMER_ASSEMBLY,
        description="Glycosyltransferase; synthesizes glycosidic linkages and non-modifier monosaccharide additions.",
        aliases=("kpss", "gtr", "glt", "htrl", "wbmv", "wbmw", "gmlc", "spsa", "wcaa", "rhamnan", "rgpf"),
    ),
    # 27: TRANSPOSASE
    RoleSpec(
        name="TRANSPOSASE",
        abbreviation="Tnp",
        category=RoleCategory.MOBILE_ELEMENT,
        description="Transposase / Insertion sequence (IS) element; mobile genetic element non-biosynthetic infiltrator.",
        aliases=("tnp", "transposase", "is_element", "insertion_sequence"),
    ),
    # 28: INTEGRASE
    RoleSpec(
        name="INTEGRASE",
        abbreviation="Int",
        category=RoleCategory.MOBILE_ELEMENT,
        description="Integrase / Recombinase / Excisionase; mediates DNA integration/excision, prophage insertions, or mobile cassette movement.",
        aliases=("int", "integrase", "recombinase", "excisionase"),
    ),
    # 29: UNKNOWN
    RoleSpec(
        name="UNKNOWN",
        abbreviation="?",
        category=RoleCategory.BACKGROUND,
        description="Uncharacterized gene; background or non-biosynthetic role.",
    ),
)

# ----------------------------------------------------------------------------------------------------------------------
# Precomputed Lookup Tables & Vectorized Arrays
# ----------------------------------------------------------------------------------------------------------------------

_NAMES_ARRAY: np.ndarray[tuple[int], np.dtype[np.str_]] = np.array([s.name for s in _ROLE_SPECS], dtype="U24")
_ABBREVIATIONS_ARRAY: np.ndarray[tuple[int], np.dtype[np.str_]] = np.array(
    [s.abbreviation for s in _ROLE_SPECS], dtype="U8"
)
_CATEGORIES_ARRAY: np.ndarray[tuple[int], np.dtype[np.str_]] = np.array(
    [s.category.value for s in _ROLE_SPECS], dtype="U24"
)

_LOOKUP_MAP: dict[str, GeneRole] = {}

for _role in GeneRole:
    _spec = _ROLE_SPECS[_role]
    _LOOKUP_MAP[_spec.name] = _role
    _LOOKUP_MAP[_spec.name.lower()] = _role
    _LOOKUP_MAP[_spec.abbreviation] = _role
    _LOOKUP_MAP[_spec.abbreviation.lower()] = _role
    _LOOKUP_MAP[_spec.abbreviation.upper()] = _role

    for _al in _spec.aliases:
        _al_clean = _al.strip().lower()
        _LOOKUP_MAP[_al_clean] = _role

_UNIQUE_CATS, _CAT_INDICES = np.unique(_CATEGORIES_ARRAY, return_inverse=True)
_FEATURE_MATRIX: np.ndarray[tuple[int, int], np.dtype[np.float32]] = np.zeros(
    (len(GeneRole), len(_UNIQUE_CATS) + 1), dtype=np.float32
)
_FEATURE_MATRIX[np.arange(len(GeneRole)), _CAT_INDICES] = 1.0
_FEATURE_MATRIX[
    [int(GeneRole.POLYMERASE), int(GeneRole.TRANSMEMBRANE_SYNTHASE), int(GeneRole.GLYCOSYLTRANSFERASE)], -1
] = 1.0
_FEATURE_MATRIX.setflags(write=False)
