"""Canonical biosynthetic gene roles, metadata specifications, and 4-tier regex classification."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum

import numpy as np

_LOCUS_SUFFIX_PATTERN: re.Pattern[str] = re.compile(r"_(?:kl|gi|ph)[0-9a-z_]*$", re.IGNORECASE)


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
    """Immutable metadata and classification specification for a canonical gene role."""

    name: str
    abbreviation: str
    category: RoleCategory
    symbol_regex: str
    desc_regex: str
    description: str
    aliases: tuple[str, ...] = ()
    trusted_prefixes: tuple[str, ...] = ()


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
    def abbr(self) -> str:
        """Alias for `abbreviation`."""
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
    def trusted_prefixes(self) -> tuple[str, ...]:
        """Trusted canonical gene symbol prefixes for Tier-1 classification."""
        return _ROLE_SPECS[self].trusted_prefixes

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

    # ------------------------------------------------------------------------------------------------------------------
    # Single-Pass 4-Tier Classification Engine
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _classify_one(cls, symbol_clean: str, desc_clean: str) -> GeneRole:
        """Core 4-tier single-item classification engine."""
        # Tier 1: Canonical Trusted Core Symbol Prefixes
        if symbol_clean and symbol_clean.startswith(_TRUSTED_PREFIXES):
            for p in _TRUSTED_PREFIXES:
                if symbol_clean.startswith(p):
                    return _TRUSTED_MAP[p]

        # Tier 2: Informative Product Description Regex
        if desc_clean and _DESC_PATTERN is not None:
            match = _DESC_PATTERN.search(desc_clean)
            if match and match.lastgroup:
                return _GROUP_TO_ROLE.get(match.lastgroup, cls.UNKNOWN)

        # Tier 3: Symbol Regex & Aliases Fallback
        if symbol_clean:
            if symbol_clean in _ALIAS_MAP:
                return _ALIAS_MAP[symbol_clean]
            if _SYMBOL_PATTERN is not None:
                match = _SYMBOL_PATTERN.search(symbol_clean)
                if match and match.lastgroup:
                    return _GROUP_TO_ROLE.get(match.lastgroup, cls.UNKNOWN)
            stripped = _LOCUS_SUFFIX_PATTERN.sub("", symbol_clean)
            if stripped != symbol_clean and stripped:
                if stripped in _ALIAS_MAP:
                    return _ALIAS_MAP[stripped]
                if _SYMBOL_PATTERN is not None:
                    match = _SYMBOL_PATTERN.search(stripped)
                    if match and match.lastgroup:
                        return _GROUP_TO_ROLE.get(match.lastgroup, cls.UNKNOWN)

        # Tier 4: Fallback for empty symbol when description starts with a candidate gene identifier
        if not symbol_clean and desc_clean:
            candidate = desc_clean.split()[0].rstrip("*")
            if candidate.startswith(_TRUSTED_PREFIXES):
                for p in _TRUSTED_PREFIXES:
                    if candidate.startswith(p):
                        return _TRUSTED_MAP[p]
            if candidate in _ALIAS_MAP:
                return _ALIAS_MAP[candidate]
            if _SYMBOL_PATTERN is not None:
                match = _SYMBOL_PATTERN.search(candidate)
                if match and match.lastgroup:
                    return _GROUP_TO_ROLE.get(match.lastgroup, cls.UNKNOWN)
                stripped = _LOCUS_SUFFIX_PATTERN.sub("", candidate)
                if stripped != candidate and stripped:
                    if stripped in _ALIAS_MAP:
                        return _ALIAS_MAP[stripped]
                    m_strip = _SYMBOL_PATTERN.search(stripped)
                    if m_strip and m_strip.lastgroup:
                        return _GROUP_TO_ROLE.get(m_strip.lastgroup, cls.UNKNOWN)

        return cls.UNKNOWN

    @classmethod
    def label(
        cls,
        symbol: str | bytes | Sequence[str | bytes] | np.ndarray | None = None,
        description: str | bytes | Sequence[str | bytes] | np.ndarray | None = None,
    ) -> GeneRole | np.ndarray[tuple[int], np.dtype[np.int8]]:
        """Assign gene role indices for a single gene or batch of genes using the 4-tier strategy.

        1. Tier 1: Canonical trusted core symbol prefixes.
        2. Tier 2: Informative product description regex.
        3. Tier 3: Symbol regex and aliases fallback (with locus suffix stripping).
        4. Tier 4: Gene identifier extraction from product description fallback.

        Returns:
            `GeneRole` if inputs are scalar, or `np.ndarray[int8]` if inputs are a sequence/array.
        """
        if isinstance(symbol, (str, bytes, bytearray)) or (
            symbol is None and isinstance(description, (str, bytes, bytearray))
        ):
            s = (
                (symbol.decode("utf-8", errors="replace") if isinstance(symbol, (bytes, bytearray)) else symbol)
                .strip()
                .lower()
                if symbol
                else ""
            )
            d = (
                (
                    description.decode("utf-8", errors="replace")
                    if isinstance(description, (bytes, bytearray))
                    else str(description)
                )
                .strip()
                .lower()
                if description
                else ""
            )
            return cls._classify_one(s, d)

        if symbol is not None:
            sym_arr = symbol if isinstance(symbol, (list, tuple, np.ndarray)) else list(symbol)
            n_items = len(sym_arr)
        elif description is not None:
            desc_arr = description if isinstance(description, (list, tuple, np.ndarray)) else list(description)
            n_items = len(desc_arr)
            sym_arr = ()
        else:
            return cls.UNKNOWN

        if n_items == 0:
            return np.empty(0, dtype=np.int8)

        out = np.empty(n_items, dtype=np.int8)
        classify = cls._classify_one
        has_sym = len(sym_arr) > 0

        if description is None:
            for i in range(n_items):
                s_raw = sym_arr[i]
                if s_raw:
                    s = (
                        (
                            s_raw.decode("utf-8", errors="replace")
                            if isinstance(s_raw, (bytes, bytearray))
                            else str(s_raw)
                        )
                        .strip()
                        .lower()
                    )
                else:
                    s = ""
                out[i] = classify(s, "")
        else:
            desc_arr = description if isinstance(description, (list, tuple, np.ndarray)) else list(description)
            if len(desc_arr) != n_items:
                raise ValueError(f"descriptions length {len(desc_arr)} does not match symbols length {n_items}")
            for i in range(n_items):
                s_raw = sym_arr[i] if has_sym else None
                d_raw = desc_arr[i]
                s = (
                    (s_raw.decode("utf-8", errors="replace") if isinstance(s_raw, (bytes, bytearray)) else str(s_raw))
                    .strip()
                    .lower()
                    if s_raw
                    else ""
                )
                d = (
                    (d_raw.decode("utf-8", errors="replace") if isinstance(d_raw, (bytes, bytearray)) else str(d_raw))
                    .strip()
                    .lower()
                    if d_raw
                    else ""
                )
                out[i] = classify(s, d)

        return out


# ----------------------------------------------------------------------------------------------------------------------
# Canonical Gene Role Specifications (Ordered 0..29 matching GeneRole)
# ----------------------------------------------------------------------------------------------------------------------

_ROLE_SPECS: tuple[RoleSpec, ...] = (
    # 0: FLIPPASE
    RoleSpec(
        name="FLIPPASE",
        abbreviation="Wzx",
        category=RoleCategory.TRANSPORT,
        symbol_regex=r"^(?:wzx|gmla)(?:_[a-z0-9_]+|[a-z0-9]+)?$",
        desc_regex=r"\bflippase\b|\bwzx(?:_[a-z0-9_]+|[a-z0-9]+)?\b|o-antigen flippase|capsular polysaccharide flippase|gtra-like|prophage bactoprenol-linked glucose translocase",
        description="Wzx flippase; translocates completed lipid-linked repeat units across inner membrane.",
        aliases=("wzx", "gmla"),
        trusted_prefixes=("wzx", "gmla"),
    ),
    # 1: POLYMERASE
    RoleSpec(
        name="POLYMERASE",
        abbreviation="Wzy",
        category=RoleCategory.POLYMER_ASSEMBLY,
        symbol_regex=r"^wzy(?:_[a-z0-9_]+|[a-z0-9]+)?$",
        desc_regex=r"\bpolymerase\b|\bwzy(?:_[a-z0-9_]+|[a-z0-9]+)?\b|o-antigen polymerase|capsular polysaccharide polymerase",
        description="Wzy polymerase; forms the polymer repeat unit boundary glycosidic linkage.",
        aliases=("wzy",),
        trusted_prefixes=("wzy",),
    ),
    # 2: COPOLYMERASE
    RoleSpec(
        name="COPOLYMERASE",
        abbreviation="Wzz",
        category=RoleCategory.TRANSPORT,
        symbol_regex=r"^(?:wzz|cld|cps[0-9]*c)(?:_[0-9]+|[a-z0-9])?$",
        desc_regex=r"\bcopolymerase\b|co-polymerase|chain[\s-]length determinant|\bwzz\b",
        description="Wzz polysaccharide co-polymerase; modal chain length regulator.",
        aliases=("wzz", "cld"),
        trusted_prefixes=("wzz", "cld"),
    ),
    # 3: KINASE
    RoleSpec(
        name="KINASE",
        abbreviation="Kin",
        category=RoleCategory.TRANSPORT,
        symbol_regex=r"^(?:wzc|cps[0-9]*d|exop|ptk)(?:_[0-9]+|[a-z0-9])?$",
        desc_regex=r"tyrosine kinase|autokinase|\bwzc\b|by-kinase|protein-tyrosine kinase",
        description="Wzc / BY-kinase; tyrosine autokinase coordinating polymer export.",
        aliases=("wzc", "ptk"),
        trusted_prefixes=("wzc",),
    ),
    # 4: PHOSPHATASE
    RoleSpec(
        name="PHOSPHATASE",
        abbreviation="Wzb",
        category=RoleCategory.TRANSPORT,
        symbol_regex=r"^(?:wzb|cps[0-9]*b|ptp|cpsacp)(?:_[0-9]+|[a-z0-9])?$",
        desc_regex=r"phosphotyrosine phosphatase|tyrosine phosphatase|\bwzb\b|\bptp\b|acid\s*phosphatase|pap2|had[\s-]*superfamily|haloacid\s*dehalogenase|capsule.*(?:biosynthesis\s*)?phosphatase|inositol\s*monophosphatase",
        description="Wzb / PTP / CpsACP; low-molecular-weight phosphotyrosine and acid phosphatases.",
        aliases=("wzb", "ptp", "cpsacp"),
        trusted_prefixes=("wzb", "cpsacp"),
    ),
    # 5: ABC_PERMEASE
    RoleSpec(
        name="ABC_PERMEASE",
        abbreviation="Wzm",
        category=RoleCategory.TRANSPORT,
        symbol_regex=r"^(?:wzm|kpsm|cpsu)(?:_[0-9]+|[a-z0-9])?$",
        desc_regex=r"abc transporter permease|capsule export system permease|\bwzm\b|\bkpsm\b",
        description="Wzm ABC transporter permease; transmembrane translocation channel.",
        aliases=("wzm", "kpsm"),
        trusted_prefixes=("wzm", "kpsm"),
    ),
    # 6: ABC_ATPASE
    RoleSpec(
        name="ABC_ATPASE",
        abbreviation="Wzt",
        category=RoleCategory.TRANSPORT,
        symbol_regex=r"^(?:wzt|kpst|cpsv)(?:_[0-9]+|[a-z0-9])?$",
        desc_regex=r"abc transporter atpase|capsule export system atpase|\bwzt\b|\bkpst\b|abc transporter nucleotide-binding",
        description="Wzt ABC transporter ATPase; nucleotide-binding domain powering export.",
        aliases=("wzt", "kpst"),
        trusted_prefixes=("wzt", "kpst"),
    ),
    # 7: LIGASE
    RoleSpec(
        name="LIGASE",
        abbreviation="WaaL",
        category=RoleCategory.POLYMER_ASSEMBLY,
        symbol_regex=r"^(?:waal|rfal)(?:_[0-9]+|[a-z0-9])?$",
        desc_regex=r"o-antigen ligase|\bwaal\b|\brfal\b|lipid a ligase",
        description="WaaL ligase; attaches periplasmic polysaccharide to Lipid A core.",
        aliases=("waal", "rfal"),
        trusted_prefixes=("waal", "rfal"),
    ),
    # 8: TRANSLOCON
    RoleSpec(
        name="TRANSLOCON",
        abbreviation="Wza",
        category=RoleCategory.TRANSPORT,
        symbol_regex=r"^(?:wza|kps[de]|exoa|opx|gfc[b-d]|yjbh|ymcc)(?:_[a-z0-9_]+|[a-z0-9]+)?$",
        desc_regex=r"outer membrane.*(?:export|capsule)|\bwza\b|\bopx\b|translocon|export pore|\bgfc[b-d]\b|yjbh|ymcc|capsule.*(?:export|translocation).*(?:inner-membrane|periplasmic|protein kpse)",
        description="Wza / KpsD / KpsE outer membrane and periplasmic translocon complex for capsular polysaccharide surface export.",
        aliases=("wza", "gfcd", "gfcb", "gfcc", "kpsd", "kpse"),
        trusted_prefixes=("wza", "kpsd", "kpse", "gfcd", "gfcb", "gfcc"),
    ),
    # 9: CAPSULE_ANCHOR
    RoleSpec(
        name="CAPSULE_ANCHOR",
        abbreviation="Wzi",
        category=RoleCategory.TRANSPORT,
        symbol_regex=r"^(?:wzi|kpsc|wzg|cps[0-9]*a|capa)(?:_[a-z0-9_]+|[a-z0-9]+)?$",
        desc_regex=r"\bwzi\b|\bkpsc\b|\bwzg\b|\bcps[0-9]*a\b|\blytr[\s_-]*cpsa[\s_-]*psr\b|capsule\s*(?:attachment|anchor|assembly)|outer\s*membrane\s*lectin|wzi\s+outer\s+membrane|capsule.*export.*protein\s*kpsc|cell\s*envelope.*(?:transcriptional\s*attenuator|lytr)",
        description="Wzi outer membrane lectin / KpsC lipid anchor / Wzg (CpsA) LytR-CpsA-Psr peptidoglycan capsule anchor transferase.",
        aliases=("wzi", "anchor", "capsule_anchor", "kpsc", "wzg", "cpsa", "cps2a", "capa"),
        trusted_prefixes=("wzi", "kpsc", "wzg", "cpsa", "cps2a"),
    ),
    # 10: PYRUVYLTRANSFERASE
    RoleSpec(
        name="PYRUVYLTRANSFERASE",
        abbreviation="PT",
        category=RoleCategory.MODIFIER,
        symbol_regex=r"^(?:wcak|ptr[0-9]*|cps[0-9]*e|pvt[a-z0-9_]*|wbmx)(?:_[0-9]+|[a-z0-9])?$",
        desc_regex=r"pyruv(?:yl|ate)[\s-]?transferase|pyruvyl|\bwcak\b|\bwbmx\b",
        description="Pyruvyltransferase; adds pyruvate acetal (Pyr) modifications.",
        aliases=("wcak", "ptr", "wbmx"),
        trusted_prefixes=("ptr", "wbmx"),
    ),
    # 11: METHYLTRANSFERASE
    RoleSpec(
        name="METHYLTRANSFERASE",
        abbreviation="MetT",
        category=RoleCategory.MODIFIER,
        symbol_regex=r"^(?:met[0-9]+|omt[0-9]+)(?:_[0-9]+|[a-z0-9])?$",
        desc_regex=r"methyl[\s-]?transferase|o-methyl",
        description="Methyltransferase; adds O-methyl (OMe) groups.",
    ),
    # 12: ACETYLTRANSFERASE
    RoleSpec(
        name="ACETYLTRANSFERASE",
        abbreviation="AT",
        category=RoleCategory.MODIFIER,
        symbol_regex=r"^(?:oat[a-z]*|wecd|wb[ub][a-z]|atr[0-9]*[a-z]?|neud|qhb[ac]|at[0-9]+|fdtc|fdtd|psah|qdte|wcio)(?:_[a-z0-9_]+|[a-z0-9]+)?$",
        desc_regex=r"(?:acetyl|acyl|o-acetyl|n-acetyl)[\s-]?transferase|acetyl-?\s*(?:or|/)\s*acyl|\bwecd\b|\boata\b|\batr[0-9]*[a-z]?\b|\bwcio\b",
        description="Acetyltransferase / Acyltransferase; adds O-acetyl (OAc) or hydroxyacyl (3HOBut, 3HOMyr) groups.",
        aliases=("oata", "wecd", "atr", "qdte", "wcio"),
        trusted_prefixes=("atr", "qdte"),
    ),
    # 13: PHOSPHOTRANSFERASE
    RoleSpec(
        name="PHOSPHOTRANSFERASE",
        abbreviation="PhoT",
        category=RoleCategory.MODIFIER,
        symbol_regex=r"^(?:wcf[a-z]|pt[0-9]+|pgt[0-9]*|licd)(?:_[0-9]+|[a-z0-9])?$",
        desc_regex=r"(?:phospho(?:ethanolamine|choline|glycerol)?|cholinephospho|glycerolphospho)[\s-]?transferase|\bpgt\b|\blicd\b|phosphocholine|choline\s*phosphotransferase",
        description="Phosphotransferase; adds phosphate (P), phosphoethanolamine (PEtn), or phosphocholine (ChoP) groups.",
        aliases=("licd",),
    ),
    # 14: FORMYLTRANSFERASE
    RoleSpec(
        name="FORMYLTRANSFERASE",
        abbreviation="ForT",
        category=RoleCategory.MODIFIER,
        symbol_regex=r"^(?:fmt[0-9]+|for[a-z0-9_]*)(?:_[0-9]+|[a-z0-9])?$",
        desc_regex=r"formyl[\s-]?transferase|n-formyl",
        description="Formyltransferase; adds formyl (Fo) modifications.",
    ),
    # 15: ALANINE_TRANSFERASE
    RoleSpec(
        name="ALANINE_TRANSFERASE",
        abbreviation="Alt",
        category=RoleCategory.MODIFIER,
        symbol_regex=r"^(?:alt[0-9]*|aat[0-9]*)(?:_[a-z0-9_]+|[a-z0-9]+)?$",
        desc_regex=r"alanine\s*(?:amide\s*)?transferase|\balt[0-9]*\b|aminoacyl[\s-]transferase",
        description="Alt1 alanine transferase; transfers D-alanine to the carboxyl group of GalA/GalNAcA forming an amide.",
        aliases=("alt", "alt1", "ala_transferase"),
    ),
    # 16: AMINOTRANSFERASE
    RoleSpec(
        name="AMINOTRANSFERASE",
        abbreviation="ATn",
        category=RoleCategory.PRECURSOR_SYNTHESIS,
        symbol_regex=r"^(?:wece|psec|legc|degt|qhbb|qnr[a-z0-9_]*|qdtb|psab|rhn[bc]|wei[pq])(?:_[0-9]+|[a-z0-9])?$",
        desc_regex=r"amino[\s-]?transferase|transaminase|\bwece\b|\bpsec\b|\blegc\b|aminase|\brhn[bc]\b",
        description="Aminotransferase / Transaminase (e.g. WecE, RhnB, RhnC); converts keto-intermediates to amino sugars.",
        aliases=("wece", "rhnb", "rhnc"),
    ),
    # 17: ALDOLASE
    RoleSpec(
        name="ALDOLASE",
        abbreviation="Ald",
        category=RoleCategory.PRECURSOR_SYNTHESIS,
        symbol_regex=r"^(?:neub|psei|legi|kds[ab]|nana|psaf)(?:_[a-z0-9_]+|[a-z0-9]+)?$",
        desc_regex=r"\baldolase\b|\bcondensase\b|(?:sialic|pseudaminic|legionaminic|neuraminic|n-acetylneuraminic)[\s-]acid[\s-]?synthase|(?:neu5ac|n-acetylneuraminate)[\s-]?synthase|kdo(?:-8-phosphate)?[\s-]?synthase|\bneub\b|\bpsei\b|\blegi\b",
        description="PEP-condensing aldolase/synthase (e.g. NeuB, PseI, LegI); synthesizes nonulosonic acids.",
        aliases=("neub", "psei", "legi"),
    ),
    # 18: DEACETYLASE
    RoleSpec(
        name="DEACETYLASE",
        abbreviation="DAc",
        category=RoleCategory.MODIFIER,
        symbol_regex=r"^(?:pgab|pda[0-9a-z_]*|dac[a-z0-9_]*|ahy)(?:_[0-9]+|[a-z0-9])?$",
        desc_regex=r"de-?acetylase|\bpgab\b|\bpdaa?\b|\bpda[0-9a-z]*\b|\bahy\b|acylhydrolase",
        description="Deacetylase; generates free amino sugar derivatives.",
        aliases=("pgab", "pda1", "pda2", "pda2a", "ahy"),
        trusted_prefixes=("pda", "ahy"),
    ),
    # 19: EPIMERASE
    RoleSpec(
        name="EPIMERASE",
        abbreviation="Epi",
        category=RoleCategory.PRECURSOR_SYNTHESIS,
        symbol_regex=r"^(?:gale|rmlc|rfbc|wbpp|neuc|wca[gi]|glf|manb|pgm|gpi|mnaa|cps[0-9]*g|rfbk|gne[0-9]*|gne|fdta|fdtb|kpsf|gutq|kdsd)(?:_[0-9]+|[a-z0-9])?$",
        desc_regex=r"\bepimerase(?!\s*[/\\-]\s*dehydratase)\b|\bracemase\b|\bisomerase\b|ketoisomerase|(?:(?:udp-)?galacto(?:pyranose)?|phospho(?:manno|gluco|hexo)|(?:hexose|sugar)[\s-]phosphate)?\s*mutase|\bgale\b|\bwbpp\b|\brmlc\b|\bneuc\b|\bgpi\b|\bgne\b|dtdp-4-dehydrorhamnose\s*3|arabinose-5-phosphate\s*isomerase",
        description="Epimerase / Isomerase; interconverts UDP-sugar stereoisomers and precursor intermediates.",
        aliases=("gale", "rmlc", "neuc", "gne", "rfbc", "kpsf"),
        trusted_prefixes=("rmlc", "kpsf", "gpi", "pgm", "rfbc"),
    ),
    # 20: OXIDOREDUCTASE
    RoleSpec(
        name="OXIDOREDUCTASE",
        abbreviation="OxR",
        category=RoleCategory.PRECURSOR_SYNTHESIS,
        symbol_regex=r"^(?:rml[bd]|rfb[bd]|gmd|fcl|ugd|hasb|gna[0-9]*|fnr[0-9]*|rhna|weis|epab|epad|mnab)(?:_[0-9]+|[a-z0-9])?$",
        desc_regex=r"dehydrogenase|reductase|dehydratase|oxidoreductase|epimer(?:ase|ise)/dehydratase|\brmlb\b|\brmld\b|\brhna\b|fucosamine biosynthesis protein|dtdp-glucose\s*4",
        description="Reductase / Dehydratase (e.g. RmlB, RhnA); synthesizes deoxyhexoses, dideoxyhexoses, and alditol reducing ends.",
        aliases=("rmlb", "rmld", "rhna", "epab", "epad", "rfbb", "rfbd"),
        trusted_prefixes=("rmlb", "rmld", "ugd", "gnd", "rfbb", "rfbd"),
    ),
    # 21: TRANSMEMBRANE_SYNTHASE
    RoleSpec(
        name="TRANSMEMBRANE_SYNTHASE",
        abbreviation="TMS",
        category=RoleCategory.POLYMER_ASSEMBLY,
        symbol_regex=r"^(?:hasa|pgac|icaa|bcsa|alg8|cps3s)(?:_[0-9]+|[a-z0-9])?$",
        desc_regex=r"(?:hyaluron(?:an|ic\s?acid)|cellulose|poly-.*glucosamine|poly-beta-1,6.*|alginate)[\s-]?(?:synthase|polymerase)|processive[\s-]?glycosyltransferase|type[\s-]3[\s-]?synthase|\bhasa\b|\bpgac\b|\bbcsa\b",
        description="Processive transmembrane glycosyltransferase (e.g. HasA, PgaC, BcsA); polymerizes and extrudes glycan chains.",
        aliases=("hasa", "pgac", "bcsa"),
    ),
    # 22: NUCLEOTIDYLTRANSFERASE
    RoleSpec(
        name="NUCLEOTIDYLTRANSFERASE",
        abbreviation="NT",
        category=RoleCategory.PRECURSOR_SYNTHESIS,
        symbol_regex=r"^(?:rmla|rfba|galu|neua|glmm|glmu|man[bc]|hasc|wbp[vw]|psac|lgad|lgag|lgah|epaa|epac)(?:_[a-z0-9_]+|[a-z0-9]+)?$",
        desc_regex=r"(?:nucleotidyl|thymidylyl|uridylyl|cytidylyl|guanylyl|adenylyl)[\s-]?(?:transferase|diphosphorylase)|pyrophosphorylase|\brmla\b|\brfba\b|\bgalu\b|\bneua\b|\bglmm\b|\bglmu\b|\bcytidyltransferase\b|synthetase; involved in synthesis of legionaminic",
        description="Nucleotide-sugar precursor synthase; acts on base monosaccharide pathways.",
        aliases=("rmla", "galu", "epaa", "epac"),
        trusted_prefixes=("rmla", "galu", "galf"),
    ),
    # 23: NUCLEOTIDASE
    RoleSpec(
        name="NUCLEOTIDASE",
        abbreviation="Nuc",
        category=RoleCategory.PRECURSOR_SYNTHESIS,
        symbol_regex=r"^(?:psa[dgi]|lgaf|gmm|wcah)(?:_[a-z0-9_]+|[a-z0-9]+)?$",
        desc_regex=r"\bnucleotidase\b|sugar[\s-]nucleotide\s*hydrolase|gdp-mannose\s*(?:mannosyl\s*)?hydrolase",
        description="Sugar-nucleotide nucleotidase / hydrolase (e.g. PsaD, PsaG, PsaI, LgaF, Gmm); cleaves sugar-nucleotide intermediates.",
        aliases=("psad", "psag", "psai", "lgaf", "gmm", "wcah", "sugar_phosphatase"),
        trusted_prefixes=("gmm", "wcah"),
    ),
    # 24: INITIATOR
    RoleSpec(
        name="INITIATOR",
        abbreviation="Init",
        category=RoleCategory.POLYMER_ASSEMBLY,
        symbol_regex=r"^(?:wbap|wcaj|itr[ab]|tago|mray|cps[0-9]*a|wcfs|weca|wcha)(?:_[0-9]+|[a-z0-9])?$",
        desc_regex=r"(?:initial|initiating).*transferase|undecaprenyl-phosphate.*transferase|undecaprenyl.*phosphotransferase|polyisoprenyl-teichoic.*transferase|glucosyl[\s-]?(?:1-)?phosphate[\s-]?transferase|\bwcaj\b|\bwbap\b|\bitra\b|\bitrb\b|\bwcha\b",
        description="Initial sugar transferase (e.g. WbaP/WcaJ/WchA); attaches first non-modifier sugar onto undecaprenyl phosphate.",
        aliases=("wbap", "wcaj", "wcha", "initiating_gt"),
        trusted_prefixes=("wbap", "wcaj", "wcha", "itra", "itrb", "weca"),
    ),
    # 25: GLYCOSYLHYDROLASE
    RoleSpec(
        name="GLYCOSYLHYDROLASE",
        abbreviation="GH",
        category=RoleCategory.MODIFIER,
        symbol_regex=r"^(?:gh[0-9]+|ghl[0-9]*|gnlb|cps[0-9]*g|orf1\(ghy\)|ghy)(?:_[0-9]+|[a-z0-9])?$",
        desc_regex=r"glycosyl[\s-]?hydrolase|(?:gluc|galact|mann)osidase|\bheparinase\b|\bgnlb\b|\blyase\b|alginate\s*lyase|pectate\s*lyase|depolymerase",
        description="Glycosylhydrolase / lyase; processes, depolymerizes, or cleaves glycosidic linkages.",
        aliases=("gnlb", "ghl", "orf1(ghy)", "ghy"),
        trusted_prefixes=("ghl", "orf1(ghy)", "ghy"),
    ),
    # 26: GLYCOSYLTRANSFERASE
    RoleSpec(
        name="GLYCOSYLTRANSFERASE",
        abbreviation="GT",
        category=RoleCategory.POLYMER_ASSEMBLY,
        symbol_regex=r"^(?:wb[dsyz][a-z]|gtr[0-9a-z]*|gtroc[0-9]*|glt[0-9]*|gt[0-9]+|wf[efg][a-z]|kfo[a-z]|kfi[ac]|kpss[0-9]*|kpss|htrl[0-9a-z*]*|sps[a-z]|wbm[vw]|gml[bc]|rhamnan|rgpf)(?:_[0-9]+|[a-z0-9])?$",
        desc_regex=r"(?:glyco(?:syl)?|sugar|sialyl|polysialyl|kdo|(?:gluc|galact(?:ofuran)?|mann|fuc|rhamn|glucuron|galacturon|xyl)osyl|(?:glucos|galactos|mannos)aminyl)[\s-]?transferase|oct-2-ulosonic\s*acid\s*transferase|pseudaminyltransferase|\bhtrl\b|wca[al]\b|rhamnan\s*synthesis\s*protein",
        description="Glycosyltransferase; synthesizes glycosidic linkages and non-modifier monosaccharide additions.",
        aliases=("kpss", "gtr", "glt", "htrl", "wbmv", "wbmw", "gmlc", "spsa", "wcaa", "rhamnan", "rgpf"),
        trusted_prefixes=("gtr", "htrl", "wbmv", "wbmw", "gmlc", "spsa", "wcaa", "rhamnan", "rgpf"),
    ),
    # 27: TRANSPOSASE
    RoleSpec(
        name="TRANSPOSASE",
        abbreviation="Tnp",
        category=RoleCategory.MOBILE_ELEMENT,
        symbol_regex=r"^(?:tnp[0-9a-z_]*|is[0-9a-z_]+|ins[0-9a-z_]*)(?:_[0-9]+|[a-z0-9])?$",
        desc_regex=r"\btransposase\b|\btransposon\b|transposase-like|\binsertion\s*sequence\b|\bis[0-9a-z_]+(?:\s*family)?\s*(?:transposase|element)?\b|\bis[0-9a-z_]+\s*genes\b",
        description="Transposase / Insertion sequence (IS) element; mobile genetic element non-biosynthetic infiltrator.",
        aliases=("tnp", "transposase", "is_element", "insertion_sequence"),
        trusted_prefixes=("tnp",),
    ),
    # 28: INTEGRASE
    RoleSpec(
        name="INTEGRASE",
        abbreviation="Int",
        category=RoleCategory.MOBILE_ELEMENT,
        symbol_regex=r"^(?:int[0-9a-z_]*|xis[0-9a-z_]*|rec[0-9a-z_]*)(?:_[0-9]+|[a-z0-9])?$",
        desc_regex=r"\bintegrase\b|\brecombinase\b|\bexcisionase\b|phage\s*integrase|site-specific\s*recombinase|\bresolvase\b",
        description="Integrase / Recombinase / Excisionase; mediates DNA integration/excision, prophage insertions, or mobile cassette movement.",
        aliases=("int", "integrase", "recombinase", "excisionase"),
    ),
    # 29: UNKNOWN
    RoleSpec(
        name="UNKNOWN",
        abbreviation="?",
        category=RoleCategory.BACKGROUND,
        symbol_regex=r"^$",
        desc_regex=r"^$",
        description="Uncharacterized gene; background or non-biosynthetic role.",
    ),
)

# ----------------------------------------------------------------------------------------------------------------------
# Precomputed Lookup Tables, Compiled Regexes & Vectorized Arrays
# ----------------------------------------------------------------------------------------------------------------------

_NAMES_ARRAY: np.ndarray[tuple[int], np.dtype[np.str_]] = np.array([s.name for s in _ROLE_SPECS], dtype="U24")
_ABBREVIATIONS_ARRAY: np.ndarray[tuple[int], np.dtype[np.str_]] = np.array(
    [s.abbreviation for s in _ROLE_SPECS], dtype="U8"
)
_CATEGORIES_ARRAY: np.ndarray[tuple[int], np.dtype[np.str_]] = np.array(
    [s.category.value for s in _ROLE_SPECS], dtype="U24"
)

_LOOKUP_MAP: dict[str, GeneRole] = {}
_ALIAS_MAP: dict[str, GeneRole] = {}
_GROUP_TO_ROLE: dict[str, GeneRole] = {}
_ALL_TRUSTED: list[tuple[str, GeneRole]] = []
_SYM_PARTS: list[str] = []
_DESC_PARTS: list[str] = []

for _role in GeneRole:
    _spec = _ROLE_SPECS[_role]
    _LOOKUP_MAP[_spec.name] = _role
    _LOOKUP_MAP[_spec.name.lower()] = _role
    _LOOKUP_MAP[_spec.abbreviation] = _role
    _LOOKUP_MAP[_spec.abbreviation.lower()] = _role
    _LOOKUP_MAP[_spec.abbreviation.upper()] = _role
    _GROUP_TO_ROLE[_spec.name] = _role

    for _al in _spec.aliases:
        _al_clean = _al.strip().lower()
        _ALIAS_MAP[_al_clean] = _role
        _LOOKUP_MAP[_al_clean] = _role

    for _p in _spec.trusted_prefixes:
        _p_clean = _p.strip().lower()
        _ALL_TRUSTED.append((_p_clean, _role))

    if _role != GeneRole.UNKNOWN:
        _SYM_PARTS.append(f"(?P<{_spec.name}>{_spec.symbol_regex})")
        _DESC_PARTS.append(f"(?P<{_spec.name}>{_spec.desc_regex})")

_SORTED_TRUSTED = sorted(_ALL_TRUSTED, key=lambda item: len(item[0]), reverse=True)
_TRUSTED_PREFIXES: tuple[str, ...] = tuple(p for p, _ in _SORTED_TRUSTED)
_TRUSTED_MAP: dict[str, GeneRole] = dict(_SORTED_TRUSTED)

_SYMBOL_PATTERN: re.Pattern[str] | None = re.compile("|".join(_SYM_PARTS), re.IGNORECASE) if _SYM_PARTS else None
_DESC_PATTERN: re.Pattern[str] | None = re.compile("|".join(_DESC_PARTS), re.IGNORECASE) if _DESC_PARTS else None

_UNIQUE_CATS, _CAT_INDICES = np.unique(_CATEGORIES_ARRAY, return_inverse=True)
_FEATURE_MATRIX: np.ndarray[tuple[int, int], np.dtype[np.float32]] = np.zeros(
    (len(GeneRole), len(_UNIQUE_CATS) + 1), dtype=np.float32
)
_FEATURE_MATRIX[np.arange(len(GeneRole)), _CAT_INDICES] = 1.0
_FEATURE_MATRIX[
    [int(GeneRole.POLYMERASE), int(GeneRole.TRANSMEMBRANE_SYNTHASE), int(GeneRole.GLYCOSYLTRANSFERASE)], -1
] = 1.0
_FEATURE_MATRIX.setflags(write=False)

GeneRoles = GeneRole
