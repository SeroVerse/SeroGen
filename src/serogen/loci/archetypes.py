"""Biosynthetic locus archetypes, ontology categories, and structural specifications."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from functools import lru_cache
from typing import Self

import numpy as np

from serogen.genes.roles import GeneRole


def roles_to_mask(roles: Iterable[int | GeneRole] | np.ndarray) -> int:
    """Pack a collection of `GeneRole` values into a 32-bit integer presence bitmask."""
    mask = 0
    if isinstance(roles, np.ndarray):
        if roles.size == 0:
            return 0
        valid = roles[(roles >= 0) & (roles < 30)]
        if valid.size == 0:
            return 0
        return int(np.bitwise_or.reduce(1 << valid.astype(np.int64)))

    for r in roles:
        val = int(r)
        if 0 <= val < 30:
            mask |= 1 << val
    return mask


def mask_to_roles(mask: int) -> tuple[GeneRole, ...]:
    """Unpack a 32-bit integer presence bitmask into a sorted tuple of `GeneRole` enums."""
    return tuple(role for role in GeneRole if (mask & (1 << int(role))) != 0)


class ArchetypeCategory(StrEnum):
    """High-level structural category of a polysaccharide biosynthetic locus archetype."""

    CAPSULE = "Capsule"
    O_ANTIGEN = "O-Antigen"
    EXOPOLYSACCHARIDE = "Exopolysaccharide"
    OTHER = "Other"


@dataclass(frozen=True, slots=True)
class ArchetypeSpec:
    """Immutable biological specification for a canonical `LocusArchetype`."""

    name: str
    category: ArchetypeCategory
    forbidden_mask: int
    cargo_mask: int
    description: str
    module_masks: tuple[int, ...] = ()
    aliases: tuple[str, ...] = ()

    @property
    def required_mask(self) -> int:
        """Combined 32-bit union mask of all roles participating in any required module."""
        mask = 0
        for m in self.module_masks:
            mask |= m
        return mask

    @property
    def all_relevant_mask(self) -> int:
        """Combined 32-bit bitmask of all required and optional cargo roles."""
        return self.required_mask | self.cargo_mask

    @property
    def num_modules(self) -> int:
        """Number of required functional modules (`M_A`) for this archetype."""
        return len(self.module_masks)


class LocusArchetype(IntEnum):
    """Canonical biosynthetic locus archetypes (0..10).

    Defines the ontological classification and structural requirements of bacterial
    surface polysaccharide biosynthetic gene clusters. Scoring, completeness evaluation,
    and segment classification are encapsulated in [`ArchetypeClassifier`][serogen.loci.classifier.ArchetypeClassifier].
    """

    CPS_WZY = 0
    CPS_ABC = 1
    CPS_SYNTHASE = 2
    LPS_WZY = 3
    LPS_ABC = 4
    EPS_PEL = 5
    EPS_PGA = 6
    EPS_ALGINATE = 7
    EPS_BCS = 8
    TEICHOIC_ACID = 9
    UNKNOWN = 10

    @property
    def spec(self) -> ArchetypeSpec:
        """Return the `ArchetypeSpec` metadata record for this archetype."""
        return _get_archetype_specs()[int(self)]

    @property
    def category(self) -> ArchetypeCategory:
        """High-level `ArchetypeCategory` (`Capsule`, `O-Antigen`, `Exopolysaccharide`, `Other`)."""
        return self.spec.category

    @property
    def forbidden_mask(self) -> int:
        """32-bit bitmask of `GeneRole` bits that are incompatible with this archetype."""
        return self.spec.forbidden_mask

    @property
    def cargo_mask(self) -> int:
        """32-bit bitmask of optional cargo/modifier/precursor `GeneRole` bits."""
        return self.spec.cargo_mask

    @property
    def required_mask(self) -> int:
        """Combined 32-bit union mask of all roles participating in any required module."""
        return self.spec.required_mask

    @property
    def all_relevant_mask(self) -> int:
        """Combined 32-bit bitmask of all required and optional cargo roles."""
        return self.spec.all_relevant_mask

    @property
    def description(self) -> str:
        """Human-readable biological description of the pathway."""
        return self.spec.description

    @property
    def module_masks(self) -> tuple[int, ...]:
        """Tuple of 32-bit disjunctive module bitmasks (`M_1, ..., M_k`) required for completeness."""
        return self.spec.module_masks

    @property
    def num_modules(self) -> int:
        """Number of required functional modules (`M_A`) for this archetype."""
        return self.spec.num_modules

    @classmethod
    def from_name(cls, key: str | bytes | int | LocusArchetype) -> LocusArchetype:
        """Resolve an archetype by integer ID, enum name, or biological alias."""
        if isinstance(key, cls):
            return key
        if isinstance(key, (int, np.integer)):
            return cls(int(key))
        s = key.decode("utf-8", errors="replace") if isinstance(key, (bytes, bytearray)) else str(key)
        s_clean = s.strip().lower().replace("-", "_").replace(" ", "_")
        lookup = _get_archetype_lookup()
        if s_clean in lookup:
            return lookup[s_clean]
        raise KeyError(f"Unknown LocusArchetype identifier: {key!r}")

    @classmethod
    def resolve(
        cls,
        key: str | bytes | int | LocusArchetype,
        default: LocusArchetype | None = None,
    ) -> LocusArchetype:
        """Resolve an archetype identifier, falling back to `default` (or `UNKNOWN`)."""
        fallback = cls.UNKNOWN if default is None else default
        try:
            return cls.from_name(key)
        except KeyError, ValueError:
            return fallback


@lru_cache(maxsize=1)
def _get_archetype_specs() -> tuple[ArchetypeSpec, ...]:
    """Build and cache the ordered tuple of canonical `ArchetypeSpec` definitions (0..10)."""
    common_precursors_and_mods = roles_to_mask(
        (
            GeneRole.GLYCOSYLTRANSFERASE,
            GeneRole.INITIATOR,
            GeneRole.PYRUVYLTRANSFERASE,
            GeneRole.METHYLTRANSFERASE,
            GeneRole.ACETYLTRANSFERASE,
            GeneRole.PHOSPHOTRANSFERASE,
            GeneRole.FORMYLTRANSFERASE,
            GeneRole.ALANINE_TRANSFERASE,
            GeneRole.AMINOTRANSFERASE,
            GeneRole.ALDOLASE,
            GeneRole.DEACETYLASE,
            GeneRole.EPIMERASE,
            GeneRole.OXIDOREDUCTASE,
            GeneRole.NUCLEOTIDYLTRANSFERASE,
            GeneRole.NUCLEOTIDASE,
            GeneRole.GLYCOSYLHYDROLASE,
        )
    )
    return (
        # 0: CPS_WZY
        ArchetypeSpec(
            name="CPS_WZY",
            category=ArchetypeCategory.CAPSULE,
            module_masks=(
                GeneRole.TRANSLOCON.bitmask | GeneRole.CAPSULE_ANCHOR.bitmask | GeneRole.PHOSPHATASE.bitmask,
                GeneRole.KINASE.bitmask,
                GeneRole.FLIPPASE.bitmask,
                GeneRole.POLYMERASE.bitmask | GeneRole.LIGASE.bitmask | GeneRole.COPOLYMERASE.bitmask,
                GeneRole.INITIATOR.bitmask | GeneRole.GLYCOSYLTRANSFERASE.bitmask,
            ),
            forbidden_mask=roles_to_mask(
                (
                    GeneRole.ABC_PERMEASE,
                    GeneRole.ABC_ATPASE,
                    GeneRole.TRANSMEMBRANE_SYNTHASE,
                )
            ),
            cargo_mask=common_precursors_and_mods
            | roles_to_mask(
                (
                    GeneRole.PHOSPHATASE,
                    GeneRole.CAPSULE_ANCHOR,
                    GeneRole.COPOLYMERASE,
                    GeneRole.LIGASE,
                )
            ),
            description=(
                "Group 1 / 4 Wzy-dependent capsular polysaccharide locus (K-locus); "
                "requires outer membrane export (Wza/Wzi), chain-length regulation (Wzc/Wzb/Wzz), "
                "flippase (Wzx), polymerase (Wzy), and initiating/repeating glycosyltransferases."
            ),
            aliases=("k_locus", "klocus", "cps_group1", "cps_group4", "wzy_capsule"),
        ),
        # 1: CPS_ABC
        ArchetypeSpec(
            name="CPS_ABC",
            category=ArchetypeCategory.CAPSULE,
            module_masks=(
                GeneRole.TRANSLOCON.bitmask | GeneRole.CAPSULE_ANCHOR.bitmask,
                GeneRole.ABC_PERMEASE.bitmask,
                GeneRole.ABC_ATPASE.bitmask,
                GeneRole.GLYCOSYLTRANSFERASE.bitmask | GeneRole.INITIATOR.bitmask,
            ),
            forbidden_mask=roles_to_mask(
                (
                    GeneRole.FLIPPASE,
                    GeneRole.POLYMERASE,
                    GeneRole.TRANSMEMBRANE_SYNTHASE,
                )
            ),
            cargo_mask=common_precursors_and_mods | GeneRole.CAPSULE_ANCHOR.bitmask,
            description=(
                "Group 2 / 3 ABC-transporter-dependent capsular polysaccharide locus (kps cluster); "
                "requires KpsD/KpsE translocon, KpsM permease, KpsT ATPase, and glycosyltransferases."
            ),
            aliases=("kps", "cps_group2", "cps_group3", "abc_capsule"),
        ),
        # 2: CPS_SYNTHASE
        ArchetypeSpec(
            name="CPS_SYNTHASE",
            category=ArchetypeCategory.CAPSULE,
            module_masks=(GeneRole.TRANSMEMBRANE_SYNTHASE.bitmask,),
            forbidden_mask=roles_to_mask(
                (
                    GeneRole.FLIPPASE,
                    GeneRole.POLYMERASE,
                    GeneRole.ABC_PERMEASE,
                    GeneRole.ABC_ATPASE,
                    GeneRole.DEACETYLASE,
                    GeneRole.GLYCOSYLHYDROLASE,
                )
            ),
            cargo_mask=common_precursors_and_mods
            | roles_to_mask(
                (
                    GeneRole.CAPSULE_ANCHOR,
                    GeneRole.PHOSPHATASE,
                    GeneRole.KINASE,
                    GeneRole.COPOLYMERASE,
                    GeneRole.PHOSPHOTRANSFERASE,
                )
            ),
            description=(
                "Synthase-dependent capsule locus (e.g., HasA hyaluronic acid, Cps3S, or ectopic Tts synthase); "
                "coupled polymerization and transmembrane translocation by a single processive synthase."
            ),
            aliases=("synthase_capsule", "has_locus", "cps3", "tts"),
        ),
        # 3: LPS_WZY
        ArchetypeSpec(
            name="LPS_WZY",
            category=ArchetypeCategory.O_ANTIGEN,
            module_masks=(
                GeneRole.FLIPPASE.bitmask,
                GeneRole.POLYMERASE.bitmask | GeneRole.LIGASE.bitmask | GeneRole.COPOLYMERASE.bitmask,
                GeneRole.GLYCOSYLTRANSFERASE.bitmask | GeneRole.INITIATOR.bitmask,
            ),
            forbidden_mask=roles_to_mask(
                (
                    GeneRole.TRANSLOCON,
                    GeneRole.CAPSULE_ANCHOR,
                    GeneRole.KINASE,
                    GeneRole.ABC_PERMEASE,
                    GeneRole.ABC_ATPASE,
                    GeneRole.TRANSMEMBRANE_SYNTHASE,
                )
            ),
            cargo_mask=common_precursors_and_mods
            | roles_to_mask(
                (
                    GeneRole.COPOLYMERASE,
                    GeneRole.LIGASE,
                )
            ),
            description=(
                "Wzy-dependent lipopolysaccharide O-antigen locus (O-locus); "
                "requires Wzx flippase, Wzy polymerase, and glycosyltransferases without Wza/Wzc capsule export."
            ),
            aliases=("o_locus", "olocus", "wzy_o_antigen", "rfb_wzy"),
        ),
        # 4: LPS_ABC
        ArchetypeSpec(
            name="LPS_ABC",
            category=ArchetypeCategory.O_ANTIGEN,
            module_masks=(
                GeneRole.ABC_PERMEASE.bitmask,
                GeneRole.ABC_ATPASE.bitmask,
                GeneRole.GLYCOSYLTRANSFERASE.bitmask | GeneRole.INITIATOR.bitmask,
            ),
            forbidden_mask=roles_to_mask(
                (
                    GeneRole.TRANSLOCON,
                    GeneRole.CAPSULE_ANCHOR,
                    GeneRole.KINASE,
                    GeneRole.FLIPPASE,
                    GeneRole.POLYMERASE,
                    GeneRole.TRANSMEMBRANE_SYNTHASE,
                    GeneRole.ALANINE_TRANSFERASE,
                )
            ),
            cargo_mask=(common_precursors_and_mods & ~GeneRole.ALANINE_TRANSFERASE.bitmask) | GeneRole.LIGASE.bitmask,
            description=(
                "ABC-transporter-dependent lipopolysaccharide O-antigen locus; "
                "requires Wzm permease, Wzt ATPase, and glycosyltransferases without outer-membrane Wza/KpsD translocon."
            ),
            aliases=("abc_o_antigen", "wzm_wzt_o_antigen", "rfb_abc"),
        ),
        # 5: EPS_PEL
        ArchetypeSpec(
            name="EPS_PEL",
            category=ArchetypeCategory.EXOPOLYSACCHARIDE,
            module_masks=(
                GeneRole.GLYCOSYLTRANSFERASE.bitmask | GeneRole.TRANSMEMBRANE_SYNTHASE.bitmask,
                GeneRole.DEACETYLASE.bitmask,
                GeneRole.GLYCOSYLHYDROLASE.bitmask,
            ),
            forbidden_mask=roles_to_mask(
                (
                    GeneRole.FLIPPASE,
                    GeneRole.POLYMERASE,
                    GeneRole.ABC_PERMEASE,
                    GeneRole.ABC_ATPASE,
                )
            ),
            cargo_mask=common_precursors_and_mods | GeneRole.TRANSLOCON.bitmask,
            description="Pel biofilm exopolysaccharide locus (PelA deacetylase/hydrolase, PelF GT, PelD/E/G machinery).",
            aliases=("pel", "pel_locus"),
        ),
        # 6: EPS_PGA
        ArchetypeSpec(
            name="EPS_PGA",
            category=ArchetypeCategory.EXOPOLYSACCHARIDE,
            module_masks=(
                GeneRole.TRANSMEMBRANE_SYNTHASE.bitmask,
                GeneRole.DEACETYLASE.bitmask,
            ),
            forbidden_mask=roles_to_mask(
                (
                    GeneRole.FLIPPASE,
                    GeneRole.POLYMERASE,
                    GeneRole.ABC_PERMEASE,
                    GeneRole.ABC_ATPASE,
                )
            ),
            cargo_mask=common_precursors_and_mods | GeneRole.TRANSLOCON.bitmask,
            description="PNAG / PgaABCD / IcaADBC poly-beta-1,6-GlcNAc biofilm exopolysaccharide locus.",
            aliases=("pga", "ica", "pnag"),
        ),
        # 7: EPS_ALGINATE
        ArchetypeSpec(
            name="EPS_ALGINATE",
            category=ArchetypeCategory.EXOPOLYSACCHARIDE,
            module_masks=(
                GeneRole.TRANSMEMBRANE_SYNTHASE.bitmask,
                GeneRole.OXIDOREDUCTASE.bitmask | GeneRole.EPIMERASE.bitmask,
                GeneRole.GLYCOSYLHYDROLASE.bitmask,
                GeneRole.ACETYLTRANSFERASE.bitmask,
            ),
            forbidden_mask=roles_to_mask(
                (
                    GeneRole.FLIPPASE,
                    GeneRole.POLYMERASE,
                    GeneRole.ABC_PERMEASE,
                    GeneRole.ABC_ATPASE,
                )
            ),
            cargo_mask=common_precursors_and_mods | GeneRole.TRANSLOCON.bitmask | GeneRole.CAPSULE_ANCHOR.bitmask,
            description="Alginate exopolysaccharide locus (Alg8 synthase, AlgD/AlgG dehydrogenase/epimerase, AlgL lyase, AlgI/J/F acetylation).",
            aliases=("alg", "alginate"),
        ),
        # 8: EPS_BCS
        ArchetypeSpec(
            name="EPS_BCS",
            category=ArchetypeCategory.EXOPOLYSACCHARIDE,
            module_masks=(
                GeneRole.TRANSMEMBRANE_SYNTHASE.bitmask,
                GeneRole.GLYCOSYLHYDROLASE.bitmask,
            ),
            forbidden_mask=roles_to_mask(
                (
                    GeneRole.FLIPPASE,
                    GeneRole.POLYMERASE,
                    GeneRole.ABC_PERMEASE,
                    GeneRole.ABC_ATPASE,
                    GeneRole.DEACETYLASE,
                    GeneRole.OXIDOREDUCTASE,
                    GeneRole.CAPSULE_ANCHOR,
                    GeneRole.PHOSPHATASE,
                    GeneRole.COPOLYMERASE,
                    GeneRole.KINASE,
                )
            ),
            cargo_mask=common_precursors_and_mods | GeneRole.TRANSLOCON.bitmask,
            description="Bacterial cellulose synthesis (BcsABZC) exopolysaccharide locus.",
            aliases=("bcs", "cellulose"),
        ),
        # 9: TEICHOIC_ACID
        ArchetypeSpec(
            name="TEICHOIC_ACID",
            category=ArchetypeCategory.OTHER,
            module_masks=(
                GeneRole.ABC_PERMEASE.bitmask | GeneRole.FLIPPASE.bitmask,
                GeneRole.ABC_ATPASE.bitmask | GeneRole.PHOSPHOTRANSFERASE.bitmask,
                GeneRole.PHOSPHOTRANSFERASE.bitmask | GeneRole.ALANINE_TRANSFERASE.bitmask,
                GeneRole.INITIATOR.bitmask
                | GeneRole.GLYCOSYLTRANSFERASE.bitmask
                | GeneRole.NUCLEOTIDYLTRANSFERASE.bitmask
                | GeneRole.OXIDOREDUCTASE.bitmask,
            ),
            forbidden_mask=roles_to_mask(
                (
                    GeneRole.TRANSLOCON,
                    GeneRole.CAPSULE_ANCHOR,
                    GeneRole.TRANSMEMBRANE_SYNTHASE,
                    GeneRole.POLYMERASE,
                    GeneRole.COPOLYMERASE,
                    GeneRole.KINASE,
                    GeneRole.LIGASE,
                )
            ),
            cargo_mask=common_precursors_and_mods
            | roles_to_mask(
                (
                    GeneRole.ABC_PERMEASE,
                    GeneRole.ABC_ATPASE,
                    GeneRole.FLIPPASE,
                )
            ),
            description="Wall teichoic acid (WTA / tag / tar / dlt) or phosphocholine-decorated teichoic acid (lic / tarIJL) locus.",
            aliases=("wta", "teichoic", "tag_tar", "lic"),
        ),
        # 10: UNKNOWN
        ArchetypeSpec(
            name="UNKNOWN",
            category=ArchetypeCategory.OTHER,
            module_masks=(),
            forbidden_mask=0,
            cargo_mask=common_precursors_and_mods,
            description="Unclassified, incomplete, or orphan biosynthetic gene cluster.",
            aliases=("unclassified", "orphan", "other"),
        ),
    )


@lru_cache(maxsize=1)
def _get_archetype_lookup() -> dict[str, LocusArchetype]:
    """Build and cache the name/alias lookup dictionary for `LocusArchetype`."""
    specs = _get_archetype_specs()
    lookup: dict[str, LocusArchetype] = {}
    for arch in LocusArchetype:
        spec = specs[int(arch)]
        lookup[spec.name.lower()] = arch
        for alias in spec.aliases:
            lookup[alias.strip().lower()] = arch
    return lookup


@dataclass(frozen=True, slots=True)
class ArchetypeTensors:
    """Immutable container holding precomputed bitmask and role-indicator tensors for all `LocusArchetype`s.

    Encapsulates the vectorized lookup tables required by [`ArchetypeClassifier`][serogen.loci.classifier.ArchetypeClassifier]
    and [`LocusSegmenter`][serogen.loci.segmenter.LocusSegmenter].
    """

    names: np.ndarray[tuple[int], np.dtype[np.str_]]
    categories: np.ndarray[tuple[int], np.dtype[np.str_]]
    forbidden_masks: np.ndarray[tuple[int], np.dtype[np.uint32]]
    cargo_masks: np.ndarray[tuple[int], np.dtype[np.uint32]]
    required_masks: np.ndarray[tuple[int], np.dtype[np.uint32]]
    all_relevant_masks: np.ndarray[tuple[int], np.dtype[np.uint32]]
    module_masks: np.ndarray[tuple[int, int], np.dtype[np.uint32]]
    num_modules: np.ndarray[tuple[int], np.dtype[np.int32]]
    active_module_mask: np.ndarray[tuple[int, int], np.dtype[np.bool_]]
    module_indicators: np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
    forbidden_indicators: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    relevant_indicators: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    null_priors: np.ndarray[tuple[int], np.dtype[np.float64]]
    machinery_roles: np.ndarray[tuple[int], np.dtype[np.int32]]
    biosynthetic_mask: np.ndarray[tuple[int], np.dtype[np.bool_]]

    @classmethod
    def build(cls) -> Self:
        """Construct and freeze all archetype bitmask and indicator tensors from `LocusArchetype` specs."""
        specs = _get_archetype_specs()
        n_arch = len(LocusArchetype)
        n_roles = len(GeneRole)
        max_mods = max(s.num_modules for s in specs)

        names = np.array([s.name for s in specs], dtype="U16")
        categories = np.array([s.category.value for s in specs], dtype="U20")
        forbidden_masks = np.array([s.forbidden_mask for s in specs], dtype=np.uint32)
        cargo_masks = np.array([s.cargo_mask for s in specs], dtype=np.uint32)
        required_masks = np.array([s.required_mask for s in specs], dtype=np.uint32)
        all_relevant_masks = np.array([s.all_relevant_mask for s in specs], dtype=np.uint32)

        module_masks = np.zeros((n_arch, max_mods), dtype=np.uint32)
        num_modules = np.zeros(n_arch, dtype=np.int32)
        for a_idx, spec in enumerate(specs):
            num_modules[a_idx] = spec.num_modules
            for m_idx, m_mask in enumerate(spec.module_masks):
                module_masks[a_idx, m_idx] = m_mask

        module_indicators = np.zeros((n_arch, max_mods, n_roles), dtype=np.float64)
        forbidden_indicators = np.zeros((n_arch, n_roles), dtype=np.float64)
        relevant_indicators = np.zeros((n_arch, n_roles), dtype=np.float64)

        for a_idx in range(n_arch):
            f_mask = int(forbidden_masks[a_idx])
            r_mask = int(all_relevant_masks[a_idx])
            n_m = int(num_modules[a_idx])
            for r_idx in range(n_roles):
                bit = 1 << r_idx
                if f_mask & bit:
                    forbidden_indicators[a_idx, r_idx] = 1.0
                if r_mask & bit:
                    relevant_indicators[a_idx, r_idx] = 1.0
                for m_idx in range(n_m):
                    if int(module_masks[a_idx, m_idx]) & bit:
                        module_indicators[a_idx, m_idx, r_idx] = 1.0

        null_priors = np.full(n_roles, 0.005, dtype=np.float64)
        null_priors[int(GeneRole.UNKNOWN)] = 0.825
        null_priors[int(GeneRole.TRANSPOSASE)] = 0.015
        null_priors[int(GeneRole.INTEGRASE)] = 0.015
        null_priors /= null_priors.sum()

        active_module_mask = np.arange(max_mods)[None, :] < num_modules[:, None]

        machinery_roles = np.array(
            [
                int(GeneRole.FLIPPASE),
                int(GeneRole.POLYMERASE),
                int(GeneRole.COPOLYMERASE),
                int(GeneRole.KINASE),
                int(GeneRole.PHOSPHATASE),
                int(GeneRole.ABC_PERMEASE),
                int(GeneRole.ABC_ATPASE),
                int(GeneRole.LIGASE),
                int(GeneRole.TRANSLOCON),
                int(GeneRole.CAPSULE_ANCHOR),
                int(GeneRole.TRANSMEMBRANE_SYNTHASE),
                int(GeneRole.INITIATOR),
            ],
            dtype=np.int32,
        )
        biosynthetic_mask = np.array([r.is_biosynthetic for r in GeneRole], dtype=np.bool_)

        for arr in (
            names,
            categories,
            forbidden_masks,
            cargo_masks,
            required_masks,
            all_relevant_masks,
            module_masks,
            num_modules,
            active_module_mask,
            module_indicators,
            forbidden_indicators,
            relevant_indicators,
            null_priors,
            machinery_roles,
            biosynthetic_mask,
        ):
            arr.setflags(write=False)

        return cls(
            names=names,
            categories=categories,
            forbidden_masks=forbidden_masks,
            cargo_masks=cargo_masks,
            required_masks=required_masks,
            all_relevant_masks=all_relevant_masks,
            module_masks=module_masks,
            num_modules=num_modules,
            active_module_mask=active_module_mask,
            module_indicators=module_indicators,
            forbidden_indicators=forbidden_indicators,
            relevant_indicators=relevant_indicators,
            null_priors=null_priors,
            machinery_roles=machinery_roles,
            biosynthetic_mask=biosynthetic_mask,
        )


@lru_cache(maxsize=1)
def get_archetype_tensors() -> ArchetypeTensors:
    """Return the cached singleton `ArchetypeTensors` instance."""
    return ArchetypeTensors.build()
