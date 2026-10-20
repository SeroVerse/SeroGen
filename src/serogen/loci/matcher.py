"""Vectorized locus comparison, structural difference bitflags, and database novelty matching."""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntFlag
from pathlib import Path
from typing import Any, Literal, Self, cast, overload

import numpy as np
from numba import njit, prange

from serogen.core.collections import BatchedContainer, ScalarView
from serogen.core.graph.align import Alignment, CostStrategy, GraphAligner
from serogen.core.graph.models import GraphModel
from serogen.core.seq.alphabet import Alphabet
from serogen.core.seq.clustering import Clusterer
from serogen.core.seq.kmers import RandstrobeIndex
from serogen.core.seq.pairwise import PairwiseAligner
from serogen.core.seq.seq import Sequences
from serogen.genes.roles import GeneRole
from serogen.loci.archetypes import LocusArchetype
from serogen.loci.models import GeneNode, IntergenicType, Loci, Locus


# ----------------------------------------------------------------------------------------------------------------------
# MatchType Bitflag Enumeration
# ----------------------------------------------------------------------------------------------------------------------
class MatchType(IntFlag):
    """Composable bitflag enumeration describing how two biosynthetic loci differ.

    `MatchType.EXACT` (`0`) indicates that two loci share identical non-MGE gene content,
    gene order, relative transcriptional orientations, dominant strand, frameshift state,
    and mobile genetic element (MGE) state. Any combination of structural or allelic
    differences is represented as the bitwise OR of the corresponding variant flags.
    """

    EXACT = 0
    STRAND_VARIANT = 1 << 0
    MGE_VARIANT = 1 << 1
    FRAMESHIFT_VARIANT = 1 << 2
    SYNTENY_VARIANT = 1 << 3
    GENE_VARIANT = 1 << 4
    ROLE_VARIANT = 1 << 5
    ARCHETYPE_MISMATCH = 1 << 6
    FRAGMENTED_VARIANT = 1 << 7
    NOVEL_LOCUS = 1 << 8

    @property
    def is_exact(self) -> bool:
        """True if no variant flags are set (`self == MatchType.EXACT`)."""
        return self == MatchType.EXACT

    @property
    def is_novel(self) -> bool:
        """True if the comparison involves gene gain/loss, synteny rearrangement, archetype mismatch, or novel locus divergence."""
        novel_mask = (
            MatchType.GENE_VARIANT | MatchType.SYNTENY_VARIANT | MatchType.ARCHETYPE_MISMATCH | MatchType.NOVEL_LOCUS
        )
        return bool(self & novel_mask)

    @property
    def is_novel_locus(self) -> bool:
        """True if the comparison represents major locus-level divergence (`MatchType.NOVEL_LOCUS`, e.g. Jaccard < 0.50)."""
        return bool(self & MatchType.NOVEL_LOCUS)

    @property
    def flag_count(self) -> int:
        """Number of active difference bitflags set in this `MatchType`."""
        return int(self).bit_count()


BENIGN_VARIANT_MASK: MatchType = (
    MatchType.STRAND_VARIANT
    | MatchType.MGE_VARIANT
    | MatchType.FRAMESHIFT_VARIANT
    | MatchType.ROLE_VARIANT
    | MatchType.FRAGMENTED_VARIANT
)


# ----------------------------------------------------------------------------------------------------------------------
# Scalar & Batched Match Containers
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class Match(ScalarView):
    """Scalar view of a pairwise comparison between a query locus and a target locus."""

    query_index: int
    target_index: int
    match_type: MatchType
    score: float
    jaccard: float
    query_coverage: float
    target_coverage: float
    shared_genes: int
    missing_genes: int
    extra_genes: int
    mean_aa_identity: float = 1.0
    target_name: str = ""

    @property
    def is_exact(self) -> bool:
        """True if the query and target loci are an exact match (`MatchType.EXACT`)."""
        return self.match_type.is_exact

    @property
    def is_novel(self) -> bool:
        """True if the query differs from this target by gene content, synteny, or archetype."""
        return self.match_type.is_novel

    @property
    def is_novel_locus(self) -> bool:
        """True if this match exhibits major locus-level divergence (`MatchType.NOVEL_LOCUS`)."""
        return self.match_type.is_novel_locus

    @property
    def is_intra_serogroup_variant(self) -> bool:
        """True if this match is a structural variant within the same serogroup (`is_novel` and not `NOVEL_LOCUS`)."""
        return self.is_novel and not self.is_novel_locus and not bool(self.match_type & MatchType.ARCHETYPE_MISMATCH)


@dataclass(slots=True, frozen=True)
class Matches(BatchedContainer[Match, "Matches"]):
    """Vectorized Structure-of-Arrays (SoA) batch container of locus comparison results."""

    query_indices: np.ndarray[tuple[int], np.dtype[np.int32]]
    target_indices: np.ndarray[tuple[int], np.dtype[np.int32]]
    match_types: np.ndarray[tuple[int], np.dtype[np.uint32]]
    scores: np.ndarray[tuple[int], np.dtype[np.float32]]
    jaccards: np.ndarray[tuple[int], np.dtype[np.float32]]
    query_coverages: np.ndarray[tuple[int], np.dtype[np.float32]]
    target_coverages: np.ndarray[tuple[int], np.dtype[np.float32]]
    shared_genes: np.ndarray[tuple[int], np.dtype[np.int32]]
    missing_genes: np.ndarray[tuple[int], np.dtype[np.int32]]
    extra_genes: np.ndarray[tuple[int], np.dtype[np.int32]]
    mean_aa_identities: np.ndarray[tuple[int], np.dtype[np.float32]] = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    target_names: np.ndarray[tuple[int], np.dtype[np.str_]] = field(default_factory=lambda: np.empty(0, dtype="U1"))

    def __post_init__(self) -> None:
        n = len(self.scores)
        if len(self.mean_aa_identities) == 0 and n > 0:
            object.__setattr__(self, "mean_aa_identities", np.ones(n, dtype=np.float32))
        if len(self.target_names) == 0 and n > 0:
            object.__setattr__(self, "target_names", np.full(n, "", dtype="U1"))
        elif self.target_names.dtype.kind == "S":
            object.__setattr__(
                self, "target_names", np.char.decode(cast(Any, self.target_names), "utf-8", errors="replace")
            )
        elif self.target_names.dtype.kind != "U":
            object.__setattr__(self, "target_names", np.asarray(self.target_names, dtype=str))

    def __len__(self) -> int:
        return len(self.scores)

    @classmethod
    def empty(cls) -> Self:
        """Create an empty 0-length `Matches` container."""
        return cls(
            query_indices=np.empty(0, dtype=np.int32),
            target_indices=np.empty(0, dtype=np.int32),
            match_types=np.empty(0, dtype=np.uint32),
            scores=np.empty(0, dtype=np.float32),
            jaccards=np.empty(0, dtype=np.float32),
            query_coverages=np.empty(0, dtype=np.float32),
            target_coverages=np.empty(0, dtype=np.float32),
            shared_genes=np.empty(0, dtype=np.int32),
            missing_genes=np.empty(0, dtype=np.int32),
            extra_genes=np.empty(0, dtype=np.int32),
            mean_aa_identities=np.empty(0, dtype=np.float32),
            target_names=np.empty(0, dtype="U1"),
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        """Concatenate multiple `Matches` containers into a single batch."""
        non_empty = [b for b in batches if len(b) > 0]
        if not non_empty:
            return cls.empty()
        if len(non_empty) == 1:
            return non_empty[0]
        return cls(
            query_indices=np.concatenate([b.query_indices for b in non_empty]),
            target_indices=np.concatenate([b.target_indices for b in non_empty]),
            match_types=np.concatenate([b.match_types for b in non_empty]),
            scores=np.concatenate([b.scores for b in non_empty]),
            jaccards=np.concatenate([b.jaccards for b in non_empty]),
            query_coverages=np.concatenate([b.query_coverages for b in non_empty]),
            target_coverages=np.concatenate([b.target_coverages for b in non_empty]),
            shared_genes=np.concatenate([b.shared_genes for b in non_empty]),
            missing_genes=np.concatenate([b.missing_genes for b in non_empty]),
            extra_genes=np.concatenate([b.extra_genes for b in non_empty]),
            mean_aa_identities=np.concatenate([b.mean_aa_identities for b in non_empty]),
            target_names=np.concatenate([b.target_names for b in non_empty]),
        )

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> Match: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> Self: ...

    @overload
    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> Match | Self: ...

    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> Match | Self:
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Matches index {item} out of bounds [0, {len(self) - 1}]")
            return Match(
                query_index=int(self.query_indices[idx]),
                target_index=int(self.target_indices[idx]),
                match_type=MatchType(int(self.match_types[idx])),
                score=float(self.scores[idx]),
                jaccard=float(self.jaccards[idx]),
                query_coverage=float(self.query_coverages[idx]),
                target_coverage=float(self.target_coverages[idx]),
                shared_genes=int(self.shared_genes[idx]),
                missing_genes=int(self.missing_genes[idx]),
                extra_genes=int(self.extra_genes[idx]),
                mean_aa_identity=float(self.mean_aa_identities[idx]) if len(self.mean_aa_identities) > idx else 1.0,
                target_name=str(self.target_names[idx]) if len(self.target_names) > idx else "",
            )

        indices = self._normalize_indices(item)
        if len(indices) == 0:
            return self.empty()
        return type(self)(
            query_indices=self.query_indices[indices],
            target_indices=self.target_indices[indices],
            match_types=self.match_types[indices],
            scores=self.scores[indices],
            jaccards=self.jaccards[indices],
            query_coverages=self.query_coverages[indices],
            target_coverages=self.target_coverages[indices],
            shared_genes=self.shared_genes[indices],
            missing_genes=self.missing_genes[indices],
            extra_genes=self.extra_genes[indices],
            mean_aa_identities=self.mean_aa_identities[indices]
            if len(self.mean_aa_identities) == len(self.scores)
            else np.ones(len(indices), dtype=np.float32),
            target_names=self.target_names[indices]
            if len(self.target_names) == len(self.scores)
            else np.full(len(indices), "", dtype="U1"),
        )

    def _popcounts(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """Vectorized population count (number of set bits) for `self.match_types`."""
        x = self.match_types.copy()
        x = x - ((x >> np.uint32(1)) & np.uint32(0x55555555))
        x = (x & np.uint32(0x33333333)) + ((x >> np.uint32(2)) & np.uint32(0x33333333))
        x = (x + (x >> np.uint32(4))) & np.uint32(0x0F0F0F0F)
        return ((x * np.uint32(0x01010101)) >> np.uint32(24)).astype(np.int32)

    def sort(self, descending: bool = True) -> Self:
        """Sort matches by score, breaking ties by fewer `MatchType` bitflags, higher AA identity, and lower `target_index`."""
        if len(self) <= 1:
            return self
        pop = self._popcounts()
        if descending:
            order = np.lexsort((self.target_indices, -self.mean_aa_identities, pop, -self.scores))
        else:
            order = np.lexsort((-self.target_indices, self.mean_aa_identities, -pop, self.scores))
        return self[order]

    def best(self) -> Match | None:
        """Return the highest-scoring `Match` across the container (or `None` if empty)."""
        if len(self) == 0:
            return None
        pop = self._popcounts()
        best_idx = int(np.lexsort((self.target_indices, -self.mean_aa_identities, pop, -self.scores))[0])
        return self[best_idx]

    def best_per_query(self) -> Self:
        """Return a `Matches` container holding the single best target match for each unique `query_index`."""
        if len(self) == 0:
            return self.empty()
        pop = self._popcounts()
        # Sort by query_index ascending, then score descending, popcount ascending, AA identity descending, target_index ascending
        order = np.lexsort((self.target_indices, -self.mean_aa_identities, pop, -self.scores, self.query_indices))
        sorted_q = self.query_indices[order]
        first_mask = np.empty(len(order), dtype=np.bool_)
        first_mask[0] = True
        if len(order) > 1:
            first_mask[1:] = sorted_q[1:] != sorted_q[:-1]
        return self[order[first_mask]]

    def filter(
        self,
        *,
        min_score: float = 0.0,
        min_jaccard: float = 0.0,
        min_coverage: float = 0.0,
        allowed_flags: MatchType | None = None,
    ) -> Self:
        """Filter matches by minimum similarity metrics and/or permitted `MatchType` flags."""
        if len(self) == 0:
            return self
        mask = (
            (self.scores >= min_score)
            & (self.jaccards >= min_jaccard)
            & (self.query_coverages >= min_coverage)
            & (self.target_coverages >= min_coverage)
        )
        if allowed_flags is not None:
            disallowed = ~np.uint32(int(allowed_flags))
            mask &= (self.match_types & disallowed) == 0
        return self[mask]

    @property
    def is_novel_locus(self) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        """1D boolean array (`(N,)`) indicating whether each match has `MatchType.NOVEL_LOCUS` set."""
        if len(self) == 0:
            return np.empty(0, dtype=np.bool_)
        return (self.match_types & np.uint32(int(MatchType.NOVEL_LOCUS))) != 0

    @property
    def is_intra_serogroup_variant(self) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        """1D boolean array (`(N,)`) indicating intra-serogroup gene/synteny/role variants (`Jaccard >= threshold`)."""
        if len(self) == 0:
            return np.empty(0, dtype=np.bool_)
        disallowed = ~np.uint32(int(BENIGN_VARIANT_MASK))
        novel_mask = (self.match_types & disallowed) != 0
        arch_mask = (self.match_types & np.uint32(int(MatchType.ARCHETYPE_MISMATCH))) != 0
        return novel_mask & ~self.is_novel_locus & ~arch_mask

    def is_novel(
        self,
        *,
        allow_variants: MatchType = BENIGN_VARIANT_MASK,
        min_coverage: float = 1.0,
    ) -> bool:
        """Return True if no target match satisfies `allow_variants` at `>= min_coverage`."""
        if len(self) == 0:
            return True
        disallowed = ~np.uint32(int(allow_variants))
        known_hits = (
            ((self.match_types & disallowed) == 0)
            & (self.query_coverages >= min_coverage)
            & (self.target_coverages >= min_coverage)
        )
        return not bool(np.any(known_hits))


# ----------------------------------------------------------------------------------------------------------------------
# Precompiled SoA Locus Representation & Parallel Numba Matching Kernel
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class _PreparedLoci:
    """Internal flattened SoA buffers for canonicalized (+1 strand) loci ready for `_match_loci_kernel`."""

    n_loci: int
    orig_strands: np.ndarray[tuple[int], np.dtype[np.int8]]
    archetypes: np.ndarray[tuple[int], np.dtype[np.int8]]
    is_fragmented: np.ndarray[tuple[int], np.dtype[np.uint8]]
    # Non-MGE (core/cargo) gene arrays in canonical 5'->3' order
    core_offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    core_keys: np.ndarray[tuple[int], np.dtype[np.int32]]
    core_roles: np.ndarray[tuple[int], np.dtype[np.int32]]
    core_fs: np.ndarray[tuple[int], np.dtype[np.uint8]]
    core_next_orient: np.ndarray[tuple[int], np.dtype[np.int8]]
    core_gidx1: np.ndarray[tuple[int], np.dtype[np.int32]]
    core_gidx2: np.ndarray[tuple[int], np.dtype[np.int32]]
    # MGE signature arrays (relative position and key/role for each MGE in canonical 5'->3' order)
    mge_offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    mge_pos: np.ndarray[tuple[int], np.dtype[np.int32]]
    mge_keys: np.ndarray[tuple[int], np.dtype[np.int32]]


def _prepare_loci_arrays(
    loci: Loci,
    *,
    use_clusters: bool,
    unclustered_base: int = -1,
) -> _PreparedLoci:
    """Convert a `Loci` batch into canonicalized (+1 strand) core and MGE arrays for Numba evaluation."""
    n_loci = len(loci)
    if n_loci == 0:
        z_i32 = np.zeros(1, dtype=np.int32)
        e_i32 = np.empty(0, dtype=np.int32)
        return _PreparedLoci(
            n_loci=0,
            orig_strands=np.empty(0, dtype=np.int8),
            archetypes=np.empty(0, dtype=np.int8),
            is_fragmented=np.empty(0, dtype=np.uint8),
            core_offsets=z_i32,
            core_keys=e_i32,
            core_roles=e_i32,
            core_fs=np.empty(0, dtype=np.uint8),
            core_next_orient=np.empty(0, dtype=np.int8),
            core_gidx1=e_i32,
            core_gidx2=e_i32,
            mge_offsets=z_i32.copy(),
            mge_pos=e_i32,
            mge_keys=e_i32,
        )

    orig_strands = loci.intervals.strands.copy()
    archetypes = loci.archetypes.copy()
    is_fragmented = loci.is_fragmented.astype(np.uint8)
    canon = loci.canonicalize_strands()
    healed = canon.heal_mobile_elements(remove_nodes=True)

    # Map each node in `canon` (and non-MGE node in `healed`) back to its original flat node index in `loci`
    canon_orig_idx = np.empty(canon.total_nodes, dtype=np.int32)
    for i in range(n_loci):
        s_o = int(loci.node_offsets[i])
        e_o = int(loci.node_offsets[i + 1])
        s_c = int(canon.node_offsets[i])
        e_c = int(canon.node_offsets[i + 1])
        if orig_strands[i] < 0:
            canon_orig_idx[s_c:e_c] = np.arange(e_o - 1, s_o - 1, -1, dtype=np.int32)
        else:
            canon_orig_idx[s_c:e_c] = np.arange(s_o, e_o, dtype=np.int32)

    canon_roles = GeneNode.ROLE_ID.extract_array(canon.node_types).astype(np.int32)
    canon_cids = GeneNode.CLUSTER_ID.extract_array(canon.node_types).astype(np.int32)
    tnp_id = int(GeneRole.TRANSPOSASE)
    int_id = int(GeneRole.INTEGRASE)
    non_mge_mask = (canon_roles != tnp_id) & (canon_roles != int_id)
    raw_gidx = canon_orig_idx[non_mge_mask]

    # 1. Extract non-MGE (core/cargo) genes and healed intergenic orientations from `healed`
    raw_offsets = healed.node_offsets.astype(np.int32)
    tot_raw = int(raw_offsets[-1])
    if len(raw_gidx) != tot_raw:
        raw_gidx = np.arange(tot_raw, dtype=np.int32)
    raw_roles = GeneNode.ROLE_ID.extract_array(healed.node_types).astype(np.int32)
    raw_fs = GeneNode.IS_FRAMESHIFT.extract_array(healed.node_types).astype(np.uint8)
    raw_cids = GeneNode.CLUSTER_ID.extract_array(healed.node_types).astype(np.int32)
    orig_partials = (
        loci.genes.is_partial
        if len(loci.genes.is_partial) == loci.total_nodes
        else np.zeros(loci.total_nodes, dtype=np.bool_)
    )
    raw_partials = (
        orig_partials[raw_gidx]
        if (tot_raw > 0 and len(orig_partials) > int(np.max(raw_gidx)))
        else np.zeros(tot_raw, dtype=np.bool_)
    )

    if use_clusters:
        raw_keys = raw_cids.copy()
        zero_mask = raw_keys == 0
        if np.any(zero_mask):
            n_zeros = int(np.sum(zero_mask))
            raw_keys[zero_mask] = np.arange(
                unclustered_base,
                unclustered_base - n_zeros,
                -1,
                dtype=np.int32,
            )
    else:
        raw_keys = raw_roles.copy()

    # For each core gene k, store the IntergenicOrientation (0..2) and whether edge k -> k+1 is a contig break
    raw_next_orient = np.full(tot_raw, -1, dtype=np.int8)
    raw_next_is_break = np.zeros(tot_raw, dtype=np.bool_)
    if len(healed.edge_sources) > 0:
        e_orients = IntergenicType.ORIENTATION.extract_array(healed.edge_types).astype(np.int8)
        e_breaks = IntergenicType.GAP_BIN.extract_array(healed.edge_types) == 14
        raw_next_orient[healed.edge_sources] = e_orients
        raw_next_is_break[healed.edge_sources] = e_breaks

    # Collapse adjacent half-genes split across a contig break (`raw_next_is_break`) that share the same
    # cluster_id (or where one half is clustered and the adjacent partial contig-edge half has the same role)
    if np.any(raw_next_is_break):
        core_offsets = np.zeros(n_loci + 1, dtype=np.int32)
        keep_keys: list[int] = []
        keep_roles: list[int] = []
        keep_fs: list[int] = []
        keep_orient: list[int] = []
        keep_gidx1: list[int] = []
        keep_gidx2: list[int] = []
        for i in range(n_loci):
            s = int(raw_offsets[i])
            e = int(raw_offsets[i + 1])
            k = s
            while k < e:
                same_break_cluster = (
                    k + 1 < e and bool(raw_next_is_break[k]) and raw_keys[k] > 0 and raw_keys[k] == raw_keys[k + 1]
                )
                asym_break_split = (
                    use_clusters
                    and k + 1 < e
                    and bool(raw_next_is_break[k])
                    and raw_roles[k] == raw_roles[k + 1]
                    and (raw_keys[k] > 0 or raw_keys[k + 1] > 0)
                    and (raw_keys[k] < 0 or raw_keys[k + 1] < 0)
                    and (bool(raw_partials[k]) or bool(raw_partials[k + 1]) or bool(raw_fs[k]) or bool(raw_fs[k + 1]))
                )
                if same_break_cluster or asym_break_split:
                    keep_keys.append(int(max(raw_keys[k], raw_keys[k + 1])))
                    keep_roles.append(int(raw_roles[k]))
                    # Clear artificial frameshift flags caused by the assembly contig break
                    keep_fs.append(0)
                    keep_orient.append(int(raw_next_orient[k + 1]))
                    keep_gidx1.append(int(raw_gidx[k]))
                    keep_gidx2.append(int(raw_gidx[k + 1]))
                    k += 2
                else:
                    # Also clear artificial frameshift flag on a single contig-edge partial gene if it is not a true frameshift
                    is_break_adj = bool(raw_next_is_break[k]) or (k > s and bool(raw_next_is_break[k - 1]))
                    fs_val = 0 if (is_break_adj and bool(raw_partials[k])) else int(raw_fs[k])
                    keep_keys.append(int(raw_keys[k]))
                    keep_roles.append(int(raw_roles[k]))
                    keep_fs.append(fs_val)
                    keep_orient.append(int(raw_next_orient[k]))
                    keep_gidx1.append(int(raw_gidx[k]))
                    keep_gidx2.append(-1)
                    k += 1
            core_offsets[i + 1] = len(keep_keys)
        core_keys = np.array(keep_keys, dtype=np.int32)
        core_roles = np.array(keep_roles, dtype=np.int32)
        core_fs = np.array(keep_fs, dtype=np.uint8)
        core_next_orient = np.array(keep_orient, dtype=np.int8)
        core_gidx1 = np.array(keep_gidx1, dtype=np.int32)
        core_gidx2 = np.array(keep_gidx2, dtype=np.int32)
    else:
        core_offsets = raw_offsets
        core_keys = raw_keys
        core_roles = raw_roles
        core_fs = raw_fs
        core_next_orient = raw_next_orient
        core_gidx1 = raw_gidx
        core_gidx2 = np.full(tot_raw, -1, dtype=np.int32)

    # 2. Extract MGE positions and keys from `canon` (before MGE excision)
    mge_offsets = np.zeros(n_loci + 1, dtype=np.int32)
    mge_pos_list: list[int] = []
    mge_key_list: list[int] = []

    for i in range(n_loci):
        s = int(canon.node_offsets[i])
        e = int(canon.node_offsets[i + 1])
        core_rank = 0
        for k in range(s, e):
            r = int(canon_roles[k])
            if r == tnp_id or r == int_id:
                mge_pos_list.append(core_rank)
                cid = int(canon_cids[k])
                mge_key_list.append(cid if (use_clusters and cid > 0) else r)
            else:
                core_rank += 1
        mge_offsets[i + 1] = len(mge_pos_list)

    return _PreparedLoci(
        n_loci=n_loci,
        orig_strands=orig_strands,
        archetypes=archetypes,
        is_fragmented=is_fragmented,
        core_offsets=core_offsets,
        core_keys=core_keys,
        core_roles=core_roles,
        core_fs=core_fs,
        core_next_orient=core_next_orient,
        core_gidx1=core_gidx1,
        core_gidx2=core_gidx2,
        mge_offsets=mge_offsets,
        mge_pos=np.array(mge_pos_list, dtype=np.int32) if mge_pos_list else np.empty(0, dtype=np.int32),
        mge_keys=np.array(mge_key_list, dtype=np.int32) if mge_key_list else np.empty(0, dtype=np.int32),
    )


@njit(cache=True, nogil=True, parallel=True)
def _match_loci_kernel(
    n_q: int,
    q_strands: np.ndarray,
    q_archs: np.ndarray,
    q_frag: np.ndarray,
    q_core_offs: np.ndarray,
    q_core_keys: np.ndarray,
    q_core_roles: np.ndarray,
    q_core_fs: np.ndarray,
    q_core_orient: np.ndarray,
    q_core_gidx1: np.ndarray,
    q_core_gidx2: np.ndarray,
    q_mge_offs: np.ndarray,
    q_mge_pos: np.ndarray,
    q_mge_keys: np.ndarray,
    n_t: int,
    t_strands: np.ndarray,
    t_archs: np.ndarray,
    t_frag: np.ndarray,
    t_core_offs: np.ndarray,
    t_core_keys: np.ndarray,
    t_core_roles: np.ndarray,
    t_core_fs: np.ndarray,
    t_core_orient: np.ndarray,
    t_core_gidx1: np.ndarray,
    t_core_gidx2: np.ndarray,
    t_mge_offs: np.ndarray,
    t_mge_pos: np.ndarray,
    t_mge_keys: np.ndarray,
    pident_mat: np.ndarray,
    has_pident: bool,
    unknown_arch: int,
    novel_jaccard_threshold: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Evaluate all `(query_i, target_j)` locus pairs in parallel."""
    total_pairs = n_q * n_t
    out_q = np.empty(total_pairs, dtype=np.int32)
    out_t = np.empty(total_pairs, dtype=np.int32)
    out_flags = np.empty(total_pairs, dtype=np.uint32)
    out_scores = np.empty(total_pairs, dtype=np.float32)
    out_jaccards = np.empty(total_pairs, dtype=np.float32)
    out_q_cov = np.empty(total_pairs, dtype=np.float32)
    out_t_cov = np.empty(total_pairs, dtype=np.float32)
    out_shared = np.empty(total_pairs, dtype=np.int32)
    out_missing = np.empty(total_pairs, dtype=np.int32)
    out_extra = np.empty(total_pairs, dtype=np.int32)
    out_aa_id = np.empty(total_pairs, dtype=np.float32)

    FLAG_STRAND = np.uint32(1 << 0)
    FLAG_MGE = np.uint32(1 << 1)
    FLAG_FS = np.uint32(1 << 2)
    FLAG_SYNTENY = np.uint32(1 << 3)
    FLAG_GENE = np.uint32(1 << 4)
    FLAG_ROLE = np.uint32(1 << 5)
    FLAG_ARCH = np.uint32(1 << 6)
    FLAG_FRAG = np.uint32(1 << 7)
    FLAG_NOVEL_LOCUS = np.uint32(1 << 8)
    novel_thresh = np.float32(novel_jaccard_threshold)

    for pair_idx in prange(total_pairs):
        qi = pair_idx // n_t
        ti = pair_idx % n_t

        out_q[pair_idx] = qi
        out_t[pair_idx] = ti
        flags = np.uint32(0)

        # 1. Archetype compatibility
        qa = int(q_archs[qi])
        ta = int(t_archs[ti])
        if qa != unknown_arch and ta != unknown_arch and qa != ta:
            flags |= FLAG_ARCH

        # 2. Dominant strand & assembly fragmentation comparison
        if q_strands[qi] != t_strands[ti]:
            flags |= FLAG_STRAND
        if q_frag[qi] != t_frag[ti]:
            flags |= FLAG_FRAG

        # 3. Mobile Genetic Element (MGE) comparison
        qm_s, qm_e = int(q_mge_offs[qi]), int(q_mge_offs[qi + 1])
        tm_s, tm_e = int(t_mge_offs[ti]), int(t_mge_offs[ti + 1])
        qm_len = qm_e - qm_s
        tm_len = tm_e - tm_s
        if qm_len != tm_len:
            flags |= FLAG_MGE
        else:
            for m in range(qm_len):
                if q_mge_pos[qm_s + m] != t_mge_pos[tm_s + m] or q_mge_keys[qm_s + m] != t_mge_keys[tm_s + m]:
                    flags |= FLAG_MGE
                    break

        # 4. Core / cargo non-MGE gene multiset matching
        qc_s, qc_e = int(q_core_offs[qi]), int(q_core_offs[qi + 1])
        tc_s, tc_e = int(t_core_offs[ti]), int(t_core_offs[ti + 1])
        len_q = qc_e - qc_s
        len_t = tc_e - tc_s

        t_used = np.zeros(len_t, dtype=np.uint8)
        q_match_t = np.full(len_q, -1, dtype=np.int32)
        shared = 0

        for a in range(len_q):
            k_q = q_core_keys[qc_s + a]
            best_b = -1
            best_pid = np.float32(-1.0)
            for b in range(len_t):
                if t_used[b] == 0 and t_core_keys[tc_s + b] == k_q:
                    if not has_pident:
                        best_b = b
                        break
                    qg1 = int(q_core_gidx1[qc_s + a])
                    qg2 = int(q_core_gidx2[qc_s + a])
                    tg1 = int(t_core_gidx1[tc_s + b])
                    tg2 = int(t_core_gidx2[tc_s + b])
                    pid = pident_mat[qg1, tg1]
                    if qg2 >= 0:
                        p2 = pident_mat[qg2, tg1]
                        if p2 > pid:
                            pid = p2
                    if tg2 >= 0:
                        p3 = pident_mat[qg1, tg2]
                        if p3 > pid:
                            pid = p3
                        if qg2 >= 0:
                            p4 = pident_mat[qg2, tg2]
                            if p4 > pid:
                                pid = p4
                    if pid > best_pid:
                        best_pid = pid
                        best_b = b
            if best_b != -1:
                t_used[best_b] = 1
                q_match_t[a] = best_b
                shared += 1

        missing = len_t - shared
        extra = len_q - shared
        union = len_q + len_t - shared

        jaccard = np.float32(shared / union) if union > 0 else np.float32(1.0)
        q_cov = np.float32(shared / len_q) if len_q > 0 else np.float32(1.0)
        t_cov = np.float32(shared / len_t) if len_t > 0 else np.float32(1.0)

        if missing > 0 or extra > 0:
            flags |= FLAG_GENE
            if jaccard < novel_thresh:
                flags |= FLAG_NOVEL_LOCUS
        if (flags & FLAG_ARCH) != 0 and jaccard < novel_thresh:
            flags |= FLAG_NOVEL_LOCUS

        # 5. Frameshift, Role, Synteny / Orientation, and AA identity across shared genes
        prev_a = -1
        prev_b = -1
        collinear_steps = 0
        total_steps = 0
        aa_sum = np.float32(0.0)

        for a in range(len_q):
            b = int(q_match_t[a])
            if b == -1:
                continue

            if has_pident:
                qg1 = int(q_core_gidx1[qc_s + a])
                qg2 = int(q_core_gidx2[qc_s + a])
                tg1 = int(t_core_gidx1[tc_s + b])
                tg2 = int(t_core_gidx2[tc_s + b])
                pid = pident_mat[qg1, tg1]
                if qg2 >= 0:
                    p2 = pident_mat[qg2, tg1]
                    if p2 > pid:
                        pid = p2
                if tg2 >= 0:
                    p3 = pident_mat[qg1, tg2]
                    if p3 > pid:
                        pid = p3
                    if qg2 >= 0:
                        p4 = pident_mat[qg2, tg2]
                        if p4 > pid:
                            pid = p4
                if pid <= np.float32(0.0):
                    pid = np.float32(0.85)
                aa_sum += pid
            else:
                aa_sum += np.float32(1.0)

            if q_core_fs[qc_s + a] != t_core_fs[tc_s + b]:
                flags |= FLAG_FS
            if q_core_roles[qc_s + a] != t_core_roles[tc_s + b]:
                flags |= FLAG_ROLE

            if prev_b != -1:
                total_steps += 1
                is_step_collinear = True
                if b <= prev_b:
                    flags |= FLAG_SYNTENY
                    is_step_collinear = False
                elif (a == prev_a + 1) and (b == prev_b + 1):
                    # Both pairs are immediately adjacent in the healed loci; check relative orientation
                    if q_core_orient[qc_s + prev_a] != t_core_orient[tc_s + prev_b]:
                        flags |= FLAG_SYNTENY
                        is_step_collinear = False
                if is_step_collinear:
                    collinear_steps += 1

            prev_a = a
            prev_b = b

        mean_aa_id = np.float32(aa_sum / shared) if shared > 0 else np.float32(0.0)
        allelic_deduction = np.float32(0.02) * max(np.float32(0.0), np.float32(1.0) - mean_aa_id)

        # Check if a pure whole-locus inversion on same-strand inputs explains a reversed synteny match
        if (flags & FLAG_SYNTENY) and shared >= 2 and collinear_steps == 0:
            # Check if matched target indices are strictly descending with matching reversed orientations
            rev_ok = True
            prev_b_rev = len_t + 1
            for a in range(len_q):
                b = int(q_match_t[a])
                if b == -1:
                    continue
                if b >= prev_b_rev:
                    rev_ok = False
                    break
                prev_b_rev = b
            if rev_ok and missing == 0 and extra == 0:
                # Whole-locus reverse complement when dominant strand tie-broke opposite ways
                flags = (flags & ~FLAG_SYNTENY) | FLAG_STRAND
                collinear_steps = total_steps

        # 6. Composite similarity score in [0.0, 1.0]
        if flags == 0:
            score = np.float32(1.0) - allelic_deduction
        elif (flags & (FLAG_GENE | FLAG_SYNTENY | FLAG_ARCH)) == 0:
            # Only benign variants (STRAND, FRAGMENTED, MGE, FRAMESHIFT, ROLE)
            deduction = allelic_deduction
            if flags & FLAG_STRAND:
                deduction += np.float32(0.001)
            if flags & FLAG_FRAG:
                deduction += np.float32(0.001)
            if flags & FLAG_FS:
                deduction += np.float32(0.002)
            if flags & FLAG_ROLE:
                deduction += np.float32(0.003)
            if flags & FLAG_MGE:
                deduction += np.float32(0.005)
            score = np.float32(1.0) - deduction
        else:
            dice = (2.0 * shared) / (len_q + len_t) if (len_q + len_t) > 0 else 1.0
            base = 0.5 * float(jaccard) + 0.5 * dice
            if flags & FLAG_SYNTENY:
                col_frac = (collinear_steps / total_steps) if total_steps > 0 else 0.0
                base *= 0.85 + 0.13 * col_frac
            else:
                base *= 0.98
            if flags & FLAG_ARCH:
                base *= 0.50
            if flags & FLAG_MGE:
                base -= 0.002
            if flags & FLAG_FS:
                base -= 0.001
            if flags & FLAG_FRAG:
                base -= 0.0005
            if flags & FLAG_STRAND:
                base -= 0.0005
            base -= float(allelic_deduction)
            score = np.float32(max(0.0, min(0.985, base)))

        out_flags[pair_idx] = flags
        out_scores[pair_idx] = score
        out_jaccards[pair_idx] = jaccard
        out_q_cov[pair_idx] = q_cov
        out_t_cov[pair_idx] = t_cov
        out_shared[pair_idx] = shared
        out_missing[pair_idx] = missing
        out_extra[pair_idx] = extra
        out_aa_id[pair_idx] = mean_aa_id

    return (
        out_q,
        out_t,
        out_flags,
        out_scores,
        out_jaccards,
        out_q_cov,
        out_t_cov,
        out_shared,
        out_missing,
        out_extra,
        out_aa_id,
    )


# ----------------------------------------------------------------------------------------------------------------------
# Cluster-Aware Cost Strategy for Detailed Bipartite Graph Alignment
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LocusNodeCostStrategy(CostStrategy):
    """Semantic node compatibility cost model for aligning two `Locus` graphs via `GraphAligner`.

    Assigns `0.0` cost for identical `(ROLE_ID, CLUSTER_ID)`, a small penalty (`0.15`) for
    same `CLUSTER_ID` with different `ROLE_ID` or frameshift state, a moderate cost (`1.5`)
    for same `ROLE_ID` with different `CLUSTER_ID`, and `np.inf` for incompatible node types
    (e.g., matching a primary `GeneNode` to an `IntergenicType` edge node).
    """

    def compute_cost(
        self,
        labels_a: np.ndarray[tuple[int], np.dtype[Any]],
        labels_b: np.ndarray[tuple[int], np.dtype[Any]],
        penalty: float = 1e6,
        graph_a: GraphModel | None = None,
        graph_b: GraphModel | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        if len(labels_a) == 0 or len(labels_b) == 0:
            return np.empty((len(labels_a), len(labels_b)), dtype=np.float64)

        la = labels_a.astype(np.uint32)
        lb = labels_b.astype(np.uint32)

        edge_a = GeneNode.TAG_EDGE.extract_array(la)[:, None]
        edge_b = GeneNode.TAG_EDGE.extract_array(lb)[None, :]

        # Primary vs Edge nodes cannot match each other
        incompatible = edge_a != edge_b
        is_edge_pair = (edge_a == 1) & (edge_b == 1)

        # Edge-to-edge cost: 0.0 if identical orientation, 0.5 otherwise
        orient_a = IntergenicType.ORIENTATION.extract_array(la)[:, None]
        orient_b = IntergenicType.ORIENTATION.extract_array(lb)[None, :]
        edge_cost = np.where(orient_a == orient_b, 0.0, 0.5)

        # Gene-to-gene cost
        cid_a = GeneNode.CLUSTER_ID.extract_array(la)[:, None]
        cid_b = GeneNode.CLUSTER_ID.extract_array(lb)[None, :]
        role_a = GeneNode.ROLE_ID.extract_array(la)[:, None]
        role_b = GeneNode.ROLE_ID.extract_array(lb)[None, :]
        fs_a = GeneNode.IS_FRAMESHIFT.extract_array(la)[:, None]
        fs_b = GeneNode.IS_FRAMESHIFT.extract_array(lb)[None, :]

        both_clustered = (cid_a > 0) & (cid_b > 0)
        same_cid = both_clustered & (cid_a == cid_b)
        same_role = role_a == role_b
        same_fs = fs_a == fs_b

        gene_cost = np.full((len(la), len(lb)), np.inf, dtype=np.float64)
        # Same homology cluster
        gene_cost = np.where(same_cid & same_role & same_fs, 0.0, gene_cost)
        gene_cost = np.where(same_cid & same_role & ~same_fs, 0.05, gene_cost)
        gene_cost = np.where(same_cid & ~same_role, 0.15, gene_cost)
        # Unclustered fallback or same-role paralog/ortholog candidate
        unclustered_pair = (cid_a == 0) & (cid_b == 0)
        gene_cost = np.where(unclustered_pair & same_role & same_fs, 0.0, gene_cost)
        gene_cost = np.where(unclustered_pair & same_role & ~same_fs, 0.05, gene_cost)
        gene_cost = np.where(~same_cid & ~unclustered_pair & same_role, 1.5, gene_cost)

        out = np.where(is_edge_pair, edge_cost, gene_cost)
        return np.where(incompatible, np.inf, out)


# ----------------------------------------------------------------------------------------------------------------------
# High-Performance Vectorized LocusMatcher
# ----------------------------------------------------------------------------------------------------------------------
class LocusMatcher:
    """Vectorized 1-to-many and batch-to-many biosynthetic locus matcher and novelty detector.

    Indexes a reference database of target [`Loci`][serogen.loci.models.Loci] (automatically clustering
    target proteins and building a [`RandstrobeIndex`][serogen.core.seq.kmers.RandstrobeIndex] over cluster
    centroids when protein translations are present) and compares query [`Locus`][serogen.loci.models.Locus]
    or [`Loci`][serogen.loci.models.Loci] instances in parallel via `_match_loci_kernel`.
    """

    def __init__(
        self,
        targets: Loci | Locus | Iterable[Locus],
        *,
        min_identity: float = 80.0,
        min_coverage: float = 0.8,
        min_bitscore: float = 0.0,
        min_score: int = 2,
        cluster_mode: Literal["cdhit", "mcl", "greedy_cover", "upgma", "connected_components"] = "cdhit",
        inflation: float = 2.0,
        novel_jaccard_threshold: float | Literal["auto"] = 0.50,
        serogroups: Sequence[str] | np.ndarray[tuple[int], np.dtype[Any]] | None = None,
        k: int = 10,
        s: int = 5,
        aligner_k: int = 20,
    ) -> None:
        if isinstance(targets, Locus):
            target_batch = Loci.from_list([targets])
        elif isinstance(targets, Loci):
            target_batch = targets
        else:
            target_batch = Loci.from_list(list(targets))

        eff_min_identity = 25.0 if (cluster_mode == "mcl" and min_identity == 80.0) else float(min_identity)
        eff_min_bitscore = 40.0 if (cluster_mode == "mcl" and min_bitscore <= 0.0) else float(min_bitscore)

        self.min_identity = eff_min_identity
        self.min_coverage = min_coverage
        self.min_bitscore = eff_min_bitscore
        self.min_score = min_score
        self.cluster_mode: str = cluster_mode
        self.inflation = float(inflation)
        self.novel_jaccard_threshold: float = (
            0.50 if isinstance(novel_jaccard_threshold, str) else float(novel_jaccard_threshold)
        )
        self.k = k
        self.s = s
        self.aligner_k = aligner_k
        self.aligner = PairwiseAligner(k=aligner_k)

        # Determine whether targets have protein sequences and/or existing cluster IDs
        has_target_seqs = (
            target_batch.total_nodes > 0
            and len(target_batch.genes) == target_batch.total_nodes
            and len(target_batch.genes.translations) == target_batch.total_nodes
        )
        has_target_cids = target_batch.total_nodes > 0 and bool(np.any(target_batch.cluster_ids > 0))

        if has_target_seqs and not has_target_cids:
            clusterer = Clusterer(
                min_identity=self.min_identity,
                min_coverage=min_coverage,
                min_bitscore=self.min_bitscore,
                inflation=self.inflation,
                min_score=min_score,
                k=k,
                s=s,
                aligner_k=aligner_k,
                mode=cluster_mode,
            )
            target_batch = target_batch.cluster_genes(clusterer=clusterer)
            has_target_cids = bool(np.any(target_batch.cluster_ids > 0))

        self.targets: Loci = target_batch
        self.use_clusters: bool = has_target_cids
        self._has_target_seqs: bool = has_target_seqs
        self._target_partial_mask: np.ndarray[tuple[int], np.dtype[np.bool_]] = self._partial_node_mask(self.targets)

        sensitive_mode = self.cluster_mode == "mcl" or self.min_identity < 60.0 or self.min_bitscore > 0.0
        k_eff = min(k, 5) if sensitive_mode else k
        s_eff = min(s, 2) if sensitive_mode else s

        if has_target_seqs:
            self._target_seqs = self.targets.genes.translations
            self._target_index: RandstrobeIndex | None = RandstrobeIndex.build(
                self._target_seqs,
                k=k_eff,
                s=s_eff,
                sort_by_hash=True,
            )
        else:
            self._target_seqs = Sequences.empty()
            self._target_index = None

        # Build a RandstrobeIndex over the longest representative protein per target cluster_id
        if has_target_seqs and has_target_cids:
            t_proteins = self.targets.genes.translations
            t_cids = self.targets.cluster_ids
            unique_cids = np.unique(t_cids[t_cids > 0])
            rep_indices = np.empty(len(unique_cids), dtype=np.int32)
            lengths = t_proteins.lengths
            for idx, cid in enumerate(unique_cids):
                members = np.nonzero(t_cids == cid)[0]
                best_member = members[int(np.argmax(lengths[members]))]
                rep_indices[idx] = int(best_member)

            self._centroid_cids = unique_cids.astype(np.int32)
            self._centroid_seqs = t_proteins[rep_indices]
            self._centroid_partial_mask = self._target_partial_mask[rep_indices]
            self._centroid_index: RandstrobeIndex | None = RandstrobeIndex.build(
                self._centroid_seqs,
                k=k_eff,
                s=s_eff,
                sort_by_hash=True,
            )
        else:
            self._centroid_cids = np.empty(0, dtype=np.int32)
            self._centroid_seqs = Sequences.empty()
            self._centroid_partial_mask = np.empty(0, dtype=np.bool_)
            self._centroid_index = None

        self._prepared_targets = _prepare_loci_arrays(
            self.targets,
            use_clusters=self.use_clusters,
            unclustered_base=-1,
        )

        if novel_jaccard_threshold == "auto":
            self.novel_jaccard_threshold = self.infer_novel_threshold(serogroups=serogroups)

    @staticmethod
    def _extract_serogroups(
        names: np.ndarray[tuple[int], np.dtype[Any]],
        serogroups: Sequence[str] | np.ndarray[tuple[int], np.dtype[Any]] | None = None,
    ) -> np.ndarray[tuple[int], np.dtype[np.str_]]:
        """Resolve serogroup labels from explicit `serogroups` or leading numeric prefixes of serotype `names`."""
        n = len(names)
        if serogroups is not None:
            sg_arr = np.asarray([str(x).strip() for x in serogroups], dtype=np.str_)
            if len(sg_arr) == n and bool(np.any(sg_arr != "")):
                return sg_arr

        parsed: list[str] = []
        for raw_name in names:
            s = str(raw_name).strip()
            m = re.match(r"^(\d+)[A-Za-z]+$", s)
            parsed.append(m.group(1) if m is not None else s)
        return np.asarray(parsed, dtype=np.str_)

    def infer_novel_threshold(
        self,
        serogroups: Sequence[str] | np.ndarray[tuple[int], np.dtype[Any]] | None = None,
    ) -> float:
        """Infer the optimal `novel_jaccard_threshold` separating intra-serogroup variants from novel serogroups.

        Uses supervised Youden-$J$ ($\text{TPR} - \text{FPR}$) optimization over leave-one-out
        nearest intra-serogroup vs. nearest inter-serogroup Jaccard similarities when multi-member
        serogroups are available, or unsupervised 1D Otsu bimodal thresholding on the off-diagonal
        pairwise Jaccard distribution otherwise.
        """
        m = len(self.targets)
        if m < 2:
            return 0.50

        # Evaluate all-vs-all Jaccard matrix across target loci
        prep = self._prepared_targets
        _, _, _, _, j_flat, _, _, _, _, _, _ = _match_loci_kernel(
            prep.n_loci,
            prep.orig_strands,
            prep.archetypes,
            prep.is_fragmented,
            prep.core_offsets,
            prep.core_keys,
            prep.core_roles,
            prep.core_fs,
            prep.core_next_orient,
            prep.core_gidx1,
            prep.core_gidx2,
            prep.mge_offsets,
            prep.mge_pos,
            prep.mge_keys,
            prep.n_loci,
            prep.orig_strands,
            prep.archetypes,
            prep.is_fragmented,
            prep.core_offsets,
            prep.core_keys,
            prep.core_roles,
            prep.core_fs,
            prep.core_next_orient,
            prep.core_gidx1,
            prep.core_gidx2,
            prep.mge_offsets,
            prep.mge_pos,
            prep.mge_keys,
            np.empty((0, 0), dtype=np.float32),
            False,
            int(LocusArchetype.UNKNOWN),
            0.50,
        )
        j_mat = j_flat.reshape(m, m).astype(np.float64)

        sg_labels = self._extract_serogroups(self.targets.names, serogroups=serogroups)
        non_empty_sg = sg_labels != ""
        unique_sg, sg_counts = np.unique(sg_labels[non_empty_sg], return_counts=True)

        # 1. Supervised Calibration (when >= 2 distinct serogroups and >= 1 multi-member serogroup exist)
        if len(unique_sg) >= 2 and bool(np.any(sg_counts >= 2)):
            pos_list: list[float] = []
            neg_list: list[float] = []
            for i in range(m):
                sg_i = sg_labels[i]
                if not sg_i:
                    continue
                same_mask = (sg_labels == sg_i) & (np.arange(m) != i)
                diff_mask = (sg_labels != sg_i) & (sg_labels != "")
                if np.any(same_mask):
                    pos_list.append(float(np.max(j_mat[i, same_mask])))
                if np.any(diff_mask):
                    neg_list.append(float(np.max(j_mat[i, diff_mask])))

            if pos_list and neg_list:
                pos_arr = np.asarray(pos_list, dtype=np.float64)
                neg_arr = np.asarray(neg_list, dtype=np.float64)
                vals = np.unique(np.concatenate([pos_arr, neg_arr]))
                if len(vals) == 1:
                    return float(np.clip(vals[0], 0.15, 0.85))
                candidates = 0.5 * (vals[:-1] + vals[1:])
                best_tau = 0.50
                best_youden = -2.0
                for tau in candidates:
                    tpr = float(np.mean(pos_arr >= tau))
                    tnr = float(np.mean(neg_arr < tau))
                    youden = tpr + tnr - 1.0
                    if youden > best_youden + 1e-9:
                        best_youden = youden
                        best_tau = float(tau)
                return float(np.clip(best_tau, 0.15, 0.85))

        # 2. Unsupervised 1D Otsu Bimodal Thresholding on off-diagonal pairwise Jaccard values
        tri_u, tri_v = np.triu_indices(m, k=1)
        off_diag = j_mat[tri_u, tri_v]
        u_vals = np.unique(off_diag)
        if len(u_vals) < 2 or float(u_vals[-1] - u_vals[0]) < 0.05:
            return 0.50

        candidates = 0.5 * (u_vals[:-1] + u_vals[1:])
        best_tau = 0.50
        best_var = -1.0
        for tau in candidates:
            low = off_diag[off_diag < tau]
            high = off_diag[off_diag >= tau]
            if len(low) == 0 or len(high) == 0:
                continue
            w0 = len(low) / len(off_diag)
            w1 = len(high) / len(off_diag)
            mu_diff = float(np.mean(high) - np.mean(low))
            between_var = w0 * w1 * (mu_diff * mu_diff)
            if between_var > best_var:
                best_var = between_var
                best_tau = float(tau)

        return float(np.clip(best_tau, 0.15, 0.85))

    @staticmethod
    def _partial_node_mask(loci: Loci) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        """Return a 1D boolean mask (`(total_nodes,)`) marking partial/truncated or contig-break-adjacent genes."""
        n = loci.total_nodes
        if n == 0:
            return np.empty(0, dtype=np.bool_)
        mask = GeneNode.IS_FRAMESHIFT.extract_array(loci.node_types).astype(np.bool_)
        if len(loci.genes.is_partial) == n:
            mask |= loci.genes.is_partial
        if len(loci.edge_types) > 0:
            break_edges = IntergenicType.GAP_BIN.extract_array(loci.edge_types) == 14
            if np.any(break_edges):
                mask[loci.edge_sources[break_edges]] = True
                mask[loci.edge_targets[break_edges]] = True
        return mask

    @staticmethod
    def _randstrobe_to_dict(idx: RandstrobeIndex | None, prefix: str) -> dict[str, np.ndarray]:
        """Serialize a `RandstrobeIndex` into flat NumPy arrays."""
        if idx is None:
            return {
                f"{prefix}hashes": np.empty(0, dtype=np.uint64),
                f"{prefix}seq_indices": np.empty(0, dtype=np.uint32),
                f"{prefix}positions": np.empty(0, dtype=np.uint32),
                f"{prefix}offsets": np.empty(0, dtype=np.int32),
                f"{prefix}lengths": np.empty(0, dtype=np.int32),
                f"{prefix}meta": np.array([0, 0, 10, 5, 1, 5, 0], dtype=np.int64),
            }
        return {
            f"{prefix}hashes": idx.hashes,
            f"{prefix}seq_indices": idx.seq_indices,
            f"{prefix}positions": idx.positions,
            f"{prefix}offsets": idx.offsets,
            f"{prefix}lengths": idx.lengths,
            f"{prefix}meta": np.array(
                [idx.n_seqs, int(idx.is_sorted), idx.k, idx.s, idx.w_min, idx.w_max, 1],
                dtype=np.int64,
            ),
        }

    @staticmethod
    def _randstrobe_from_dict(data: Mapping[str, Any], prefix: str) -> RandstrobeIndex | None:
        """Reconstruct a `RandstrobeIndex` from flat NumPy arrays without re-hashing."""
        if f"{prefix}meta" not in data:
            return None
        meta = data[f"{prefix}meta"]
        if len(meta) < 7 or int(meta[6]) == 0:
            return None
        return RandstrobeIndex(
            hashes=data[f"{prefix}hashes"],
            seq_indices=data[f"{prefix}seq_indices"],
            positions=data[f"{prefix}positions"],
            offsets=data[f"{prefix}offsets"],
            lengths=data[f"{prefix}lengths"],
            n_seqs=int(meta[0]),
            is_sorted=bool(int(meta[1])),
            k=int(meta[2]),
            alphabet=Alphabet.MMSEQS12,
            s=int(meta[3]),
            w_min=int(meta[4]),
            w_max=int(meta[5]),
            reduce=True,
            lut=None,
        )

    def to_npz_dict(self, prefix: str = "matcher_", include_targets: bool = False) -> dict[str, np.ndarray]:
        """Export pre-built `LocusMatcher` index arrays into a flat dictionary for zero-compute `.npz` persistence."""
        p = prefix
        out: dict[str, np.ndarray] = {
            f"{p}meta_int": np.array(
                [
                    self.min_score,
                    self.k,
                    self.s,
                    self.aligner_k,
                    int(self.use_clusters),
                    int(self._has_target_seqs),
                ],
                dtype=np.int64,
            ),
            f"{p}meta_float": np.array(
                [
                    self.min_identity,
                    self.min_coverage,
                    self.novel_jaccard_threshold,
                    self.min_bitscore,
                    self.inflation,
                ],
                dtype=np.float64,
            ),
            f"{p}cluster_mode": np.asarray([self.cluster_mode], dtype="U32"),
            f"{p}centroid_cids": self._centroid_cids,
            f"{p}centroid_seq_ids": self._centroid_seqs.ids,
            f"{p}centroid_seq_bytes": self._centroid_seqs.seqs,
            f"{p}centroid_seq_offsets": self._centroid_seqs.offsets,
            f"{p}centroid_seq_lengths": self._centroid_seqs.lengths,
        }
        out.update(self._randstrobe_to_dict(self._centroid_index, f"{p}cidx_"))
        out.update(self._randstrobe_to_dict(self._target_index, f"{p}tidx_"))
        if include_targets:
            out.update(self.targets.to_npz_dict(prefix=f"{p}locus_"))
        return out

    @classmethod
    def from_npz_dict(
        cls,
        data: Mapping[str, Any],
        targets: Loci | None = None,
        prefix: str = "matcher_",
    ) -> Self:
        """Reconstruct a pre-indexed `LocusMatcher` from `.npz` arrays without re-clustering or re-hashing."""
        p = prefix
        target_batch = targets if targets is not None else Loci.from_npz_dict(data, prefix=f"{p}locus_")
        if f"{p}meta_int" not in data:
            return cls(target_batch)

        meta_int = data[f"{p}meta_int"]
        meta_float = data[f"{p}meta_float"]

        inst = cls.__new__(cls)
        inst.min_identity = float(meta_float[0])
        inst.min_coverage = float(meta_float[1])
        inst.novel_jaccard_threshold = float(meta_float[2]) if len(meta_float) > 2 else 0.50
        inst.min_bitscore = float(meta_float[3]) if len(meta_float) > 3 else 0.0
        inst.inflation = float(meta_float[4]) if len(meta_float) > 4 else 2.0
        inst.cluster_mode = str(data[f"{p}cluster_mode"][0]) if f"{p}cluster_mode" in data else "cdhit"
        inst.min_score = int(meta_int[0])
        inst.k = int(meta_int[1])
        inst.s = int(meta_int[2])
        inst.aligner_k = int(meta_int[3])
        inst.aligner = PairwiseAligner(k=inst.aligner_k)
        inst.targets = target_batch
        inst.use_clusters = bool(int(meta_int[4]))
        inst._has_target_seqs = bool(int(meta_int[5]))
        inst._target_partial_mask = cls._partial_node_mask(inst.targets)

        if inst._has_target_seqs and len(target_batch.genes) > 0:
            inst._target_seqs = target_batch.genes.translations
            inst._target_index = cls._randstrobe_from_dict(data, f"{p}tidx_")
        else:
            inst._target_seqs = Sequences.empty()
            inst._target_index = None

        inst._centroid_cids = data[f"{p}centroid_cids"].astype(np.int32)
        if len(data[f"{p}centroid_seq_lengths"]) > 0:
            inst._centroid_seqs = Sequences(
                ids=data[f"{p}centroid_seq_ids"],
                seqs=data[f"{p}centroid_seq_bytes"],
                offsets=data[f"{p}centroid_seq_offsets"],
                lengths=data[f"{p}centroid_seq_lengths"],
            )
            inst._centroid_partial_mask = np.zeros(len(inst._centroid_cids), dtype=np.bool_)
            inst._centroid_index = cls._randstrobe_from_dict(data, f"{p}cidx_")
        else:
            inst._centroid_seqs = Sequences.empty()
            inst._centroid_partial_mask = np.empty(0, dtype=np.bool_)
            inst._centroid_index = None

        inst._prepared_targets = _prepare_loci_arrays(
            inst.targets,
            use_clusters=inst.use_clusters,
            unclustered_base=-1,
        )
        return inst

    def save(self, path: Path | str) -> Path:
        """Save this `LocusMatcher` and its indexed target `Loci` to a compressed `.npz` archive (`allow_pickle=False`)."""
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".npz")
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **cast(Any, self.to_npz_dict(prefix="matcher_", include_targets=True)))
        return p

    @classmethod
    def load(cls, path: Path | str) -> Self:
        """Load a pre-indexed `LocusMatcher` from a compressed `.npz` archive (`allow_pickle=False`)."""
        p = Path(path)
        if not p.suffix and not p.exists():
            p = p.with_suffix(".npz")
        with np.load(p, allow_pickle=False) as data:
            return cls.from_npz_dict(data, prefix="matcher_")

    def _compute_pident_matrix(
        self,
        queries: Loci,
    ) -> tuple[np.ndarray[tuple[int, int], np.dtype[np.float32]], bool]:
        """Compute coverage-weighted pairwise amino-acid identity matrix `(n_q_nodes, n_t_nodes)` in `[0.0, 1.0]`."""
        if (
            not self._has_target_seqs
            or self._target_index is None
            or queries.total_nodes == 0
            or len(queries.genes) != queries.total_nodes
            or len(queries.genes.translations) != queries.total_nodes
        ):
            return np.empty((0, 0), dtype=np.float32), False

        q_proteins = queries.genes.translations
        n_q_nodes = len(q_proteins)
        n_t_nodes = len(self._target_seqs)
        pident_mat = np.zeros((n_q_nodes, n_t_nodes), dtype=np.float32)

        q_partial_mask = self._partial_node_mask(queries)
        t_partial_mask = self._target_partial_mask
        seed_min = 1 if bool(np.any(q_partial_mask) or np.any(t_partial_mask)) else max(1, self.min_score // 2)
        seeds = self._target_index.hits(q_proteins, min_score=seed_min)
        if len(seeds) > 0:
            alignments = self.aligner.align_seeds(q_proteins, self._target_seqs, seeds)
            q_idx = seeds.query_indices
            t_idx = seeds.target_indices
            q_lens = q_proteins.lengths[q_idx]
            t_lens = self._target_seqs.lengths[t_idx]
            aligned_span = np.maximum(alignments.q_ends - alignments.q_starts, alignments.t_ends - alignments.t_starts)
            is_part = q_partial_mask[q_idx] | (
                t_partial_mask[t_idx] if len(t_partial_mask) == n_t_nodes else np.zeros(len(t_idx), dtype=np.bool_)
            )
            denom = np.where(is_part, np.minimum(q_lens, t_lens), np.maximum(q_lens, t_lens))
            cov = np.clip(aligned_span / np.maximum(denom, 1), 0.0, 1.0)
            pids = np.clip((alignments.pidents / 100.0) * cov, 0.0, 1.0).astype(np.float32)
            np.maximum.at(pident_mat, (q_idx, t_idx), pids)

        return pident_mat, True

    def assign_query_clusters(self, queries: Loci) -> Loci:
        """Assign target homology `CLUSTER_ID`s to unclustered query genes via `RandstrobeIndex` + `PairwiseAligner`."""
        if (
            not self.use_clusters
            or queries.total_nodes == 0
            or bool(np.any(queries.cluster_ids > 0))
            or self._centroid_index is None
            or len(queries.genes) != queries.total_nodes
        ):
            return queries

        q_proteins = queries.genes.translations
        if len(q_proteins) != queries.total_nodes:
            return queries

        is_partial_node = self._partial_node_mask(queries)
        sensitive_mode = self.cluster_mode == "mcl" or self.min_identity < 60.0 or self.min_bitscore > 0.0

        # In MCL / bitscore-sensitive mode, match against all target proteins (with role-consistent fallback)
        # so query genes can join multi-clade functional orthogroups.
        if sensitive_mode and self._target_index is not None and len(self._target_seqs) == self.targets.total_nodes:
            t_proteins = self._target_seqs
            t_partial_node = (
                self._target_partial_mask
                if len(self._target_partial_mask) == len(t_proteins)
                else np.zeros(len(t_proteins), dtype=np.bool_)
            )
            q_roles = queries.roles
            t_roles = self.targets.roles
            t_cids = self.targets.cluster_ids

            seeds = self._target_index.hits(q_proteins, min_score=1)
            q_chunks: list[np.ndarray[tuple[int], np.dtype[np.int32]]] = []
            t_chunks: list[np.ndarray[tuple[int], np.dtype[np.int32]]] = []
            pident_chunks: list[np.ndarray[tuple[int], np.dtype[np.float32]]] = []
            bitscore_chunks: list[np.ndarray[tuple[int], np.dtype[np.float32]]] = []
            q_span_chunks: list[np.ndarray[tuple[int], np.dtype[np.int32]]] = []
            t_span_chunks: list[np.ndarray[tuple[int], np.dtype[np.int32]]] = []

            seeded_pairs: set[int] = set()
            n_t = len(t_proteins)
            if len(seeds) > 0:
                valid_t = t_cids[seeds.target_indices] > 0
                c_seeds = seeds.filter(valid_t)
                if len(c_seeds) > 0:
                    aln = self.aligner.align_seeds(q_proteins, t_proteins, c_seeds)
                    qi = c_seeds.query_indices.astype(np.int32)
                    ti = c_seeds.target_indices.astype(np.int32)
                    r_q = q_roles[qi]
                    r_t = t_roles[ti]
                    unk = int(GeneRole.UNKNOWN)
                    compat = (r_q == r_t) | (r_q == unk) | (r_t == unk) | (aln.pidents >= 80.0)
                    if np.any(compat):
                        qi = qi[compat]
                        ti = ti[compat]
                        p_sub = aln.pidents[compat]
                        b_sub = aln.bitscores[compat]
                        qs_sub = (aln.q_ends - aln.q_starts)[compat]
                        ts_sub = (aln.t_ends - aln.t_starts)[compat]
                        q_chunks.append(qi)
                        t_chunks.append(ti)
                        pident_chunks.append(p_sub)
                        bitscore_chunks.append(b_sub)
                        q_span_chunks.append(qs_sub)
                        t_span_chunks.append(ts_sub)

                        passed_seed = (p_sub >= self.min_identity) & (b_sub >= self.min_bitscore)
                        if self.min_coverage > 0.0:
                            q_l_s = np.maximum(q_proteins.lengths[qi], 1)
                            t_l_s = np.maximum(t_proteins.lengths[ti], 1)
                            passed_seed &= ((qs_sub / q_l_s) >= self.min_coverage) & (
                                (ts_sub / t_l_s) >= self.min_coverage
                            )
                        for u_i, v_i in zip(qi[passed_seed], ti[passed_seed], strict=False):
                            seeded_pairs.add(int(u_i) * n_t + int(v_i))

            extra_q_list: list[int] = []
            extra_t_list: list[int] = []
            unk_role = int(GeneRole.UNKNOWN)
            for r_val in np.unique(q_roles):
                if r_val < 0 or r_val == unk_role:
                    continue
                q_r = np.where(q_roles == r_val)[0]
                t_r = np.where((t_roles == r_val) & (t_cids > 0))[0]
                if len(q_r) * len(t_r) <= 2000:
                    for u_idx in q_r:
                        u_int = int(u_idx)
                        for v_idx in t_r:
                            v_int = int(v_idx)
                            if (u_int * n_t + v_int) not in seeded_pairs:
                                extra_q_list.append(u_int)
                                extra_t_list.append(v_int)

            if extra_q_list:
                extra_q = np.array(extra_q_list, dtype=np.int32)
                extra_t = np.array(extra_t_list, dtype=np.int32)
                extra_aln = self.aligner(q_proteins[extra_q], t_proteins[extra_t], seeds=None)
                q_chunks.append(extra_q)
                t_chunks.append(extra_t)
                pident_chunks.append(extra_aln.pidents)
                bitscore_chunks.append(extra_aln.bitscores)
                q_span_chunks.append(extra_aln.q_ends - extra_aln.q_starts)
                t_span_chunks.append(extra_aln.t_ends - extra_aln.t_starts)

            assigned_cids = np.full(queries.total_nodes, -1, dtype=np.int32)
            if q_chunks:
                all_q = np.concatenate(q_chunks)
                all_t = np.concatenate(t_chunks)
                all_p = np.concatenate(pident_chunks)
                all_b = np.concatenate(bitscore_chunks)
                all_qs = np.concatenate(q_span_chunks)
                all_ts = np.concatenate(t_span_chunks)

                pair_part = is_partial_node[all_q] | t_partial_node[all_t]
                valid = (all_p >= self.min_identity) & (all_b >= self.min_bitscore)
                if self.min_coverage > 0.0:
                    q_l = np.maximum(q_proteins.lengths[all_q], 1)
                    t_l = np.maximum(t_proteins.lengths[all_t], 1)
                    q_c = all_qs / q_l
                    t_c = all_ts / t_l
                    u_c = np.maximum(all_qs, all_ts) / np.maximum(np.minimum(q_l, t_l), 1)
                    valid &= np.where(
                        pair_part, u_c >= self.min_coverage, (q_c >= self.min_coverage) & (t_c >= self.min_coverage)
                    )

                if np.any(valid):
                    vq = all_q[valid]
                    vt = all_t[valid]
                    vp = all_p[valid]
                    vb = all_b[valid]
                    order = np.lexsort((vp, vb, (vp >= 80.0).astype(np.int32), vq))
                    assigned_cids[vq[order]] = t_cids[vt[order]] - 1

            return queries.with_clusters(assigned_cids)

        c_partial_node = (
            self._centroid_partial_mask
            if len(self._centroid_partial_mask) == len(self._centroid_seqs)
            else np.zeros(len(self._centroid_seqs), dtype=np.bool_)
        )
        has_any_partial = bool(np.any(is_partial_node) or np.any(c_partial_node))
        seed_min_score = 1 if has_any_partial else self.min_score

        seeds = self._centroid_index.hits(q_proteins, min_score=seed_min_score)
        assigned_cids = np.full(queries.total_nodes, -1, dtype=np.int32)

        if len(seeds) > 0:
            alignments = self.aligner.align_seeds(q_proteins, self._centroid_seqs, seeds)
            pidents = alignments.pidents
            q_is_partial = is_partial_node[seeds.query_indices]
            t_is_partial = c_partial_node[seeds.target_indices]
            pair_is_partial = q_is_partial | t_is_partial

            valid = pidents >= self.min_identity
            if self.min_bitscore > 0.0:
                valid &= alignments.bitscores >= self.min_bitscore
            if has_any_partial and self.min_score > 1:
                valid &= pair_is_partial | (seeds.scores >= self.min_score)

            if self.min_coverage > 0.0:
                q_lens = q_proteins.lengths[seeds.query_indices]
                t_lens = self._centroid_seqs.lengths[seeds.target_indices]
                q_span = alignments.q_ends - alignments.q_starts
                t_span = alignments.t_ends - alignments.t_starts
                q_cov = q_span / np.maximum(q_lens, 1)
                t_cov = t_span / np.maximum(t_lens, 1)
                unidir_cov = np.maximum(q_span, t_span) / np.maximum(np.minimum(q_lens, t_lens), 1)
                both_cov = (q_cov >= self.min_coverage) & (t_cov >= self.min_coverage)
                valid &= np.where(pair_is_partial, unidir_cov >= self.min_coverage, both_cov)

            if np.any(valid):
                v_q = seeds.query_indices[valid].astype(np.int32)
                v_t = seeds.target_indices[valid].astype(np.int32)
                v_pident = pidents[valid]
                v_scores = alignments.scores[valid]

                # Sort ascending so the last entry per query index has highest pident & score
                order = np.lexsort((v_scores, v_pident, v_q))
                v_q_sorted = v_q[order]
                v_t_sorted = v_t[order]
                # Map 1-based target cluster IDs back to 0-based for `with_clusters` (which adds +1)
                assigned_cids[v_q_sorted] = self._centroid_cids[v_t_sorted] - 1

        return queries.with_clusters(assigned_cids)

    def match(
        self,
        query: Locus | Loci | Iterable[Locus],
        *,
        top_k: int | None = None,
        min_score: float = 0.0,
        sort: bool = True,
    ) -> Matches:
        """Compare one or more query loci against all indexed target loci.

        Args:
            query: A single `Locus`, a `Loci` batch, or an iterable of `Locus` objects.
            top_k: Optional maximum number of top matches to retain per query locus.
            min_score: Optional minimum similarity score filter in `[0.0, 1.0]`.
            sort: If True (default), sort returned matches by descending score per query.

        Returns:
            Matches: Vectorized SoA container of pairwise locus match results.
        """
        if isinstance(query, Locus):
            q_batch = Loci.from_list([query])
        elif isinstance(query, Loci):
            q_batch = query
        else:
            q_batch = Loci.from_list(list(query))

        if len(q_batch) == 0 or len(self.targets) == 0:
            return Matches.empty()

        q_mapped = self.assign_query_clusters(q_batch)
        use_clusters = self.use_clusters or bool(np.any(q_mapped.cluster_ids > 0))
        pident_mat, has_pident = self._compute_pident_matrix(q_mapped)

        prep_t = (
            self._prepared_targets
            if use_clusters == self.use_clusters
            else _prepare_loci_arrays(self.targets, use_clusters=use_clusters, unclustered_base=-1)
        )
        # Ensure unclustered query genes receive negative keys distinct from any unclustered target genes
        q_unclustered_base = -1 - int(self.targets.total_nodes) - 1
        prep_q = _prepare_loci_arrays(
            q_mapped,
            use_clusters=use_clusters,
            unclustered_base=q_unclustered_base,
        )

        (
            out_q,
            out_t,
            out_flags,
            out_scores,
            out_jaccards,
            out_q_cov,
            out_t_cov,
            out_shared,
            out_missing,
            out_extra,
            out_aa_id,
        ) = _match_loci_kernel(
            prep_q.n_loci,
            prep_q.orig_strands,
            prep_q.archetypes,
            prep_q.is_fragmented,
            prep_q.core_offsets,
            prep_q.core_keys,
            prep_q.core_roles,
            prep_q.core_fs,
            prep_q.core_next_orient,
            prep_q.core_gidx1,
            prep_q.core_gidx2,
            prep_q.mge_offsets,
            prep_q.mge_pos,
            prep_q.mge_keys,
            prep_t.n_loci,
            prep_t.orig_strands,
            prep_t.archetypes,
            prep_t.is_fragmented,
            prep_t.core_offsets,
            prep_t.core_keys,
            prep_t.core_roles,
            prep_t.core_fs,
            prep_t.core_next_orient,
            prep_t.core_gidx1,
            prep_t.core_gidx2,
            prep_t.mge_offsets,
            prep_t.mge_pos,
            prep_t.mge_keys,
            pident_mat,
            has_pident,
            int(LocusArchetype.UNKNOWN),
            float(self.novel_jaccard_threshold),
        )

        target_names = (
            self.targets.names[out_t]
            if len(self.targets.names) == len(self.targets)
            else np.full(len(out_t), "", dtype="U1")
        )
        res = Matches(
            query_indices=out_q,
            target_indices=out_t,
            match_types=out_flags,
            scores=out_scores,
            jaccards=out_jaccards,
            query_coverages=out_q_cov,
            target_coverages=out_t_cov,
            shared_genes=out_shared,
            missing_genes=out_missing,
            extra_genes=out_extra,
            mean_aa_identities=out_aa_id,
            target_names=target_names,
        )

        if min_score > 0.0:
            res = res.filter(min_score=min_score)

        if len(res) == 0:
            return res

        if sort or top_k is not None:
            pop = res._popcounts()
            order = np.lexsort((res.target_indices, -res.mean_aa_identities, pop, -res.scores, res.query_indices))
            res = res[order]
            if top_k is not None and top_k > 0:
                if top_k == 1:
                    return res.best_per_query()
                # Retain up to top_k matches per query_index
                q_ids = res.query_indices
                keep = np.ones(len(res), dtype=np.bool_)
                rank = 0
                prev_q = -1
                for i in range(len(res)):
                    qi = int(q_ids[i])
                    if qi != prev_q:
                        prev_q = qi
                        rank = 1
                    else:
                        rank += 1
                    if rank > top_k:
                        keep[i] = False
                res = res[keep]

        return res

    def is_novel(
        self,
        query: Locus | Loci,
        *,
        allow_variants: MatchType = BENIGN_VARIANT_MASK,
        min_coverage: float = 1.0,
    ) -> bool | np.ndarray[tuple[int], np.dtype[np.bool_]]:
        """Test whether a single `Locus` or each locus in a `Loci` batch is novel relative to `self.targets`."""
        if isinstance(query, Locus):
            matches = self.match(query, sort=False)
            return matches.is_novel(allow_variants=allow_variants, min_coverage=min_coverage)

        n_q = len(query)
        if n_q == 0:
            return np.empty(0, dtype=np.bool_)
        if len(self.targets) == 0:
            return np.ones(n_q, dtype=np.bool_)

        matches = self.match(query, sort=False)
        disallowed = ~np.uint32(int(allow_variants))
        valid_known = (
            ((matches.match_types & disallowed) == 0)
            & (matches.query_coverages >= min_coverage)
            & (matches.target_coverages >= min_coverage)
        )
        # Reshape (n_q, n_t) to check if each query has at least one known target match
        known_per_q = np.any(valid_known.reshape(n_q, len(self.targets)), axis=1)
        return ~known_per_q

    def align(
        self,
        query: Locus,
        target_index: int | str,
        *,
        canonicalize_strand: bool = True,
        heal_mges: bool = False,
    ) -> Alignment:
        """Perform detailed node-by-node bipartite `GraphAligner` alignment between `query` and `targets[target_index]`."""
        q_batch = self.assign_query_clusters(Loci.from_list([query]))
        q_loc = q_batch[0]
        t_loc = self.targets[target_index]
        if heal_mges:
            q_loc = q_loc.heal_mobile_elements(remove_nodes=True)
            t_loc = t_loc.heal_mobile_elements(remove_nodes=True)
        if canonicalize_strand:
            q_loc = q_loc.canonicalize_strand()
            t_loc = t_loc.canonicalize_strand()

        aligner = GraphAligner(cost_strategy=LocusNodeCostStrategy())
        return aligner.align(q_loc, t_loc)
