"""Vectorized locus comparison, structural difference bitflags, and database matching."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import IntFlag
from typing import Any, Literal, Self, cast, overload

import numpy as np
from numba import njit, prange

from serogen.core.collections import BatchedContainer, ScalarView
from serogen.core.graph.align import Alignment, CostStrategy, GraphAligner
from serogen.core.graph.models import GraphModel
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

    @property
    def is_exact(self) -> bool:
        """True if no variant flags are set (`self == MatchType.EXACT`)."""
        return self == MatchType.EXACT

    @property
    def is_novel(self) -> bool:
        """True if the comparison involves gene gain/loss, synteny rearrangement, or archetype mismatch."""
        novel_mask = MatchType.GENE_VARIANT | MatchType.SYNTENY_VARIANT | MatchType.ARCHETYPE_MISMATCH
        return bool(self & novel_mask)

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

    @property
    def is_novel(self) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        """1D boolean array (`(N,)`) indicating whether each match involves gene, synteny, or archetype divergence."""
        if len(self) == 0:
            return np.empty(0, dtype=np.bool_)
        novel_mask = np.uint32(int(MatchType.GENE_VARIANT | MatchType.SYNTENY_VARIANT | MatchType.ARCHETYPE_MISMATCH))
        return (self.match_types & novel_mask) != 0

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
    core_offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    core_keys: np.ndarray[tuple[int], np.dtype[np.int32]]
    core_roles: np.ndarray[tuple[int], np.dtype[np.int32]]
    core_fs: np.ndarray[tuple[int], np.dtype[np.uint8]]
    core_next_orient: np.ndarray[tuple[int], np.dtype[np.int8]]
    core_pidents: np.ndarray[tuple[int], np.dtype[np.float32]]
    mge_offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    mge_pos: np.ndarray[tuple[int], np.dtype[np.int32]]
    mge_keys: np.ndarray[tuple[int], np.dtype[np.int32]]


def _prepare_loci_arrays(
    loci: Loci,
    *,
    use_clusters: bool,
    unclustered_base: int = -1,
    gene_pidents: np.ndarray | None = None,
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
            core_pidents=np.empty(0, dtype=np.float32),
            mge_offsets=z_i32.copy(),
            mge_pos=e_i32,
            mge_keys=e_i32,
        )

    orig_strands = loci.intervals.strands.copy()
    archetypes = loci.archetypes.copy()
    is_fragmented = loci.is_fragmented.astype(np.uint8)
    canon = loci.canonicalize_strands()
    healed = canon.heal_mobile_elements(remove_nodes=True)

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
    orig_pidents = (
        gene_pidents.astype(np.float32, copy=False)
        if (gene_pidents is not None and len(gene_pidents) == loci.total_nodes)
        else np.ones(loci.total_nodes, dtype=np.float32)
    )
    raw_pidents = (
        orig_pidents[raw_gidx]
        if (tot_raw > 0 and len(orig_pidents) > int(np.max(raw_gidx)))
        else np.ones(tot_raw, dtype=np.float32)
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

    raw_next_orient = np.full(tot_raw, -1, dtype=np.int8)
    raw_next_is_break = np.zeros(tot_raw, dtype=np.bool_)
    if len(healed.edge_sources) > 0:
        e_orients = IntergenicType.ORIENTATION.extract_array(healed.edge_types).astype(np.int8)
        e_breaks = IntergenicType.GAP_BIN.extract_array(healed.edge_types) == 14
        raw_next_orient[healed.edge_sources] = e_orients
        raw_next_is_break[healed.edge_sources] = e_breaks

    if np.any(raw_next_is_break):
        core_offsets = np.zeros(n_loci + 1, dtype=np.int32)
        keep_keys: list[int] = []
        keep_roles: list[int] = []
        keep_fs: list[int] = []
        keep_orient: list[int] = []
        keep_pident: list[float] = []
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
                    keep_fs.append(0)
                    keep_orient.append(int(raw_next_orient[k + 1]))
                    keep_pident.append(float(max(raw_pidents[k], raw_pidents[k + 1])))
                    k += 2
                else:
                    is_break_adj = bool(raw_next_is_break[k]) or (k > s and bool(raw_next_is_break[k - 1]))
                    fs_val = 0 if (is_break_adj and bool(raw_partials[k])) else int(raw_fs[k])
                    keep_keys.append(int(raw_keys[k]))
                    keep_roles.append(int(raw_roles[k]))
                    keep_fs.append(fs_val)
                    keep_orient.append(int(raw_next_orient[k]))
                    keep_pident.append(float(raw_pidents[k]))
                    k += 1
            core_offsets[i + 1] = len(keep_keys)
        core_keys = np.array(keep_keys, dtype=np.int32)
        core_roles = np.array(keep_roles, dtype=np.int32)
        core_fs = np.array(keep_fs, dtype=np.uint8)
        core_next_orient = np.array(keep_orient, dtype=np.int8)
        core_pidents = np.array(keep_pident, dtype=np.float32)
    else:
        core_offsets = raw_offsets
        core_keys = raw_keys
        core_roles = raw_roles
        core_fs = raw_fs
        core_next_orient = raw_next_orient
        core_pidents = raw_pidents

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
        core_pidents=core_pidents,
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
    q_core_pidents: np.ndarray,
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
    t_mge_offs: np.ndarray,
    t_mge_pos: np.ndarray,
    t_mge_keys: np.ndarray,
    target_gene_pidents: np.ndarray,
    has_target_pident: bool,
    unknown_arch: int,
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

    for pair_idx in prange(total_pairs):
        qi = pair_idx // n_t
        ti = pair_idx % n_t

        out_q[pair_idx] = qi
        out_t[pair_idx] = ti
        flags = np.uint32(0)

        qa = int(q_archs[qi])
        ta = int(t_archs[ti])
        if qa != unknown_arch and ta != unknown_arch and qa != ta:
            flags |= FLAG_ARCH

        if q_strands[qi] != t_strands[ti]:
            flags |= FLAG_STRAND
        if q_frag[qi] != t_frag[ti]:
            flags |= FLAG_FRAG

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

        qc_s, qc_e = int(q_core_offs[qi]), int(q_core_offs[qi + 1])
        tc_s, tc_e = int(t_core_offs[ti]), int(t_core_offs[ti + 1])
        len_q = qc_e - qc_s
        len_t = tc_e - tc_s

        t_used = np.zeros(len_t, dtype=np.uint8)
        q_match_t = np.full(len_q, -1, dtype=np.int32)
        shared = 0

        for a in range(len_q):
            k_q = q_core_keys[qc_s + a]
            for b in range(len_t):
                if t_used[b] == 0 and t_core_keys[tc_s + b] == k_q:
                    t_used[b] = 1
                    q_match_t[a] = b
                    shared += 1
                    break

        missing = len_t - shared
        extra = len_q - shared
        union = len_q + len_t - shared

        jaccard = np.float32(shared / union) if union > 0 else np.float32(1.0)
        q_cov = np.float32(shared / len_q) if len_q > 0 else np.float32(1.0)
        t_cov = np.float32(shared / len_t) if len_t > 0 else np.float32(1.0)

        if missing > 0 or extra > 0:
            flags |= FLAG_GENE

        prev_a = -1
        prev_b = -1
        collinear_steps = 0
        total_steps = 0
        aa_sum = np.float32(0.0)

        for a in range(len_q):
            b = int(q_match_t[a])
            if b == -1:
                continue

            if has_target_pident:
                pid = target_gene_pidents[qc_s + a, tc_s + b]
                if pid <= np.float32(0.0):
                    pid = q_core_pidents[qc_s + a]
            else:
                pid = q_core_pidents[qc_s + a]
            aa_sum += pid

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
                    if q_core_orient[qc_s + prev_a] != t_core_orient[tc_s + prev_b]:
                        flags |= FLAG_SYNTENY
                        is_step_collinear = False
                if is_step_collinear:
                    collinear_steps += 1

            prev_a = a
            prev_b = b

        mean_aa_id = np.float32(aa_sum / shared) if shared > 0 else np.float32(0.0)
        allelic_deduction = np.float32(0.02) * max(np.float32(0.0), np.float32(1.0) - mean_aa_id)

        if (flags & FLAG_SYNTENY) and shared >= 2 and collinear_steps == 0:
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
                flags = (flags & ~FLAG_SYNTENY) | FLAG_STRAND
                collinear_steps = total_steps

        if flags == 0:
            score = np.float32(1.0) - allelic_deduction
        elif (flags & (FLAG_GENE | FLAG_SYNTENY | FLAG_ARCH)) == 0:
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
    """Semantic node compatibility cost model for aligning two `Locus` graphs via `GraphAligner`."""

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

        incompatible = edge_a != edge_b
        is_edge_pair = (edge_a == 1) & (edge_b == 1)

        orient_a = IntergenicType.ORIENTATION.extract_array(la)[:, None]
        orient_b = IntergenicType.ORIENTATION.extract_array(lb)[None, :]
        edge_cost = np.where(orient_a == orient_b, 0.0, 0.5)

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
        gene_cost = np.where(same_cid & same_role & same_fs, 0.0, gene_cost)
        gene_cost = np.where(same_cid & same_role & ~same_fs, 0.05, gene_cost)
        gene_cost = np.where(same_cid & ~same_role, 0.15, gene_cost)
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
    """Vectorized 1-to-many and batch-to-many biosynthetic locus matcher.

    Indexes a reference database of target [`Loci`][serogen.loci.models.Loci] (automatically clustering
    target proteins and building a single [`RandstrobeIndex`][serogen.core.seq.kmers.RandstrobeIndex] over
    target proteins when translations are present) and compares query [`Locus`][serogen.loci.models.Locus]
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
        self.k = k
        self.s = s
        self.aligner_k = aligner_k
        self.aligner = PairwiseAligner(k=aligner_k)

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

        if has_target_seqs:
            self._target_seqs = self.targets.genes.translations
            self._target_index: RandstrobeIndex | None = RandstrobeIndex.build(
                self._target_seqs,
                k=k,
                s=s,
                sort_by_hash=True,
            )
        else:
            self._target_seqs = Sequences.empty()
            self._target_index = None

        self._prepared_targets = _prepare_loci_arrays(
            self.targets,
            use_clusters=self.use_clusters,
            unclustered_base=-1,
        )

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

    def assign_query_clusters(
        self,
        queries: Loci,
    ) -> tuple[Loci, np.ndarray[tuple[int], np.dtype[np.float32]], np.ndarray[tuple[int, int], np.dtype[np.float32]]]:
        """Assign target homology `CLUSTER_ID`s and compute per-gene AA identity in a single `RandstrobeIndex` pass."""
        n_q_nodes = queries.total_nodes
        n_t_nodes = self.targets.total_nodes
        best_pidents = np.ones(n_q_nodes, dtype=np.float32)
        pident_mat = np.zeros((n_q_nodes, n_t_nodes), dtype=np.float32)

        if (
            not self._has_target_seqs
            or self._target_index is None
            or n_q_nodes == 0
            or len(queries.genes) != n_q_nodes
            or len(queries.genes.translations) != n_q_nodes
        ):
            return queries, best_pidents, pident_mat

        q_proteins = queries.genes.translations
        q_partial = self._partial_node_mask(queries)
        t_partial = self._target_partial_mask
        has_any_partial = bool(np.any(q_partial) or np.any(t_partial))
        seed_min = 1 if has_any_partial else max(1, self.min_score // 2)

        seeds = self._target_index.hits(q_proteins, min_score=seed_min)
        if len(seeds) == 0:
            return queries, best_pidents, pident_mat

        alignments = self.aligner.align_seeds(q_proteins, self._target_seqs, seeds)
        q_idx = seeds.query_indices.astype(np.int32)
        t_idx = seeds.target_indices.astype(np.int32)
        q_lens = q_proteins.lengths[q_idx]
        t_lens = self._target_seqs.lengths[t_idx]
        q_span = alignments.q_ends - alignments.q_starts
        t_span = alignments.t_ends - alignments.t_starts
        aligned_span = np.maximum(q_span, t_span)
        pair_is_partial = q_partial[q_idx] | (
            t_partial[t_idx] if len(t_partial) == n_t_nodes else np.zeros(len(t_idx), dtype=np.bool_)
        )
        denom = np.where(pair_is_partial, np.minimum(q_lens, t_lens), np.maximum(q_lens, t_lens))
        cov = np.clip(aligned_span / np.maximum(denom, 1), 0.0, 1.0)
        pids = np.clip((alignments.pidents / 100.0) * cov, 0.0, 1.0).astype(np.float32)
        np.maximum.at(pident_mat, (q_idx, t_idx), pids)

        if self.use_clusters and not bool(np.any(queries.cluster_ids > 0)):
            valid = alignments.pidents >= self.min_identity
            if self.min_bitscore > 0.0:
                valid &= alignments.bitscores >= self.min_bitscore
            if has_any_partial and self.min_score > 1:
                valid &= pair_is_partial | (seeds.scores >= self.min_score)
            if self.min_coverage > 0.0:
                q_cov = q_span / np.maximum(q_lens, 1)
                t_cov = t_span / np.maximum(t_lens, 1)
                unidir_cov = aligned_span / np.maximum(np.minimum(q_lens, t_lens), 1)
                both_cov = (q_cov >= self.min_coverage) & (t_cov >= self.min_coverage)
                valid &= np.where(pair_is_partial, unidir_cov >= self.min_coverage, both_cov)

            t_cids = self.targets.cluster_ids
            valid &= t_cids[t_idx] > 0
            assigned_cids = np.full(n_q_nodes, -1, dtype=np.int32)
            if np.any(valid):
                v_q = q_idx[valid]
                v_t = t_idx[valid]
                order = np.lexsort((alignments.scores[valid], pids[valid], v_q))
                v_q_sorted = v_q[order]
                v_t_sorted = v_t[order]
                assigned_cids[v_q_sorted] = t_cids[v_t_sorted] - 1
                best_pidents[v_q_sorted] = pids[valid][order]
            queries = queries.with_clusters(assigned_cids)

        return queries, best_pidents, pident_mat

    def match(
        self,
        query: Locus | Loci | Iterable[Locus],
        *,
        top_k: int | None = None,
        min_score: float = 0.0,
        sort: bool = True,
    ) -> Matches:
        """Compare one or more query loci against all indexed target loci."""
        if isinstance(query, Locus):
            q_batch = Loci.from_list([query])
        elif isinstance(query, Loci):
            q_batch = query
        else:
            q_batch = Loci.from_list(list(query))

        if len(q_batch) == 0 or len(self.targets) == 0:
            return Matches.empty()

        q_mapped, best_pidents, pident_mat = self.assign_query_clusters(q_batch)
        use_clusters = self.use_clusters or bool(np.any(q_mapped.cluster_ids > 0))
        has_target_pident = pident_mat.size > 0 and bool(np.any(pident_mat > 0.0))

        prep_t = (
            self._prepared_targets
            if use_clusters == self.use_clusters
            else _prepare_loci_arrays(self.targets, use_clusters=use_clusters, unclustered_base=-1)
        )
        q_unclustered_base = -1 - int(self.targets.total_nodes) - 1
        prep_q = _prepare_loci_arrays(
            q_mapped,
            use_clusters=use_clusters,
            unclustered_base=q_unclustered_base,
            gene_pidents=best_pidents,
        )

        # Map pident_mat from raw node indices to prepared core gene indices if identical counts
        core_pident_mat = (
            pident_mat
            if (has_target_pident and pident_mat.shape == (len(prep_q.core_keys), len(prep_t.core_keys)))
            else np.empty((0, 0), dtype=np.float32)
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
            prep_q.core_pidents,
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
            prep_t.mge_offsets,
            prep_t.mge_pos,
            prep_t.mge_keys,
            core_pident_mat,
            core_pident_mat.size > 0,
            int(LocusArchetype.UNKNOWN),
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

    def align(
        self,
        query: Locus,
        target_index: int | str,
        *,
        canonicalize_strand: bool = True,
        heal_mges: bool = False,
    ) -> Alignment:
        """Perform detailed node-by-node bipartite `GraphAligner` alignment between `query` and `targets[target_index]`."""
        q_batch, _, _ = self.assign_query_clusters(Loci.from_list([query]))
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


LocusMatch = Match
LocusMatches = Matches
