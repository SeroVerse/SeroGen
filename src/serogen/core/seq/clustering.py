r"""Sequence clustering algorithms, ragged SoA data structures, and Numba kernels.

Provides high-performance sequence clustering ([`Clusterer`][serogen.core.seq.clustering.Clusterer])
and ragged Structure-of-Arrays cluster containers ([`Clusters`][serogen.core.seq.clustering.Clusters],
[`Cluster`][serogen.core.seq.clustering.Cluster]) combining sparse syncmer-linked randstrobe
pre-filtering, parallel banded Smith-Waterman-Gotoh alignment, and sub-millisecond Numba
greedy set-cover / CD-HIT clustering.
"""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self, overload

import numpy as np
from numba import njit, prange
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csgraph, csr_array

from serogen.core.collections import RaggedArrayContainer
from serogen.core.seq.alphabet import Alphabet
from serogen.core.seq.kmers import RandstrobeIndex
from serogen.core.seq.pairwise import PairwiseAligner, _blosum62_matrix
from serogen.core.seq.seq import Sequences


# Containers -----------------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True, eq=False)
class Cluster:
    r"""Scalar view of a single sequence cluster.

    Attributes:
        representative: Sequence index of the cluster centroid in the source collection.
        members: 1D array of sequence indices belonging to this cluster.
    """

    representative: int
    members: np.ndarray[tuple[int], np.dtype[np.int32]]

    def __len__(self) -> int:
        r"""Return the number of member sequences in this cluster."""
        return len(self.members)

    def __eq__(self, other: object) -> bool:
        r"""Compare two Cluster views for representative and member equality."""
        if not isinstance(other, Cluster):
            return NotImplemented
        return self.representative == other.representative and bool(np.array_equal(self.members, other.members))


@dataclass(slots=True, frozen=True)
class Clusters(RaggedArrayContainer[Cluster, "Clusters"]):
    r"""Ragged Structure-of-Arrays (SoA) container for sequence clustering results.

    Stores centroid representative indices (`representatives`) and variable-length
    member sequence indices in a flat contiguous array (`members`) indexed by
    `offsets` and `lengths` for $O(1)$ cluster slicing and member lookup.

    Attributes:
        representatives: 1D array of sequence indices representing the centroid of each cluster.
        members: Flat 1D array of member sequence indices concatenated across all clusters.
        offsets: Starting index into `members` for each cluster.
        lengths: Number of member sequences in each cluster.
    """

    representatives: np.ndarray[tuple[int], np.dtype[np.int32]]
    members: np.ndarray[tuple[int], np.dtype[np.int32]]
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]]

    def __len__(self) -> int:
        r"""Return the total number of clusters in the container."""
        return len(self.offsets)

    @property
    def n_clusters(self) -> int:
        r"""Total number of clusters $C$."""
        return len(self.offsets)

    @property
    def n_seqs(self) -> int:
        r"""Total number of clustered sequences $N$."""
        return len(self.members)

    @property
    def sizes(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        r"""1D array of member counts for each cluster (alias for `lengths`)."""
        return self.lengths

    @property
    def representative_indices(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        r"""1D array of centroid sequence indices (alias for `representatives`)."""
        return self.representatives

    @property
    def cluster_ids(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        r"""1D array mapping each sequence index to its cluster ID $c \in [0, C-1]$."""
        if len(self.members) == 0:
            return np.empty(0, dtype=np.int32)
        n = int(self.members.max()) + 1
        out = np.full(n, -1, dtype=np.int32)
        out[self.members] = np.repeat(np.arange(len(self.lengths), dtype=np.int32), self.lengths)
        return out

    def as_csr(self, n_seqs: int | None = None) -> csr_array:
        r"""Return a zero-copy $(C \times N)$ sparse CSR cluster-membership incidence matrix.

        Row $c$ contains `1` at column $i$ for each sequence $i \in \text{members}_c$.

        Args:
            n_seqs: Total number of sequences $N$ (columns). If `None`, inferred from `max(members) + 1`.

        Returns:
            csr_array: Sparse $(C \times N)$ `uint8` membership matrix.
        """
        c = len(self)
        if c == 0 or len(self.members) == 0:
            cols = n_seqs if n_seqs is not None else 0
            return csr_array((c, cols), dtype=np.uint8)
        cols = n_seqs if n_seqs is not None else int(self.members.max()) + 1
        data = np.ones(len(self.members), dtype=np.uint8)
        return csr_array((data, self.members, self.indptr), shape=(c, cols), copy=False)

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> Cluster: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> Clusters: ...

    @overload
    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> Cluster | Clusters: ...

    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> Cluster | Clusters:
        r"""Access a single [`Cluster`][serogen.core.seq.clustering.Cluster] or a sliced [`Clusters`][serogen.core.seq.clustering.Clusters] subset."""
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Cluster index {item} out of bounds [0, {len(self) - 1}]")
            off = int(self.offsets[idx])
            length = int(self.lengths[idx])
            return Cluster(
                representative=int(self.representatives[idx]),
                members=self.members[off : off + length],
            )

        indices, new_offsets, new_lengths, (new_members,) = self._slice_ragged(item, self.members)
        if len(indices) == 0:
            return self.empty()

        return type(self)(
            representatives=self.representatives[indices],
            members=new_members,
            offsets=new_offsets,
            lengths=new_lengths,
        )

    def __iter__(self) -> Iterator[Cluster]:
        r"""Iterate over all clusters in the container."""
        for i in range(len(self)):
            off = int(self.offsets[i])
            length = int(self.lengths[i])
            yield Cluster(
                representative=int(self.representatives[i]),
                members=self.members[off : off + length],
            )

    def get_members(self, cluster_id: int) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        r"""Return 1D array of original sequence indices assigned to the specified cluster ID.

        Args:
            cluster_id: 0-based integer cluster identifier.

        Returns:
            np.ndarray: Zero-copy slice of sequence indices belonging to this cluster.
        """
        return self[cluster_id].members

    def centroids(self, seqs: Sequences) -> Sequences:
        r"""Extract the representative centroid sequences from the source collection.

        Args:
            seqs: Source [`Sequences`][serogen.core.seq.seq.Sequences] collection.

        Returns:
            Sequences: Sliced collection containing only cluster centroids.
        """
        if len(seqs) != self.n_seqs:
            raise ValueError(f"Sequence length mismatch: expected {self.n_seqs}, got {len(seqs)}")
        return seqs[self.representatives]

    @classmethod
    def from_assignments(
        cls,
        cluster_ids: np.ndarray[tuple[int], np.dtype[np.int32]],
        representatives: np.ndarray[tuple[int], np.dtype[np.int32]],
        sizes: np.ndarray[tuple[int], np.dtype[np.int32]],
    ) -> Self:
        r"""Construct a [`Clusters`][serogen.core.seq.clustering.Clusters] container from flat cluster ID assignments.

        Args:
            cluster_ids: 1D array mapping each sequence $i \in [0, N-1]$ to cluster ID $c \in [0, C-1]$.
            representatives: 1D array of centroid sequence indices (length $C$).
            sizes: 1D array of member counts per cluster (length $C$).

        Returns:
            Clusters: Ragged SoA container with $O(1)$ member indexing.
        """
        n_clusters = len(representatives)
        if n_clusters == 0:
            return cls.empty()

        offsets = cls.lengths_to_offsets(sizes)
        members = _pack_members(cluster_ids, offsets)
        return cls(
            representatives=representatives,
            members=members,
            offsets=offsets,
            lengths=sizes,
        )

    @classmethod
    def empty(cls) -> Self:
        r"""Construct an empty 0-length [`Clusters`][serogen.core.seq.clustering.Clusters] instance."""
        return cls(
            representatives=np.empty(0, dtype=np.int32),
            members=np.empty(0, dtype=np.int32),
            offsets=np.empty(0, dtype=np.int32),
            lengths=np.empty(0, dtype=np.int32),
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        r"""Concatenate multiple [`Clusters`][serogen.core.seq.clustering.Clusters] containers into a single batch."""
        batch_list = list(batches)
        if not batch_list:
            return cls.empty()

        non_empty = [b for b in batch_list if len(b) > 0]
        if not non_empty:
            return cls.empty()

        representatives = np.concatenate([b.representatives for b in non_empty])
        lengths = np.concatenate([b.lengths for b in non_empty])
        offsets = cls.lengths_to_offsets(lengths)
        members = np.concatenate([b.members for b in non_empty if len(b.members) > 0])
        return cls(
            representatives=representatives,
            members=members,
            offsets=offsets,
            lengths=lengths,
        )


# Clustering Engine ---------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class Clusterer:
    r"""High-performance sequence clustering engine.

    Combines sparse syncmer-linked randstrobe pre-filtering ([`RandstrobeIndex`][serogen.core.seq.kmers.RandstrobeIndex]),
    SIMD-parallel banded Smith-Waterman-Gotoh alignment ([`PairwiseAligner`][serogen.core.seq.pairwise.PairwiseAligner]),
    and compiled Numba/SciPy clustering algorithms to group biological sequences.

    Modes:
        - `"cdhit"` (default): Length-descending greedy incremental clustering (CD-HIT style).
          Guarantees that full-length proteins are selected as cluster representatives rather than fragments.
        - `"mcl"`: Markov Cluster Algorithm (MCL) on BLOSUM62 alignment bitscores. Sensitive to
          divergent functional homologs (e.g., glycosyltransferases, polymerases, flippases) via
          random-walk expansion and inflation ($I$).
        - `"greedy_cover"`: Iterative maximum-uncovered-degree greedy set cover.
        - `"upgma"`: Hierarchical average-linkage (UPGMA) clustering via SciPy.
        - `"connected_components"`: Transitive closure core components.

    Args:
        min_identity: Minimum pairwise sequence percent identity threshold $[0.0, 100.0]$. Defaults to 80.0
            (automatically relaxed to 25.0 when `mode="mcl"` and left at default 80.0).
        min_coverage: Minimum alignment coverage fraction $[0.0, 1.0]$ on both query and target. Defaults to 0.0.
        min_bitscore: Minimum Karlin-Altschul alignment bitscore threshold. Defaults to 0.0
            (automatically set to 40.0 when `mode="mcl"` and left at 0.0).
        inflation: MCL inflation parameter $I > 1.0$ controlling cluster granularity. Defaults to 2.0.
        min_score: Minimum shared randstrobe hash hits for alignment candidate seed filtering. Defaults to 2.
        k: Strobemer seed length. Defaults to 10.
        s: Strobemer sampling stride. Defaults to 5.
        aligner_k: Bandwidth parameter ($2k + 1$ diagonals) for banded Gotoh alignment. Defaults to 20.
        mode: Clustering algorithm mode. Defaults to `"cdhit"`.
        embedding_weight: Blend weight $\alpha \in [0.0, 1.0]$ for active-site motif cosine similarity
            in MCL edge weighting and role-gated twilight-zone rescue. Defaults to 0.0 (automatically
            uses 0.25 when `embeddings` are explicitly supplied to `cluster()`).
        motif_rescue_threshold: Minimum cosine similarity $[0.0, 1.0]$ between active-site embeddings
            required to trigger role-gated twilight-zone identity relaxation. Defaults to 0.75.
    """

    min_identity: float = 80.0
    min_coverage: float = 0.0
    min_bitscore: float = 0.0
    inflation: float = 2.0
    min_score: int = 2
    k: int = 10
    s: int = 5
    aligner_k: int = 20
    mode: Literal["cdhit", "mcl", "greedy_cover", "upgma", "connected_components"] = "cdhit"
    embedding_weight: float = 0.0
    motif_rescue_threshold: float = 0.75

    @property
    def aligner(self) -> PairwiseAligner:
        r"""Return a configured [`PairwiseAligner`][serogen.core.seq.pairwise.PairwiseAligner] instance."""
        return PairwiseAligner(k=self.aligner_k)

    def cluster(
        self,
        seqs: Sequences,
        roles: np.ndarray[tuple[int], np.dtype[Any]] | Sequence[int] | None = None,
        embeddings: np.ndarray[tuple[int, int], np.dtype[Any]] | None = None,
    ) -> Clusters:
        r"""Cluster an input batch of sequences into non-redundant representative groups.

        Args:
            seqs: Input collection of sequences to cluster.
            roles: Optional 1D array/sequence of integer `GeneRole` IDs (length $N$). When provided,
                prevents clustering across incompatible known roles (`role > 0`) and, in `"mcl"` mode
                (or when `min_identity < 60.0`), evaluates all intra-role pairs so divergent functional
                homologs sharing the same role are aligned even without shared exact k-mer seeds.
            embeddings: Optional 2D array `(N, D)` of sequence representations (e.g., active-site
                `conv_max` or `task_weighted` features from `GeneRoleClassifier.transform()`). When
                provided, enables role-gated twilight-zone homology rescue for pairs with cosine
                similarity `>= self.motif_rescue_threshold` and blends cosine similarity into MCL
                edge weights.

        Returns:
            Clusters: Ragged SoA container of cluster assignments and centroid representatives.
        """
        n = len(seqs)
        if n == 0:
            return Clusters.empty()

        if n == 1:
            return Clusters(
                representatives=np.zeros(1, dtype=np.int32),
                members=np.zeros(1, dtype=np.int32),
                offsets=np.zeros(1, dtype=np.int32),
                lengths=np.ones(1, dtype=np.int32),
            )

        roles_arr: np.ndarray[tuple[int], np.dtype[np.int32]] | None = None
        if roles is not None:
            roles_arr = np.asarray(roles, dtype=np.int32)
            if len(roles_arr) != n:
                raise ValueError(f"roles length ({len(roles_arr)}) must match seqs length ({n}).")

        emb_norm: np.ndarray[tuple[int, int], np.dtype[np.float32]] | None = None
        if embeddings is not None:
            emb_arr = np.asarray(embeddings, dtype=np.float32)
            if emb_arr.ndim != 2 or emb_arr.shape[0] != n:
                raise ValueError(f"embeddings shape {emb_arr.shape} must be 2D with first dimension {n}.")
            norms = np.maximum(np.linalg.norm(emb_arr, axis=1, keepdims=True), np.float32(1e-8))
            emb_norm = np.ascontiguousarray(emb_arr / norms, dtype=np.float32)

        eff_min_identity = 25.0 if (self.mode == "mcl" and self.min_identity == 80.0) else float(self.min_identity)
        eff_min_bitscore = 40.0 if (self.mode == "mcl" and self.min_bitscore <= 0.0) else float(self.min_bitscore)
        sensitive_mode = self.mode == "mcl" or eff_min_identity < 60.0 or emb_norm is not None

        # 1. Sparse Candidate Pre-Filtering via RandstrobeIndex
        k_eff = min(self.k, 5) if sensitive_mode else self.k
        s_eff = min(self.s, 2) if sensitive_mode else self.s
        min_score_eff = 1 if sensitive_mode else self.min_score

        target_idx = RandstrobeIndex.build(seqs, k=k_eff, s=s_eff, sort_by_hash=True)
        query_idx = RandstrobeIndex.build(seqs, k=k_eff, s=s_eff, sort_by_hash=False)

        seeds = target_idx.hits(query_idx, min_score=min_score_eff)
        upper_mask = seeds.query_indices < seeds.target_indices
        candidate_seeds = seeds.filter(upper_mask) if len(seeds) > 0 else seeds

        q_chunks: list[np.ndarray[tuple[int], np.dtype[np.int32]]] = []
        t_chunks: list[np.ndarray[tuple[int], np.dtype[np.int32]]] = []
        pident_chunks: list[np.ndarray[tuple[int], np.dtype[np.float32]]] = []
        bitscore_chunks: list[np.ndarray[tuple[int], np.dtype[np.float32]]] = []
        q_cov_chunks: list[np.ndarray[tuple[int], np.dtype[np.float64]]] = []
        t_cov_chunks: list[np.ndarray[tuple[int], np.dtype[np.float64]]] = []

        seeded_pairs: set[int] = set()
        if len(candidate_seeds) > 0:
            alignments = self.aligner.align_seeds(seqs, seqs, candidate_seeds)
            q_idx = candidate_seeds.query_indices.astype(np.int32)
            t_idx = candidate_seeds.target_indices.astype(np.int32)
            if roles_arr is not None:
                r_q = roles_arr[q_idx]
                r_t = roles_arr[t_idx]
                compat = (r_q == r_t) | (r_q == 29) | (r_t == 29) | (r_q < 0) | (r_t < 0) | (alignments.pidents >= 80.0)
            else:
                compat = np.ones(len(q_idx), dtype=np.bool_)

            if np.any(compat):
                q_idx = q_idx[compat]
                t_idx = t_idx[compat]
                p_sub = alignments.pidents[compat]
                b_sub = alignments.bitscores[compat]
                q_lens = np.maximum(seqs.lengths[q_idx], 1)
                t_lens = np.maximum(seqs.lengths[t_idx], 1)
                qc_sub = (alignments.q_ends[compat] - alignments.q_starts[compat]) / q_lens
                tc_sub = (alignments.t_ends[compat] - alignments.t_starts[compat]) / t_lens
                q_chunks.append(q_idx)
                t_chunks.append(t_idx)
                pident_chunks.append(p_sub)
                bitscore_chunks.append(b_sub)
                q_cov_chunks.append(qc_sub)
                t_cov_chunks.append(tc_sub)

                # Only mark a seeded pair as resolved if the banded alignment actually passed thresholds;
                # spurious single-strobe collisions on off-diagonal offsets must not block Stage 1b unbanded SW.
                passed_seed = (p_sub >= eff_min_identity) & (b_sub >= eff_min_bitscore)
                if self.min_coverage > 0.0:
                    passed_seed &= (qc_sub >= self.min_coverage) & (tc_sub >= self.min_coverage)
                for u_i, v_i in zip(q_idx[passed_seed], t_idx[passed_seed], strict=False):
                    seeded_pairs.add(int(u_i) * n + int(v_i))

        # 1b. MMseqs2-Style Reduced-Alphabet + Ungapped BLOSUM62 Diagonal Prefilter & Unbanded SW for Divergent Homologs
        if sensitive_mode:
            min_cov_eff = max(float(self.min_coverage), 0.0)
            extra_q_parts: list[np.ndarray[tuple[int], np.dtype[np.int32]]] = []
            extra_t_parts: list[np.ndarray[tuple[int], np.dtype[np.int32]]] = []

            if roles_arr is not None:
                small_q_list: list[int] = []
                small_t_list: list[int] = []
                seq_lens = seqs.lengths
                for r_val in np.unique(roles_arr):
                    if r_val < 0 or r_val == 29:
                        continue
                    members_r = np.where(roles_arr == r_val)[0].astype(np.int32)
                    m_r = len(members_r)
                    if 2 <= m_r <= 250:
                        for a_i in range(m_r):
                            u_idx = int(members_r[a_i])
                            l_u = int(seq_lens[u_idx])
                            for b_i in range(a_i + 1, m_r):
                                v_idx = int(members_r[b_i])
                                l_v = int(seq_lens[v_idx])
                                if min_cov_eff > 0.0 and min(l_u, l_v) < min_cov_eff * max(l_u, l_v):
                                    continue
                                if (u_idx * n + v_idx) not in seeded_pairs:
                                    small_q_list.append(u_idx)
                                    small_t_list.append(v_idx)
                    elif m_r > 250:
                        sub_seqs = seqs[members_r]
                        sub_q, sub_t = _mmseqs12_k4_ungapped_prefilter(
                            sub_seqs.seqs,
                            sub_seqs.offsets,
                            sub_seqs.lengths,
                            min_cov_eff,
                            42,
                            Alphabet.MMSEQS12.kmer_lut,
                            _blosum62_matrix(),
                        )
                        if len(sub_q) > 0:
                            gq = members_r[sub_q]
                            gt = members_r[sub_t]
                            keep_m = np.array(
                                [
                                    (int(u_i) * n + int(v_i)) not in seeded_pairs
                                    for u_i, v_i in zip(gq, gt, strict=False)
                                ],
                                dtype=np.bool_,
                            )
                            if np.any(keep_m):
                                extra_q_parts.append(gq[keep_m])
                                extra_t_parts.append(gt[keep_m])
                        if emb_norm is not None:
                            sub_emb = emb_norm[members_r]
                            sim_mat = sub_emb @ sub_emb.T
                            eq_idx, et_idx = np.where(np.triu(sim_mat >= float(self.motif_rescue_threshold), k=1))
                            if len(eq_idx) > 0:
                                gq_e = members_r[eq_idx.astype(np.int32)]
                                gt_e = members_r[et_idx.astype(np.int32)]
                                keep_e = np.array(
                                    [
                                        (int(u_i) * n + int(v_i)) not in seeded_pairs
                                        for u_i, v_i in zip(gq_e, gt_e, strict=False)
                                    ],
                                    dtype=np.bool_,
                                )
                                if np.any(keep_e):
                                    extra_q_parts.append(gq_e[keep_e])
                                    extra_t_parts.append(gt_e[keep_e])

                if small_q_list:
                    sq_arr = np.array(small_q_list, dtype=np.int32)
                    st_arr = np.array(small_t_list, dtype=np.int32)
                    if len(sq_arr) > 250:
                        keep_u = _ungapped_pair_prefilter(
                            seqs.seqs,
                            seqs.offsets,
                            seqs.lengths,
                            sq_arr,
                            st_arr,
                            _blosum62_matrix(),
                            40,
                        )
                        if emb_norm is not None:
                            cos_small = np.sum(emb_norm[sq_arr] * emb_norm[st_arr], axis=1)
                            keep_u = keep_u | (cos_small >= float(self.motif_rescue_threshold))
                        sq_arr = sq_arr[keep_u]
                        st_arr = st_arr[keep_u]
                    if len(sq_arr) > 0:
                        extra_q_parts.append(sq_arr)
                        extra_t_parts.append(st_arr)
            else:
                cand_q, cand_t = _mmseqs12_k4_ungapped_prefilter(
                    seqs.seqs,
                    seqs.offsets,
                    seqs.lengths,
                    min_cov_eff,
                    42,
                    Alphabet.MMSEQS12.kmer_lut,
                    _blosum62_matrix(),
                )
                if len(cand_q) > 0:
                    if seeded_pairs:
                        keep_m = np.array(
                            [
                                (int(u_i) * n + int(v_i)) not in seeded_pairs
                                for u_i, v_i in zip(cand_q, cand_t, strict=False)
                            ],
                            dtype=np.bool_,
                        )
                        cand_q = cand_q[keep_m]
                        cand_t = cand_t[keep_m]
                    if len(cand_q) > 0:
                        extra_q_parts.append(cand_q)
                        extra_t_parts.append(cand_t)
                if emb_norm is not None and n <= 2000:
                    sim_mat = emb_norm @ emb_norm.T
                    eq_idx, et_idx = np.where(np.triu(sim_mat >= float(self.motif_rescue_threshold), k=1))
                    if len(eq_idx) > 0:
                        gq_e = eq_idx.astype(np.int32)
                        gt_e = et_idx.astype(np.int32)
                        keep_e = np.array(
                            [
                                (int(u_i) * n + int(v_i)) not in seeded_pairs
                                for u_i, v_i in zip(gq_e, gt_e, strict=False)
                            ],
                            dtype=np.bool_,
                        )
                        if np.any(keep_e):
                            extra_q_parts.append(gq_e[keep_e])
                            extra_t_parts.append(gt_e[keep_e])

            if extra_q_parts:
                extra_q = np.concatenate(extra_q_parts)
                extra_t = np.concatenate(extra_t_parts)
                if len(extra_q) > 1:
                    pair_keys = extra_q.astype(np.int64) * n + extra_t.astype(np.int64)
                    _, uniq_idx = np.unique(pair_keys, return_index=True)
                    extra_q = extra_q[uniq_idx]
                    extra_t = extra_t[uniq_idx]
                extra_aln = self.aligner(seqs[extra_q], seqs[extra_t], seeds=None)
                q_lens_e = np.maximum(seqs.lengths[extra_q], 1)
                t_lens_e = np.maximum(seqs.lengths[extra_t], 1)
                q_chunks.append(extra_q)
                t_chunks.append(extra_t)
                pident_chunks.append(extra_aln.pidents)
                bitscore_chunks.append(extra_aln.bitscores)
                q_cov_chunks.append((extra_aln.q_ends - extra_aln.q_starts) / q_lens_e)
                t_cov_chunks.append((extra_aln.t_ends - extra_aln.t_starts) / t_lens_e)

        if not q_chunks:
            idx_arr = np.arange(n, dtype=np.int32)
            return Clusters(
                representatives=idx_arr,
                members=idx_arr.copy(),
                offsets=idx_arr.copy(),
                lengths=np.ones(n, dtype=np.int32),
            )

        all_q = np.concatenate(q_chunks)
        all_t = np.concatenate(t_chunks)
        all_pidents = np.concatenate(pident_chunks)
        all_bitscores = np.concatenate(bitscore_chunks)
        all_q_cov = np.concatenate(q_cov_chunks)
        all_t_cov = np.concatenate(t_cov_chunks)

        # 3. Filter Matches by Identity, Bitscore, Coverage, and Role-Gated Motif Rescue
        std_valid = (all_pidents >= eff_min_identity) & (all_bitscores >= eff_min_bitscore)
        if self.min_coverage > 0.0:
            std_valid &= (all_q_cov >= self.min_coverage) & (all_t_cov >= self.min_coverage)

        if emb_norm is not None and len(all_q) > 0:
            cos_all = np.sum(emb_norm[all_q] * emb_norm[all_t], axis=1)
            if roles_arr is not None:
                rq_all = roles_arr[all_q]
                rt_all = roles_arr[all_t]
                same_role = (rq_all == rt_all) & (rq_all >= 0) & (rq_all != 29)
            else:
                same_role = np.ones(len(all_q), dtype=np.bool_)
            twilight_pident = max(15.0, eff_min_identity - 10.0)
            twilight_bitscore = max(22.0, eff_min_bitscore * 0.65)
            rescued = (
                same_role
                & (cos_all >= float(self.motif_rescue_threshold))
                & (all_pidents >= twilight_pident)
                & (all_bitscores >= twilight_bitscore)
            )
            if self.min_coverage > 0.0:
                rescued &= (all_q_cov >= self.min_coverage * 0.85) & (all_t_cov >= self.min_coverage * 0.85)
            valid = std_valid | rescued
        else:
            cos_all = None
            valid = std_valid

        q_valid = all_q[valid].astype(np.int32)
        t_valid = all_t[valid].astype(np.int32)
        p_valid = all_pidents[valid]
        b_valid = all_bitscores[valid].copy()

        if cos_all is not None and len(q_valid) > 0:
            eff_alpha = float(self.embedding_weight) if self.embedding_weight > 0.0 else 0.25
            cos_pos = np.maximum(cos_all[valid], np.float32(0.0))
            max_lens_v = np.maximum(np.maximum(seqs.lengths[q_valid], seqs.lengths[t_valid]), 1).astype(np.float32)
            b_valid = b_valid + np.float32(eff_alpha) * cos_pos * max_lens_v
            p_valid = np.maximum(p_valid, p_valid + np.float32(8.0 * eff_alpha) * cos_pos)

        # 4. Clustering Execution
        if self.mode == "cdhit":
            degree = np.zeros(n, dtype=np.int32)
            np.add.at(degree, q_valid, 1)
            np.add.at(degree, t_valid, 1)
            # Sort descending by sequence length, break ties by degree descending
            order = np.lexsort((-degree, -seqs.lengths)).astype(np.int32)
            c_ids, reps, sizes = _cdhit_cluster(n, order, q_valid, t_valid)

        elif self.mode == "mcl":
            c_ids, reps, sizes = _mcl_cluster(
                n,
                seqs.lengths,
                q_valid,
                t_valid,
                b_valid,
                inflation=self.inflation,
            )

        elif self.mode == "greedy_cover":
            c_ids, reps, sizes = _greedy_cover_cluster(n, q_valid, t_valid)

        elif self.mode == "upgma":
            c_ids, reps, sizes = _upgma_cluster(n, seqs.lengths, q_valid, t_valid, p_valid, eff_min_identity)

        elif self.mode == "connected_components":
            c_ids, reps, sizes = _connected_components_cluster(n, seqs.lengths, q_valid, t_valid)

        else:
            raise RuntimeError(f"Unhandled mode: {self.mode}")

        return Clusters.from_assignments(c_ids, reps, sizes)


# Kernels & Algorithms -------------------------------------------------------------------------------------------------
@njit(nogil=True, cache=True, inline="always")
def _score_ungapped_diagonal(
    seqs_data: np.ndarray,
    s1: int,
    l1: int,
    s2: int,
    l2: int,
    d: int,
    matrix: np.ndarray,
) -> int:
    r"""Compute maximum local ungapped BLOSUM62 score along diagonal $d = i - j$."""
    i_start = max(0, d)
    j_start = i_start - d
    span = min(l1 - i_start, l2 - j_start)
    if span < 12:
        return 0
    cur = 0
    max_sc = 0
    for k in range(span):
        cur += matrix[seqs_data[s1 + i_start + k], seqs_data[s2 + j_start + k]]
        if cur < 0:
            cur = 0
        elif cur > max_sc:
            max_sc = cur
    return max_sc


@njit(parallel=True, nogil=True, cache=True)
def _ungapped_pair_prefilter(
    seqs_data: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    q_idx: np.ndarray,
    t_idx: np.ndarray,
    matrix: np.ndarray,
    min_ungapped_score: int,
) -> np.ndarray:
    r"""Filter candidate pairs `(q_idx, t_idx)` using MMseqs2-style ungapped BLOSUM62 diagonal scoring."""
    n_pairs = len(q_idx)
    keep = np.zeros(n_pairs, dtype=np.bool_)
    for p in prange(n_pairs):  # type: ignore[not-iterable]
        u = q_idx[p]
        v = t_idx[p]
        l1 = lengths[u]
        l2 = lengths[v]
        s1 = offsets[u]
        s2 = offsets[v]
        d_lim = abs(l1 - l2) + 24
        max_sc = 0
        for d in range(-d_lim, d_lim + 1):
            sc = _score_ungapped_diagonal(seqs_data, s1, l1, s2, l2, d, matrix)
            if sc > max_sc:
                max_sc = sc
                if max_sc >= min_ungapped_score:
                    keep[p] = True
                    break
    return keep


@njit(nogil=True, cache=True)
def _build_mmseqs12_k4_csr(
    seqs_data: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    lut: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Build a direct $12^4 = 20,736$-bucket CSR inverted index of `Alphabet.MMSEQS12` 4-mers."""
    n_buckets = 20736
    counts = np.zeros(n_buckets + 1, dtype=np.int32)
    n_seqs = len(lengths)
    for u in range(n_seqs):
        seq_len = lengths[u]
        s = offsets[u]
        for i in range(seq_len - 3):
            c0 = lut[seqs_data[s + i]]
            c1 = lut[seqs_data[s + i + 1]]
            c2 = lut[seqs_data[s + i + 2]]
            c3 = lut[seqs_data[s + i + 3]]
            if c0 < 12 and c1 < 12 and c2 < 12 and c3 < 12:
                counts[((c0 * 12 + c1) * 12 + c2) * 12 + c3 + 1] += 1
    indptr = np.zeros(n_buckets + 1, dtype=np.int32)
    for b in range(n_buckets):
        indptr[b + 1] = indptr[b] + counts[b + 1]
    post_seq = np.empty(indptr[n_buckets], dtype=np.int32)
    post_pos = np.empty(indptr[n_buckets], dtype=np.int16)
    cur = indptr[:-1].copy()
    for u in range(n_seqs):
        seq_len = lengths[u]
        s = offsets[u]
        for i in range(seq_len - 3):
            c0 = lut[seqs_data[s + i]]
            c1 = lut[seqs_data[s + i + 1]]
            c2 = lut[seqs_data[s + i + 2]]
            c3 = lut[seqs_data[s + i + 3]]
            if c0 < 12 and c1 < 12 and c2 < 12 and c3 < 12:
                code = ((c0 * 12 + c1) * 12 + c2) * 12 + c3
                idx = cur[code]
                post_seq[idx] = u
                post_pos[idx] = i
                cur[code] = idx + 1
    return indptr, post_seq, post_pos


@njit(parallel=True, nogil=True, cache=True)
def _mmseqs12_k4_ungapped_prefilter(
    seqs_data: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    min_cov: float,
    min_ungapped_score: int,
    lut: np.ndarray,
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""MMseqs2-style reduced-alphabet 4-mer CSR + ungapped BLOSUM62 diagonal prefilter."""
    indptr, post_seq, post_pos = _build_mmseqs12_k4_csr(seqs_data, offsets, lengths, lut)
    n_seqs = len(lengths)
    row_counts = np.zeros(n_seqs, dtype=np.int32)
    for u in prange(n_seqs):  # type: ignore[not-iterable]
        l1 = lengths[u]
        s1 = offsets[u]
        best_sc = np.zeros(n_seqs, dtype=np.int16)
        last_diag = np.full(n_seqs, 32767, dtype=np.int16)
        for i in range(l1 - 3):
            c0 = lut[seqs_data[s1 + i]]
            c1 = lut[seqs_data[s1 + i + 1]]
            c2 = lut[seqs_data[s1 + i + 2]]
            c3 = lut[seqs_data[s1 + i + 3]]
            if c0 < 12 and c1 < 12 and c2 < 12 and c3 < 12:
                code = ((c0 * 12 + c1) * 12 + c2) * 12 + c3
                for idx in range(indptr[code], indptr[code + 1]):
                    v = post_seq[idx]
                    if v <= u:
                        continue
                    l2 = lengths[v]
                    if min_cov > 0.0 and min(l1, l2) < min_cov * max(l1, l2):
                        continue
                    d = i - int(post_pos[idx])
                    if d == last_diag[v] or best_sc[v] >= min_ungapped_score:
                        continue
                    last_diag[v] = d
                    sc = _score_ungapped_diagonal(seqs_data, s1, l1, offsets[v], l2, d, matrix)
                    if sc > best_sc[v]:
                        best_sc[v] = min(sc, 32760)
        cnt = 0
        for v in range(u + 1, n_seqs):
            if best_sc[v] >= min_ungapped_score:
                cnt += 1
        row_counts[u] = cnt

    out_offsets = np.zeros(n_seqs + 1, dtype=np.int32)
    for u in range(n_seqs):
        out_offsets[u + 1] = out_offsets[u] + row_counts[u]
    total_hits = out_offsets[n_seqs]
    out_q = np.empty(total_hits, dtype=np.int32)
    out_t = np.empty(total_hits, dtype=np.int32)

    for u in prange(n_seqs):  # type: ignore[not-iterable]
        if row_counts[u] == 0:
            continue
        l1 = lengths[u]
        s1 = offsets[u]
        best_sc = np.zeros(n_seqs, dtype=np.int16)
        last_diag = np.full(n_seqs, 32767, dtype=np.int16)
        for i in range(l1 - 3):
            c0 = lut[seqs_data[s1 + i]]
            c1 = lut[seqs_data[s1 + i + 1]]
            c2 = lut[seqs_data[s1 + i + 2]]
            c3 = lut[seqs_data[s1 + i + 3]]
            if c0 < 12 and c1 < 12 and c2 < 12 and c3 < 12:
                code = ((c0 * 12 + c1) * 12 + c2) * 12 + c3
                for idx in range(indptr[code], indptr[code + 1]):
                    v = post_seq[idx]
                    if v <= u:
                        continue
                    l2 = lengths[v]
                    if min_cov > 0.0 and min(l1, l2) < min_cov * max(l1, l2):
                        continue
                    d = i - int(post_pos[idx])
                    if d == last_diag[v] or best_sc[v] >= min_ungapped_score:
                        continue
                    last_diag[v] = d
                    sc = _score_ungapped_diagonal(seqs_data, s1, l1, offsets[v], l2, d, matrix)
                    if sc > best_sc[v]:
                        best_sc[v] = min(sc, 32760)
        w = out_offsets[u]
        for v in range(u + 1, n_seqs):
            if best_sc[v] >= min_ungapped_score:
                out_q[w] = u
                out_t[w] = v
                w += 1
    return out_q, out_t


@njit(nogil=True, cache=True)
def _pack_members(
    cluster_ids: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    r"""Scatter sequence indices `[0..N-1]` into a flat CSR-style `members` array in $O(N)$ time."""
    n_seqs = len(cluster_ids)
    members = np.empty(n_seqs, dtype=np.int32)
    cur = offsets.copy()
    for i in range(n_seqs):
        c = cluster_ids[i]
        members[cur[c]] = i
        cur[c] += 1
    return members


@njit(nogil=True, cache=True)
def _cdhit_cluster(
    n_seqs: int,
    order: np.ndarray,
    q_indices: np.ndarray,
    t_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Numba kernel for length-prioritized greedy incremental clustering (CD-HIT style)."""
    n_edges = len(q_indices)
    degree = np.zeros(n_seqs, dtype=np.int32)
    for p in range(n_edges):
        degree[q_indices[p]] += 1
        degree[t_indices[p]] += 1

    offsets = np.zeros(n_seqs + 1, dtype=np.int32)
    for i in range(n_seqs):
        offsets[i + 1] = offsets[i] + degree[i]

    adj = np.empty(offsets[n_seqs], dtype=np.int32)
    cur_pos = offsets[:-1].copy()
    for p in range(n_edges):
        u = q_indices[p]
        v = t_indices[p]
        adj[cur_pos[u]] = v
        cur_pos[u] += 1
        adj[cur_pos[v]] = u
        cur_pos[v] += 1

    cluster_ids = np.full(n_seqs, -1, dtype=np.int32)
    reps_list: list[int] = []
    sizes_list: list[int] = []

    for idx in range(n_seqs):
        i = order[idx]
        if cluster_ids[i] != -1:
            continue
        c_id = len(reps_list)
        reps_list.append(i)
        cluster_ids[i] = c_id
        cluster_size = 1

        for edge_idx in range(offsets[i], offsets[i + 1]):
            neighbor = adj[edge_idx]
            if cluster_ids[neighbor] == -1:
                cluster_ids[neighbor] = c_id
                cluster_size += 1

        sizes_list.append(cluster_size)

    reps_arr = np.empty(len(reps_list), dtype=np.int32)
    for k in range(len(reps_list)):
        reps_arr[k] = reps_list[k]

    sizes_arr = np.empty(len(sizes_list), dtype=np.int32)
    for k in range(len(sizes_list)):
        sizes_arr[k] = sizes_list[k]

    return cluster_ids, reps_arr, sizes_arr


@njit(nogil=True, cache=True)
def _greedy_cover_cluster(
    n_seqs: int,
    q_indices: np.ndarray,
    t_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Numba kernel for iterative maximum uncovered degree greedy set cover."""
    n_edges = len(q_indices)
    degree = np.zeros(n_seqs, dtype=np.int32)
    for p in range(n_edges):
        degree[q_indices[p]] += 1
        degree[t_indices[p]] += 1

    offsets = np.zeros(n_seqs + 1, dtype=np.int32)
    for i in range(n_seqs):
        offsets[i + 1] = offsets[i] + degree[i]

    adj = np.empty(offsets[n_seqs], dtype=np.int32)
    cur_pos = offsets[:-1].copy()
    for p in range(n_edges):
        u = q_indices[p]
        v = t_indices[p]
        adj[cur_pos[u]] = v
        cur_pos[u] += 1
        adj[cur_pos[v]] = u
        cur_pos[v] += 1

    cluster_ids = np.full(n_seqs, -1, dtype=np.int32)
    reps_list: list[int] = []
    sizes_list: list[int] = []

    uncovered = np.ones(n_seqs, dtype=np.int32) + degree
    unassigned_count = n_seqs

    while unassigned_count > 0:
        best_u = -1
        max_cov = -1
        for i in range(n_seqs):
            if cluster_ids[i] == -1 and uncovered[i] > max_cov:
                max_cov = uncovered[i]
                best_u = i

        if best_u == -1:
            break

        c_id = len(reps_list)
        reps_list.append(best_u)
        cluster_ids[best_u] = c_id
        cluster_size = 1
        unassigned_count -= 1

        for edge_idx in range(offsets[best_u], offsets[best_u + 1]):
            nbr = adj[edge_idx]
            uncovered[nbr] -= 1

        for edge_idx in range(offsets[best_u], offsets[best_u + 1]):
            nbr = adj[edge_idx]
            if cluster_ids[nbr] == -1:
                cluster_ids[nbr] = c_id
                cluster_size += 1
                unassigned_count -= 1
                for e2 in range(offsets[nbr], offsets[nbr + 1]):
                    w = adj[e2]
                    uncovered[w] -= 1

        sizes_list.append(cluster_size)

    reps_arr = np.empty(len(reps_list), dtype=np.int32)
    for k in range(len(reps_list)):
        reps_arr[k] = reps_list[k]

    sizes_arr = np.empty(len(sizes_list), dtype=np.int32)
    for k in range(len(sizes_list)):
        sizes_arr[k] = sizes_list[k]

    return cluster_ids, reps_arr, sizes_arr


def _upgma_cluster(
    n_seqs: int,
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    q_indices: np.ndarray[tuple[int], np.dtype[np.int32]],
    t_indices: np.ndarray[tuple[int], np.dtype[np.int32]],
    pidents: np.ndarray[Any, np.dtype[np.floating[Any]]],
    min_identity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Hierarchical average-linkage (UPGMA) clustering via SciPy."""
    if n_seqs <= 1:
        return np.zeros(n_seqs, dtype=np.int32), np.arange(n_seqs, dtype=np.int32), np.array([n_seqs], dtype=np.int32)

    total_pairs = n_seqs * (n_seqs - 1) // 2
    condensed_d = np.ones(total_pairs, dtype=np.float64)

    if len(q_indices) > 0:
        q = q_indices.astype(np.int64)
        t = t_indices.astype(np.int64)
        swap = q > t
        if np.any(swap):
            q_copy = q.copy()
            q[swap] = t[swap]
            t[swap] = q_copy[swap]

        indices = n_seqs * q - (q * (q + 1)) // 2 + t - q - 1
        dists = np.clip(1.0 - (pidents / 100.0), 0.0, 1.0)
        condensed_d[indices] = dists

    z = linkage(condensed_d, method="average")
    cutoff = max(1e-6, 1.0 - (min_identity / 100.0))
    raw_clusters = fcluster(z, t=cutoff, criterion="distance")

    unique_c, inv = np.unique(raw_clusters, return_inverse=True)
    cluster_ids = inv.astype(np.int32)
    n_clusters = len(unique_c)

    reps = np.empty(n_clusters, dtype=np.int32)
    sizes = np.bincount(cluster_ids, minlength=n_clusters).astype(np.int32)

    order = np.argsort(-lengths)
    seen = np.zeros(n_clusters, dtype=np.bool_)
    for idx in order:
        c = cluster_ids[idx]
        if not seen[c]:
            seen[c] = True
            reps[c] = idx

    return cluster_ids, reps, sizes


def _connected_components_cluster(
    n_seqs: int,
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    q_indices: np.ndarray[tuple[int], np.dtype[np.int32]],
    t_indices: np.ndarray[tuple[int], np.dtype[np.int32]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Transitive closure connected components clustering via SciPy."""
    if n_seqs <= 1:
        return np.zeros(n_seqs, dtype=np.int32), np.arange(n_seqs, dtype=np.int32), np.array([n_seqs], dtype=np.int32)

    if len(q_indices) == 0:
        return np.arange(n_seqs, dtype=np.int32), np.arange(n_seqs, dtype=np.int32), np.ones(n_seqs, dtype=np.int32)

    data = np.ones(len(q_indices), dtype=np.int8)
    adj = csr_array((data, (q_indices, t_indices)), shape=(n_seqs, n_seqs))
    n_components, labels = csgraph.connected_components(adj, directed=False)

    cluster_ids = labels.astype(np.int32)
    sizes = np.bincount(cluster_ids, minlength=n_components).astype(np.int32)

    order = np.argsort(-lengths)
    reps = np.empty(n_components, dtype=np.int32)
    seen = np.zeros(n_components, dtype=np.bool_)
    for idx in order:
        c = cluster_ids[idx]
        if not seen[c]:
            seen[c] = True
            reps[c] = idx

    return cluster_ids, reps, sizes


def _mcl_cluster(
    n_seqs: int,
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    q_indices: np.ndarray[tuple[int], np.dtype[np.int32]],
    t_indices: np.ndarray[tuple[int], np.dtype[np.int32]],
    bitscores: np.ndarray[Any, np.dtype[np.floating[Any]]],
    inflation: float = 2.0,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Markov Cluster Algorithm (MCL) on length-normalized alignment bitscores per connected component."""
    if n_seqs <= 1:
        return np.zeros(n_seqs, dtype=np.int32), np.arange(n_seqs, dtype=np.int32), np.array([n_seqs], dtype=np.int32)

    if len(q_indices) == 0:
        return np.arange(n_seqs, dtype=np.int32), np.arange(n_seqs, dtype=np.int32), np.ones(n_seqs, dtype=np.int32)

    # Normalize edge weights by maximum sequence length of the aligned pair (bitscore per residue)
    max_lens = np.maximum(np.maximum(lengths[q_indices], lengths[t_indices]), 1).astype(np.float64)
    norm_weights = np.maximum(bitscores.astype(np.float64) / max_lens, 1e-4)

    # Decompose graph into connected components so MCL runs on compact local subgraphs
    ones_data = np.ones(len(q_indices), dtype=np.int8)
    comp_adj = csr_array((ones_data, (q_indices, t_indices)), shape=(n_seqs, n_seqs))
    n_comp, comp_labels = csgraph.connected_components(comp_adj, directed=False)

    # Pre-group edges by connected component
    edge_comp = comp_labels[q_indices]
    edge_order = np.argsort(edge_comp, kind="mergesort")
    sorted_ec = edge_comp[edge_order]
    ec_starts = np.searchsorted(sorted_ec, np.arange(n_comp), side="left")
    ec_ends = np.searchsorted(sorted_ec, np.arange(n_comp), side="right")

    comp_node_order = np.argsort(comp_labels, kind="mergesort")
    sorted_nc = comp_labels[comp_node_order]
    nc_starts = np.searchsorted(sorted_nc, np.arange(n_comp), side="left")
    nc_ends = np.searchsorted(sorted_nc, np.arange(n_comp), side="right")

    cluster_ids = np.full(n_seqs, -1, dtype=np.int32)
    reps_list: list[int] = []
    sizes_list: list[int] = []
    global_local_map = np.empty(n_seqs, dtype=np.int32)
    infl = max(float(inflation), 1.05)

    for c in range(n_comp):
        nodes_c = comp_node_order[nc_starts[c] : nc_ends[c]]
        m = len(nodes_c)
        if m == 1:
            g_u = int(nodes_c[0])
            cid = len(reps_list)
            cluster_ids[g_u] = cid
            reps_list.append(g_u)
            sizes_list.append(1)
            continue

        if m == 2:
            g0, g1 = int(nodes_c[0]), int(nodes_c[1])
            cid = len(reps_list)
            cluster_ids[g0] = cid
            cluster_ids[g1] = cid
            rep = g0 if int(lengths[g0]) >= int(lengths[g1]) else g1
            reps_list.append(rep)
            sizes_list.append(2)
            continue

        global_local_map[nodes_c] = np.arange(m, dtype=np.int32)
        e_slice = edge_order[ec_starts[c] : ec_ends[c]]
        loc_u = global_local_map[q_indices[e_slice]]
        loc_v = global_local_map[t_indices[e_slice]]
        w_e = norm_weights[e_slice]

        sim = np.zeros((m, m), dtype=np.float64)
        np.maximum.at(sim, (loc_u, loc_v), w_e)
        np.maximum.at(sim, (loc_v, loc_u), w_e)

        row_max = np.max(sim, axis=1)
        np.fill_diagonal(sim, np.maximum(row_max, 1e-3))
        deg_weights = sim.sum(axis=1) - np.diag(sim)

        col_sums = sim.sum(axis=0)
        col_sums[col_sums == 0.0] = 1.0
        mat = sim / col_sums

        for _ in range(max_iter):
            prev = mat
            mat = mat @ mat
            np.power(mat, infl, out=mat)
            mat[mat < 1e-5] = 0.0
            c_sums = mat.sum(axis=0)
            c_sums[c_sums == 0.0] = 1.0
            mat /= c_sums
            if float(np.max(np.abs(mat - prev))) < tol:
                break

        attr = (mat > 1e-3).astype(np.int32)
        shared_attr = (attr.T @ attr) > 0
        sub_adj = csr_array(shared_attr | (mat > 1e-3) | (mat.T > 1e-3))
        n_sub, sub_labels = csgraph.connected_components(sub_adj, directed=False)

        for s_id in range(n_sub):
            sub_local = np.where(sub_labels == s_id)[0]
            sub_global = nodes_c[sub_local]
            cid = len(reps_list)
            cluster_ids[sub_global] = cid
            sizes_list.append(len(sub_global))
            # Pick representative with highest bitscore degree, tie-breaking by sequence length
            best_local = sub_local[int(np.lexsort((-lengths[sub_global], -deg_weights[sub_local]))[0])]
            reps_list.append(int(nodes_c[best_local]))

    return (
        cluster_ids,
        np.array(reps_list, dtype=np.int32),
        np.array(sizes_list, dtype=np.int32),
    )
