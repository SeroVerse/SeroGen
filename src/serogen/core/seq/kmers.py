r"""K-mer indexing, FracMinHash, Randstrobe sketch construction, and alignment seed management.

This module provides sequence sketch indices ([`FracMinHashIndex`][serogen.core.seq.kmers.FracMinHashIndex],
[`RandstrobeIndex`][serogen.core.seq.kmers.RandstrobeIndex]), alignment seed batch containers
([`Seeds`][serogen.core.seq.kmers.Seeds]), and Numba JIT parallel search kernels.
"""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, Self, overload

import numpy as np
from numba import njit, prange

from serogen.core.collections import BatchedContainer, RaggedArrayContainer
from serogen.core.seq.alphabet import Alphabet
from serogen.core.seq.seq import Sequences


# Functions ------------------------------------------------------------------------------------------------------------
def _compute_offsets_from_counts(
    counts: np.ndarray[tuple[int], np.dtype[np.integer]],
) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
    r"""Compute cumulative start offsets from a count/length array."""
    n = len(counts)
    offsets = np.zeros(n, dtype=np.int32)
    if n > 1:
        c = counts[:-1] if counts.dtype == np.int32 else counts[:-1].astype(np.int32)
        np.cumsum(c, out=offsets[1:])
    return offsets


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Seed:
    r"""Alignment seed representing a potential matching region between query and target.

    Attributes:
        query_index (int): Index of the query sequence.
        target_index (int): Index of the target sequence.
        score (int): Match score (number of shared k-mer hashes or randstrobes).
        offset (int): Diagonal offset calculated as `query_pos - target_pos`.
    """

    query_index: int
    target_index: int
    score: int
    offset: int


@dataclass(frozen=True, slots=True)
class Seeds(BatchedContainer[Seed, "Seeds"]):
    r"""Structure-of-Arrays (SoA) batch container for alignment seeds.

    Stores query indices, target indices, scores, and diagonal offsets as 1D NumPy arrays.

    Attributes:
        query_indices (np.ndarray[tuple[int], np.dtype[np.uint32]]): 1D array of query sequence indices.
        target_indices (np.ndarray[tuple[int], np.dtype[np.uint32]]): 1D array of target sequence indices.
        scores (np.ndarray[tuple[int], np.dtype[np.uint32]]): 1D array of alignment match scores.
        offsets (np.ndarray[tuple[int], np.dtype[np.int32]]): 1D array of diagonal offset values.
    """

    query_indices: np.ndarray[tuple[int], np.dtype[np.uint32]]
    target_indices: np.ndarray[tuple[int], np.dtype[np.uint32]]
    scores: np.ndarray[tuple[int], np.dtype[np.uint32]]
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]]

    def __len__(self) -> int:
        r"""Return the number of seeds in the batch.

        Returns:
            int: Total seed count.
        """
        return len(self.query_indices)

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> Seed: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> Seeds: ...

    @overload
    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> Seed | Seeds: ...

    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> Seed | Seeds:
        r"""Access seeds by integer index, slice, or boolean/integer NumPy array.

        Args:
            item (Any): Integer index, slice, or mask array.

        Returns:
            Seed | Seeds: A single [`Seed`][serogen.core.seq.kmers.Seed] or a
                sliced [`Seeds`][serogen.core.seq.kmers.Seeds] collection.

        Raises:
            IndexError: If integer index is out of bounds.
        """
        if isinstance(item, (int, np.integer)):
            idx_val = int(item)
            if idx_val < 0:
                idx_val += len(self)
            if idx_val < 0 or idx_val >= len(self):
                raise IndexError("Batch index out of range")
            return Seed(
                int(self.query_indices[idx_val]),
                int(self.target_indices[idx_val]),
                int(self.scores[idx_val]),
                int(self.offsets[idx_val]),
            )

        if isinstance(item, slice):
            indices = np.arange(len(self))[item]
        else:
            item_arr = np.asarray(item)
            indices = np.nonzero(item_arr)[0] if item_arr.dtype.kind == "b" else item_arr

        return Seeds(
            self.query_indices[indices],
            self.target_indices[indices],
            self.scores[indices],
            self.offsets[indices],
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        r"""Concatenate multiple [`Seeds`][serogen.core.seq.kmers.Seeds] collections into a single batch.

        Args:
            batches (Iterable[Seeds]): Iterable of [`Seeds`][serogen.core.seq.kmers.Seeds] objects.

        Returns:
            Seeds: Concatenated [`Seeds`][serogen.core.seq.kmers.Seeds] collection.
        """
        batches_list = list(batches)
        if not batches_list:
            return cls.empty()
        return cls(
            np.concatenate([b.query_indices for b in batches_list]),
            np.concatenate([b.target_indices for b in batches_list]),
            np.concatenate([b.scores for b in batches_list]),
            np.concatenate([b.offsets for b in batches_list]),
        )

    @classmethod
    def empty(cls) -> Self:
        r"""Create an empty [`Seeds`][serogen.core.seq.kmers.Seeds] collection.

        Returns:
            Seeds: Empty [`Seeds`][serogen.core.seq.kmers.Seeds] collection.
        """
        return cls(
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.int32),
        )

    def filter(self, mask: np.ndarray[tuple[int], np.dtype[np.bool_]]) -> Seeds:
        r"""Return a new Seeds collection containing only records where the mask is True.

        Args:
            mask (np.ndarray[tuple[int], np.dtype[np.bool_]]): 1D boolean mask array.

        Returns:
            Seeds: Filtered [`Seeds`][serogen.core.seq.kmers.Seeds] collection.
        """
        return Seeds(
            self.query_indices[mask],
            self.target_indices[mask],
            self.scores[mask],
            self.offsets[mask],
        )

    def top_hits(self, min_score: int = 1) -> Seeds:
        r"""Reduce the batch to only the highest-scoring seed for each query.

        Args:
            min_score (int): Minimum match score threshold. Defaults to 1.

        Returns:
            Seeds: Filtered [`Seeds`][serogen.core.seq.kmers.Seeds] collection containing top hit per query.
        """
        if len(self) == 0:
            return self

        order = np.lexsort((~self.scores, self.query_indices))
        _, unique_idx = np.unique(self.query_indices[order], return_index=True)
        best_idx = order[unique_idx]
        best_idx.sort()

        best_batch = Seeds(
            self.query_indices[best_idx],
            self.target_indices[best_idx],
            self.scores[best_idx],
            self.offsets[best_idx],
        )

        if min_score > 0:
            return best_batch.filter(best_batch.scores >= min_score)

        return best_batch

    def extract_sequences(self, queries: Sequences, targets: Sequences) -> tuple[Sequences, Sequences]:
        r"""Extract parallel batches of query and target sequences mapped by this seed batch.

        Args:
            queries (Sequences): Source collection of query sequences.
            targets (Sequences): Source collection of target sequences.

        Returns:
            tuple[Sequences, Sequences]: Paired query and target sequence collections.
        """
        return queries[self.query_indices], targets[self.target_indices]


@dataclass(frozen=True, slots=True, eq=False)
class KmerRecords:
    r"""Single-sequence sketch records slice holding parallel 1D arrays.

    Attributes:
        hashes (np.ndarray[tuple[int], np.dtype[np.uint64]]): 1D array of 64-bit sketch hashes.
        seq_indices (np.ndarray[tuple[int], np.dtype[np.uint32]]): 1D array of sequence indices.
        positions (np.ndarray[tuple[int], np.dtype[np.uint32]]): 1D array of sequence positions.
    """

    hashes: np.ndarray[tuple[int], np.dtype[np.uint64]]
    seq_indices: np.ndarray[tuple[int], np.dtype[np.uint32]]
    positions: np.ndarray[tuple[int], np.dtype[np.uint32]]

    def __len__(self) -> int:
        r"""Return the number of k-mer sketch records in this view."""
        return len(self.hashes)

    def __eq__(self, other: object) -> bool:
        r"""Compare two KmerRecords views for exact array equality."""
        if not isinstance(other, KmerRecords):
            return NotImplemented
        return bool(
            np.array_equal(self.hashes, other.hashes)
            and np.array_equal(self.seq_indices, other.seq_indices)
            and np.array_equal(self.positions, other.positions)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class BaseKmerIndex(RaggedArrayContainer[KmerRecords, "BaseKmerIndex"]):
    r"""Data-Oriented Structure-of-Arrays (SoA) container for sequence k-mer sketches.

    Attributes:
        hashes (np.ndarray[tuple[int], np.dtype[np.uint64]]): 1D array of 64-bit sketch hashes.
        seq_indices (np.ndarray[tuple[int], np.dtype[np.uint32]]): 1D array of source sequence indices.
        positions (np.ndarray[tuple[int], np.dtype[np.uint32]]): 1D array of start positions in sequences.
        offsets (np.ndarray[tuple[int], np.dtype[np.int32]]): 1D array of sequence record start offsets.
        lengths (np.ndarray[tuple[int], np.dtype[np.int32]]): 1D array of record counts per sequence.
        n_seqs (int): Total number of indexed sequences.
        is_sorted (bool): True if records are sorted by hash.
        k (int): K-mer length parameter.
        alphabet (Alphabet): Alphabet used for sequence encoding.
        reduce (bool): True if using a reduced alphabet mapping.
    """

    hashes: np.ndarray[tuple[int], np.dtype[np.uint64]]
    seq_indices: np.ndarray[tuple[int], np.dtype[np.uint32]]
    positions: np.ndarray[tuple[int], np.dtype[np.uint32]]
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]]
    n_seqs: int = 0
    is_sorted: bool = False
    k: int = 10
    alphabet: Alphabet = Alphabet.DNA
    reduce: bool = False

    def __len__(self) -> int:
        r"""Return the number of sequences in the batch container."""
        return self.n_seqs

    @property
    def total_records(self) -> int:
        r"""Total number of k-mer / sketch records across all sequences."""
        return len(self.hashes)

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> KmerRecords: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> Self: ...

    @overload
    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> KmerRecords | Self: ...

    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> KmerRecords | Self:
        r"""Access records for an individual sequence, or slice/mask a sub-index of sequences."""
        if isinstance(item, (int, np.integer)):
            idx_val = int(item)
            if idx_val < 0:
                idx_val += self.n_seqs
            if idx_val < 0 or idx_val >= self.n_seqs:
                raise IndexError("Sequence index out of range")
            start = int(self.offsets[idx_val])
            end = start + int(self.lengths[idx_val])
            return KmerRecords(
                self.hashes[start:end],
                self.seq_indices[start:end],
                self.positions[start:end],
            )

        indices, new_offsets, new_lengths, (new_hashes, new_positions) = self._slice_ragged(
            item, self.hashes, self.positions
        )
        if len(indices) == 0:
            return self.empty()

        new_seq_indices = np.repeat(np.arange(len(indices), dtype=np.uint32), new_lengths)

        return replace(
            self,
            hashes=new_hashes,
            seq_indices=new_seq_indices,
            positions=new_positions,
            offsets=new_offsets,
            lengths=new_lengths,
            n_seqs=len(indices),
            is_sorted=False,
        )

    @classmethod
    def empty(cls) -> Self:
        r"""Create an empty index instance."""
        raise NotImplementedError

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        r"""Concatenate multiple index batches into a single batch."""
        batches_list = list(batches)
        if not batches_list:
            return cls.empty()

        total_seqs = sum(b.n_seqs for b in batches_list)
        all_lengths = np.concatenate([b.lengths for b in batches_list])
        all_offsets = cls.lengths_to_offsets(all_lengths)
        all_hashes = np.concatenate([b.hashes for b in batches_list])
        all_positions = np.concatenate([b.positions for b in batches_list])

        seq_idx_list = []
        seq_offset = 0
        for b in batches_list:
            if len(b.seq_indices) > 0:
                seq_idx_list.append(b.seq_indices + np.uint32(seq_offset))
            seq_offset += b.n_seqs
        all_seq_indices = np.concatenate(seq_idx_list) if seq_idx_list else np.empty(0, dtype=np.uint32)

        first = batches_list[0]
        return replace(
            first,
            hashes=all_hashes,
            seq_indices=all_seq_indices,
            positions=all_positions,
            offsets=all_offsets,
            lengths=all_lengths,
            n_seqs=total_seqs,
            is_sorted=False,
        )

    def to_sorted(self) -> Self:
        r"""Return a copy of the index with records sorted by hash."""
        if self.is_sorted:
            return self
        sorted_h, sorted_s, sorted_p = _radix_sort_soa(self.hashes, self.seq_indices, self.positions)
        return replace(
            self,
            hashes=sorted_h,
            seq_indices=sorted_s,
            positions=sorted_p,
            is_sorted=True,
        )

    def _build_queries(self, queries: Sequences) -> Self:
        r"""Build query index with matching parameters."""
        raise NotImplementedError

    def _prep_queries(self, queries: BaseKmerIndex | Sequences) -> BaseKmerIndex:
        r"""Prepare and validate query index against target index state."""
        if not self.is_sorted:
            raise ValueError("Target index must be sorted by hash for binary search. Build it with sort_by_hash=True.")

        if isinstance(queries, Sequences):
            queries = self._build_queries(queries)

        if queries.is_sorted:
            raise ValueError("Query index must NOT be sorted by hash. Build it with sort_by_hash=False.")

        return queries

    def _prep_self_hits(
        self,
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.uint64]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint64]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
    ]:
        r"""Prepare query and target records for self-hits."""
        if self.is_sorted:
            order = np.argsort(self.seq_indices, kind="stable")
            q_hashes = self.hashes[order]
            q_positions = self.positions[order]
            q_seq_indices = self.seq_indices[order]
            q_offsets = _compute_query_offsets(q_seq_indices, self.n_seqs)
            return q_hashes, q_positions, self.hashes, self.seq_indices, self.positions, q_offsets
        else:
            q_offsets = self.indptr.astype(np.uint32)
            t_hashes, t_seq_indices, t_positions = _radix_sort_soa(self.hashes, self.seq_indices, self.positions)
            return self.hashes, self.positions, t_hashes, t_seq_indices, t_positions, q_offsets

    def top_hits(self, queries: BaseKmerIndex | Sequences | None = None, min_score: int = 1) -> Seeds:
        r"""Find the single best-matching target sequence for each query sequence.

        Args:
            queries (BaseKmerIndex | Sequences | None): Query sequences. If None, queries itself.
            min_score (int): Minimum match score threshold. Defaults to 1.

        Returns:
            Seeds: Matched [`Seeds`][serogen.core.seq.kmers.Seeds] collection.
        """
        if (queries is not None and len(queries) == 0) or len(self.hashes) == 0:
            return Seeds.empty()

        if queries is None:
            q_hashes, q_positions, t_hashes, t_seq_indices, t_positions, q_offsets = self._prep_self_hits()
            num_queries = self.n_seqs
        else:
            queries_idx = self._prep_queries(queries)
            q_hashes = queries_idx.hashes
            q_positions = queries_idx.positions
            t_hashes = self.hashes
            t_seq_indices = self.seq_indices
            t_positions = self.positions
            num_queries = queries_idx.n_seqs
            q_offsets = queries_idx.indptr.astype(np.uint32)

        seeds = Seeds(
            *_intersect_top_hit_kernel(
                q_hashes,
                q_positions,
                q_offsets,
                num_queries,
                t_hashes,
                t_seq_indices,
                t_positions,
                self.n_seqs,
            )
        )
        if min_score > 0:
            return seeds.filter(seeds.scores >= min_score)
        return seeds

    def hits(self, queries: BaseKmerIndex | Sequences, min_score: int = 1) -> Seeds:
        r"""Find all matching target sequences for each query sequence above a minimum score.

        Args:
            queries (BaseKmerIndex | Sequences): Query sequence index or raw sequences.
            min_score (int): Minimum match score threshold. Defaults to 1.

        Returns:
            Seeds: Matched [`Seeds`][serogen.core.seq.kmers.Seeds] collection containing all matches.
        """
        if len(queries) == 0 or len(self.hashes) == 0:
            return Seeds.empty()

        queries_idx = self._prep_queries(queries)
        num_queries = queries_idx.n_seqs
        q_offsets = queries_idx.indptr.astype(np.uint32)

        counts = _count_hits_kernel(
            queries_idx.hashes,
            queries_idx.positions,
            q_offsets,
            num_queries,
            self.hashes,
            self.seq_indices,
            self.positions,
            self.n_seqs,
            min_score,
        )
        total_hits = int(np.sum(counts))
        if total_hits == 0:
            return Seeds.empty()

        out_offsets = _compute_offsets_from_counts(counts)
        out_q = np.empty(total_hits, dtype=np.uint32)
        out_t = np.empty(total_hits, dtype=np.uint32)
        out_s = np.empty(total_hits, dtype=np.uint32)
        out_o = np.empty(total_hits, dtype=np.int32)

        _populate_hits_kernel(
            queries_idx.hashes,
            queries_idx.positions,
            q_offsets,
            num_queries,
            self.hashes,
            self.seq_indices,
            self.positions,
            self.n_seqs,
            min_score,
            out_offsets.astype(np.uint32),
            out_q,
            out_t,
            out_s,
            out_o,
        )
        return Seeds(out_q, out_t, out_s, out_o)


@dataclass(frozen=True, slots=True, kw_only=True)
class FracMinHashIndex(BaseKmerIndex):
    r"""Specialized index for fast sequence comparisons using FracMinHash.

    Attributes:
        scaled (int): FracMinHash scale factor (e.g. 100).
        canonical (bool): True if canonical k-mers (min of fwd/rev) are hashed.
        bits_per_char (int): Alphabet bits per character.
        lut (np.ndarray[tuple[int], np.dtype[np.uint8]] | None): Optional custom lookup table.
    """

    scaled: int = 100
    canonical: bool = True
    bits_per_char: int = 2
    lut: np.ndarray[tuple[int], np.dtype[np.uint8]] | None = None

    @classmethod
    def empty(cls) -> Self:
        r"""Create an empty FracMinHashIndex."""
        return cls(
            hashes=np.empty(0, dtype=np.uint64),
            seq_indices=np.empty(0, dtype=np.uint32),
            positions=np.empty(0, dtype=np.uint32),
            offsets=np.empty(0, dtype=np.int32),
            lengths=np.empty(0, dtype=np.int32),
            n_seqs=0,
            is_sorted=False,
            k=21,
            alphabet=Alphabet.DNA,
            scaled=100,
            canonical=True,
            bits_per_char=2,
            reduce=False,
            lut=None,
        )

    @classmethod
    def build(
        cls,
        batch: Sequences,
        k: int | None = None,
        alphabet: Alphabet | str = Alphabet.DNA,
        scaled: int = 100,
        canonical: bool | None = None,
        seed: int = 42,
        sort_by_hash: bool = False,
        lut: np.ndarray[tuple[int], np.dtype[np.uint8]] | None = None,
        bits_per_char: int | None = None,
        reduce: bool | None = None,
        **kwargs: object,
    ) -> FracMinHashIndex:
        r"""Build a FracMinHashIndex from sequence batch.

        Args:
            batch (Sequences): Input sequence collection.
            k (int | None): K-mer length. Defaults to alphabet default_k.
            alphabet (Alphabet | str): Alphabet instance or name. Defaults to Alphabet.DNA.
            scaled (int): Sampling scale factor. Defaults to 100.
            canonical (bool | None): True if using canonical k-mers. Defaults to alphabet.supports_canonical.
            seed (int): Seed for hashing. Defaults to 42.
            sort_by_hash (bool): True to sort output records by hash. Defaults to False.
            lut (np.ndarray | None): Optional custom lookup table.
            bits_per_char (int | None): Bits per character. Defaults to alphabet.bits_per_char.
            reduce (bool | None): True to reduce alphabet.
            **kwargs (object): Additional parameters.

        Returns:
            FracMinHashIndex: Built [`FracMinHashIndex`][serogen.core.seq.kmers.FracMinHashIndex] instance.
        """
        if isinstance(alphabet, str):
            alphabet_obj = (
                Alphabet.MMSEQS12
                if (reduce or alphabet == "mmseqs12")
                else (Alphabet.AMINO if alphabet in ("amino", "amino_acid") else Alphabet.DNA)
            )
        else:
            alphabet_obj = alphabet

        if len(batch) == 0:
            return cls.empty()

        n_seqs = len(batch)
        if canonical is True and not alphabet_obj.supports_canonical:
            raise ValueError(f"Canonical k-mers are not supported for alphabet {alphabet_obj}.")

        resolved_canonical = alphabet_obj.supports_canonical if canonical is None else canonical
        resolved_k = alphabet_obj.default_k if k is None else k
        resolved_bits = alphabet_obj.bits_per_char if bits_per_char is None else bits_per_char
        resolved_reduce = alphabet_obj is Alphabet.MMSEQS12 if reduce is None else reduce

        if resolved_k * resolved_bits > 64:
            raise ValueError(
                f"k-mer size ({resolved_k}) with {resolved_bits} bits/char exceeds 64-bit integer capacity."
            )

        kernel_lut = lut if lut is not None else alphabet_obj.kmer_lut

        counts = _count_fracminhash_kernel(
            batch.seqs, batch.offsets, batch.lengths, kernel_lut, resolved_k, scaled, resolved_canonical, resolved_bits
        )
        total_hashes = int(np.sum(counts))
        if total_hashes == 0:
            return cls.empty()

        out_offsets = _compute_offsets_from_counts(counts)
        out_hashes = np.empty(total_hashes, dtype=np.uint64)
        out_seq_indices = np.empty(total_hashes, dtype=np.uint32)
        out_positions = np.empty(total_hashes, dtype=np.uint32)

        _populate_fracminhash_kernel(
            batch.seqs,
            batch.offsets,
            batch.lengths,
            kernel_lut,
            resolved_k,
            scaled,
            resolved_canonical,
            resolved_bits,
            out_offsets.astype(np.uint32),
            out_hashes,
            out_seq_indices,
            out_positions,
        )

        if sort_by_hash:
            out_hashes, out_seq_indices, out_positions = _radix_sort_soa(out_hashes, out_seq_indices, out_positions)

        return cls(
            hashes=out_hashes,
            seq_indices=out_seq_indices,
            positions=out_positions,
            offsets=out_offsets,
            lengths=counts.astype(np.int32),
            n_seqs=n_seqs,
            is_sorted=sort_by_hash,
            k=resolved_k,
            alphabet=alphabet_obj,
            scaled=scaled,
            canonical=resolved_canonical,
            bits_per_char=resolved_bits,
            reduce=resolved_reduce,
            lut=lut,
        )

    def _build_queries(self, queries: Sequences) -> FracMinHashIndex:
        r"""Build query FracMinHashIndex from sequences matching current settings."""
        return self.build(
            queries,
            k=self.k,
            alphabet=self.alphabet,
            scaled=self.scaled,
            canonical=self.canonical,
            sort_by_hash=False,
            lut=self.lut,
            bits_per_char=self.bits_per_char,
            reduce=self.reduce,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RandstrobeIndex(BaseKmerIndex):
    r"""Specialized index for fast sequence comparisons using syncmer-linked randstrobes.

    Attributes:
        s (int): Sub-k-mer size for open syncmer evaluation. Defaults to 5.
        w_min (int): Minimum syncmer offset window bound. Defaults to 1.
        w_max (int): Maximum syncmer offset window bound. Defaults to 5.
        lut (np.ndarray[tuple[int], np.dtype[np.uint8]] | None): Optional lookup table.
    """

    alphabet: Alphabet = Alphabet.MMSEQS12
    s: int = 5
    w_min: int = 1
    w_max: int = 5
    reduce: bool = True
    lut: np.ndarray[tuple[int], np.dtype[np.uint8]] | None = None

    @classmethod
    def empty(cls) -> RandstrobeIndex:
        r"""Create an empty RandstrobeIndex."""
        return cls(
            hashes=np.empty(0, dtype=np.uint64),
            seq_indices=np.empty(0, dtype=np.uint32),
            positions=np.empty(0, dtype=np.uint32),
            offsets=np.empty(0, dtype=np.int32),
            lengths=np.empty(0, dtype=np.int32),
            n_seqs=0,
            is_sorted=False,
            k=10,
            alphabet=Alphabet.MMSEQS12,
            s=5,
            w_min=1,
            w_max=5,
            reduce=True,
            lut=None,
        )

    @classmethod
    def build(
        cls,
        batch: Sequences,
        k: int = 10,
        s: int = 5,
        w_min: int = 1,
        w_max: int = 5,
        alphabet: Alphabet | str = Alphabet.MMSEQS12,
        seed: int = 42,
        sort_by_hash: bool = False,
        lut: np.ndarray[tuple[int], np.dtype[np.uint8]] | None = None,
        reduce: bool | None = None,
        **kwargs: object,
    ) -> RandstrobeIndex:
        r"""Build a RandstrobeIndex from sequence batch.

        Args:
            batch (Sequences): Input sequence collection.
            k (int): K-mer length. Defaults to 10.
            s (int): Sub-k-mer size. Defaults to 5.
            w_min (int): Min syncmer window bound. Defaults to 1.
            w_max (int): Max syncmer window bound. Defaults to 5.
            alphabet (Alphabet | str): Alphabet instance or name. Defaults to Alphabet.MMSEQS12.
            seed (int): Random seed. Defaults to 42.
            sort_by_hash (bool): True to sort records by hash. Defaults to False.
            lut (np.ndarray | None): Optional alphabet lookup table.
            reduce (bool | None): True to reduce alphabet.
            **kwargs (object): Additional parameters.

        Returns:
            RandstrobeIndex: Constructed [`RandstrobeIndex`][serogen.core.seq.kmers.RandstrobeIndex].

        Raises:
            ValueError: If `s >= k`.
        """
        if s >= k:
            raise ValueError("Sub-k-mer size (s) must be strictly less than k-mer size (k).")

        if len(batch) == 0:
            return cls.empty()

        if isinstance(alphabet, str):
            alphabet_obj = (
                Alphabet.MMSEQS12
                if (reduce is not False and alphabet in ("mmseqs12", "amino_acid", "amino"))
                else (Alphabet.DNA if alphabet == "dna" else Alphabet.AMINO)
            )
        else:
            alphabet_obj = alphabet

        resolved_reduce = alphabet_obj is Alphabet.MMSEQS12 if reduce is None else reduce
        kernel_lut = lut if lut is not None else alphabet_obj.kmer_lut

        counts = _count_randstrobes_kernel(batch.seqs, batch.offsets, batch.lengths, kernel_lut, k, s, w_min)
        total_randstrobes = int(np.sum(counts))
        if total_randstrobes == 0:
            return cls.empty()

        out_offsets = _compute_offsets_from_counts(counts)
        out_hashes = np.empty(total_randstrobes, dtype=np.uint64)
        out_seq_indices = np.empty(total_randstrobes, dtype=np.uint32)
        out_positions = np.empty(total_randstrobes, dtype=np.uint32)

        _populate_randstrobes_kernel(
            batch.seqs,
            batch.offsets,
            batch.lengths,
            kernel_lut,
            k,
            s,
            w_min,
            w_max,
            out_offsets.astype(np.uint32),
            out_hashes,
            out_seq_indices,
            out_positions,
        )

        if sort_by_hash:
            out_hashes, out_seq_indices, out_positions = _radix_sort_soa(out_hashes, out_seq_indices, out_positions)

        return cls(
            hashes=out_hashes,
            seq_indices=out_seq_indices,
            positions=out_positions,
            offsets=out_offsets,
            lengths=counts.astype(np.int32),
            n_seqs=len(batch),
            is_sorted=sort_by_hash,
            k=k,
            alphabet=alphabet_obj,
            s=s,
            w_min=w_min,
            w_max=w_max,
            reduce=resolved_reduce,
            lut=lut,
        )

    def _build_queries(self, queries: Sequences) -> RandstrobeIndex:
        r"""Build query RandstrobeIndex from sequences matching current settings."""
        return self.build(
            queries,
            k=self.k,
            s=self.s,
            w_min=self.w_min,
            w_max=self.w_max,
            alphabet=self.alphabet,
            sort_by_hash=False,
            lut=self.lut,
            reduce=self.reduce,
        )


# Kernels --------------------------------------------------------------------------------------------------------------
@njit(cache=True, nogil=True, inline="always")
def _splitmix64(x: np.uint64) -> np.uint64:
    r"""Fast, invertible 64-bit integer mixing function for randomizing hashes."""
    x = np.uint64(x + np.uint64(0x9E3779B97F4A7C15))
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@njit(cache=True, nogil=True)
def _radix_sort_soa(
    hashes: np.ndarray,
    seq_indices: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""In-place radix sort of SoA arrays by 64-bit hash values."""
    n = len(hashes)
    src_h = hashes.copy()
    dst_h = np.empty(n, dtype=np.uint64)
    src_s = seq_indices.copy()
    dst_s = np.empty(n, dtype=np.uint32)
    src_p = positions.copy()
    dst_p = np.empty(n, dtype=np.uint32)

    hist = np.zeros(256, dtype=np.uint32)
    pos = np.zeros(256, dtype=np.uint32)

    for p in range(8):
        shift = np.uint64(p * 8)

        hist.fill(0)
        for i in range(n):
            byte_val = (src_h[i] >> shift) & np.uint64(0xFF)
            hist[byte_val] += 1

        curr_pos = np.uint32(0)
        for i in range(256):
            pos[i] = curr_pos
            curr_pos += hist[i]

        for i in range(n):
            byte_val = (src_h[i] >> shift) & np.uint64(0xFF)
            write_idx = pos[byte_val]
            dst_h[write_idx] = src_h[i]
            dst_s[write_idx] = src_s[i]
            dst_p[write_idx] = src_p[i]
            pos[byte_val] += 1

        tmp_h = src_h
        src_h = dst_h
        dst_h = tmp_h

        tmp_s = src_s
        src_s = dst_s
        dst_s = tmp_s

        tmp_p = src_p
        src_p = dst_p
        dst_p = tmp_p

    return src_h, src_s, src_p


@njit(cache=True, nogil=True, inline="always")
def _scan_minhashes_seq(
    seqs: np.ndarray,
    seq_start: int,
    seq_len: int,
    lut: np.ndarray,
    k: int,
    scaled: int,
    canonical: bool,
    bits_per_char: int,
    seq_idx: int,
    out_s: int,
    out_hashes: np.ndarray,
    out_seq_indices: np.ndarray,
    out_positions: np.ndarray,
    populate: bool,
) -> int:
    r"""Scan rolling k-mers for a single sequence; count or populate records."""
    if seq_len < k:
        return 0

    total_bits = bits_per_char * k
    mask = ~np.uint64(0) if total_bits >= 64 else (np.uint64(1) << np.uint64(total_bits)) - np.uint64(1)
    max_val = np.uint64(1) << np.uint64(bits_per_char)
    threshold = ~np.uint64(0) // np.uint64(scaled)

    k_val_fwd = np.uint64(0)
    k_val_rev = np.uint64(0)
    valid_len = 0
    count = 0

    for j in range(seq_len):
        char = seqs[seq_start + j]
        val = np.uint64(lut[char])

        if val < max_val:
            k_val_fwd = ((k_val_fwd << np.uint64(bits_per_char)) & mask) | val
            valid_len += 1

            if canonical:
                comp = val ^ np.uint64(2)
                k_val_rev = (k_val_rev >> np.uint64(bits_per_char)) | (comp << np.uint64(bits_per_char * (k - 1)))

            if valid_len >= k:
                if canonical:
                    h_val = min(k_val_fwd, k_val_rev)
                else:
                    h_val = k_val_fwd

                h = _splitmix64(h_val)
                if h <= threshold:
                    if populate:
                        out_hashes[out_s + count] = h
                        out_seq_indices[out_s + count] = seq_idx
                        out_positions[out_s + count] = j - k + 1
                    count += 1
        else:
            valid_len = 0
            k_val_fwd = np.uint64(0)
            k_val_rev = np.uint64(0)

    return count


@njit(parallel=True, cache=True, nogil=True)
def _count_fracminhash_kernel(
    seqs: np.ndarray[tuple[int], np.dtype[np.uint8]],
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    lut: np.ndarray[tuple[int], np.dtype[np.uint8]],
    k: int,
    scaled: int,
    canonical: bool,
    bits_per_char: int,
) -> np.ndarray[tuple[int], np.dtype[np.uint32]]:
    r"""Count FracMinHash occurrences per sequence in parallel."""
    n_seqs = len(offsets)
    counts = np.zeros(n_seqs, dtype=np.uint32)
    dummy_u64 = np.empty(0, dtype=np.uint64)
    dummy_u32 = np.empty(0, dtype=np.uint32)

    for idx in prange(n_seqs):
        counts[idx] = _scan_minhashes_seq(
            seqs,
            offsets[idx],
            lengths[idx],
            lut,
            k,
            scaled,
            canonical,
            bits_per_char,
            idx,
            0,
            dummy_u64,
            dummy_u32,
            dummy_u32,
            False,
        )
    return counts


@njit(parallel=True, cache=True, nogil=True)
def _populate_fracminhash_kernel(
    seqs: np.ndarray[tuple[int], np.dtype[np.uint8]],
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    lut: np.ndarray[tuple[int], np.dtype[np.uint8]],
    k: int,
    scaled: int,
    canonical: bool,
    bits_per_char: int,
    out_offsets: np.ndarray[tuple[int], np.dtype[np.uint32]],
    out_hashes: np.ndarray,
    out_seq_indices: np.ndarray,
    out_positions: np.ndarray,
) -> None:
    r"""Populate FracMinHash SoA arrays in parallel."""
    n_seqs = len(offsets)
    for idx in prange(n_seqs):
        _scan_minhashes_seq(
            seqs,
            offsets[idx],
            lengths[idx],
            lut,
            k,
            scaled,
            canonical,
            bits_per_char,
            idx,
            out_offsets[idx],
            out_hashes,
            out_seq_indices,
            out_positions,
            True,
        )


@njit(cache=True, nogil=True, inline="always")
def _scan_randstrobes_seq(
    seqs: np.ndarray,
    seq_start: int,
    seq_len: int,
    lut: np.ndarray,
    k: int,
    s: int,
    w_min: int,
    w_max: int,
    seq_idx: int,
    out_s: int,
    out_hashes: np.ndarray,
    out_seq_indices: np.ndarray,
    out_positions: np.ndarray,
    populate: bool,
) -> int:
    r"""Scan open syncmers and link order-2 randstrobes for a single sequence."""
    if seq_len < k:
        return 0

    max_syncmers = seq_len - k + 1

    if not populate:
        n_syncmers = 0
        for i in range(seq_len - k + 1):
            min_s_hash = ~np.uint64(0)
            min_s_idx = -1

            for j in range(k - s + 1):
                s_val = np.uint64(0)
                for c in range(s):
                    char = seqs[seq_start + i + j + c]
                    s_val = s_val * np.uint64(12) + np.uint64(lut[char])

                h = _splitmix64(s_val)
                if h < min_s_hash:
                    min_s_hash = h
                    min_s_idx = j

            if min_s_idx == 0 or min_s_idx == (k - s):
                n_syncmers += 1

        return n_syncmers - w_min if n_syncmers > w_min else 0

    # Populating phase
    syncmer_pos = np.empty(max_syncmers, dtype=np.uint32)
    syncmer_hash = np.empty(max_syncmers, dtype=np.uint64)
    n_syncmers = 0

    for i in range(seq_len - k + 1):
        min_s_hash = ~np.uint64(0)
        min_s_idx = -1

        for j in range(k - s + 1):
            s_val = np.uint64(0)
            for c in range(s):
                char = seqs[seq_start + i + j + c]
                s_val = s_val * np.uint64(12) + np.uint64(lut[char])

            h = _splitmix64(s_val)
            if h < min_s_hash:
                min_s_hash = h
                min_s_idx = j

        if min_s_idx == 0 or min_s_idx == (k - s):
            syncmer_pos[n_syncmers] = i
            k_val = np.uint64(0)
            for c in range(k):
                char = seqs[seq_start + i + c]
                k_val = k_val * np.uint64(12) + np.uint64(lut[char])

            syncmer_hash[n_syncmers] = _splitmix64(k_val)
            n_syncmers += 1

    out_idx = out_s
    for i in range(n_syncmers - w_min):
        h1 = syncmer_hash[i]
        min_randstrobe_hash = ~np.uint64(0)
        best_j = -1

        end_j = i + w_max + 1
        if end_j > n_syncmers:
            end_j = n_syncmers

        for j in range(i + w_min, end_j):
            h2 = syncmer_hash[j]
            combined = _splitmix64(h1 ^ _splitmix64(h2))
            if combined < min_randstrobe_hash:
                min_randstrobe_hash = combined
                best_j = j

        if best_j != -1:
            out_hashes[out_idx] = min_randstrobe_hash
            out_seq_indices[out_idx] = seq_idx
            out_positions[out_idx] = syncmer_pos[i]
            out_idx += 1

    return out_idx - out_s


@njit(parallel=True, cache=True, nogil=True)
def _count_randstrobes_kernel(
    seqs: np.ndarray[tuple[int], np.dtype[np.uint8]],
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    lut: np.ndarray[tuple[int], np.dtype[np.uint8]],
    k: int,
    s: int,
    w_min: int,
) -> np.ndarray[tuple[int], np.dtype[np.uint32]]:
    r"""Count randstrobes per sequence in parallel."""
    n_seqs = len(offsets)
    counts = np.zeros(n_seqs, dtype=np.uint32)
    dummy_u64 = np.empty(0, dtype=np.uint64)
    dummy_u32 = np.empty(0, dtype=np.uint32)

    for idx in prange(n_seqs):
        counts[idx] = _scan_randstrobes_seq(
            seqs,
            offsets[idx],
            lengths[idx],
            lut,
            k,
            s,
            w_min,
            0,
            idx,
            0,
            dummy_u64,
            dummy_u32,
            dummy_u32,
            False,
        )
    return counts


@njit(parallel=True, cache=True, nogil=True)
def _populate_randstrobes_kernel(
    seqs: np.ndarray[tuple[int], np.dtype[np.uint8]],
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    lut: np.ndarray[tuple[int], np.dtype[np.uint8]],
    k: int,
    s: int,
    w_min: int,
    w_max: int,
    out_offsets: np.ndarray[tuple[int], np.dtype[np.uint32]],
    out_hashes: np.ndarray,
    out_seq_indices: np.ndarray,
    out_positions: np.ndarray,
) -> None:
    r"""Populate randstrobe SoA arrays in parallel."""
    n_seqs = len(offsets)
    for idx in prange(n_seqs):
        _scan_randstrobes_seq(
            seqs,
            offsets[idx],
            lengths[idx],
            lut,
            k,
            s,
            w_min,
            w_max,
            idx,
            out_offsets[idx],
            out_hashes,
            out_seq_indices,
            out_positions,
            True,
        )


@njit(cache=True, nogil=True)
def _compute_query_offsets(q_seq_indices: np.ndarray, num_queries: int) -> np.ndarray[tuple[int], np.dtype[np.uint32]]:
    r"""Find boundary indices for each query sequence in flat hash array."""
    q_offsets = np.zeros(num_queries + 1, dtype=np.uint32)
    n = len(q_seq_indices)
    if n == 0:
        return q_offsets

    current_query = 0
    for i in range(n):
        q = q_seq_indices[i]
        while current_query < q:
            current_query += 1
            q_offsets[current_query] = i

    while current_query < num_queries:
        current_query += 1
        q_offsets[current_query] = n

    return q_offsets


@njit(cache=True, nogil=True, inline="always")
def _tally_single_query(
    start: int,
    end: int,
    q_hashes: np.ndarray,
    q_positions: np.ndarray,
    t_hashes: np.ndarray,
    t_seq_indices: np.ndarray,
    t_positions: np.ndarray,
    tally: np.ndarray[tuple[int], np.dtype[np.uint32]],
    anchors: np.ndarray[tuple[int], np.dtype[np.int32]],
) -> None:
    r"""Tally matching hashes for a single query sequence against target records."""
    t_len = len(t_hashes)
    for i in range(start, end):
        h = q_hashes[i]
        q_p = q_positions[i]

        # Manual binary search (searchsorted_left)
        low, high = 0, t_len
        while low < high:
            mid = (low + high) // 2
            if t_hashes[mid] < h:
                low = mid + 1
            else:
                high = mid

        curr = low
        while curr < t_len and t_hashes[curr] == h:
            t_id = t_seq_indices[curr]
            tally[t_id] += 1
            if tally[t_id] == 1:
                anchors[t_id] = np.int32(q_p) - np.int32(t_positions[curr])
            curr += 1


@njit(parallel=True, cache=True, nogil=True)
def _intersect_top_hit_kernel(
    q_hashes: np.ndarray,
    q_positions: np.ndarray,
    q_offsets: np.ndarray[tuple[int], np.dtype[np.uint32]],
    num_queries: int,
    t_hashes: np.ndarray,
    t_seq_indices: np.ndarray,
    t_positions: np.ndarray,
    num_targets: int,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.uint32]],
    np.ndarray[tuple[int], np.dtype[np.uint32]],
    np.ndarray[tuple[int], np.dtype[np.uint32]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
]:
    r"""Numba parallel kernel finding top-scoring target hit per query."""
    best_t = np.zeros(num_queries, dtype=np.uint32)
    max_s = np.zeros(num_queries, dtype=np.uint32)
    offsets = np.zeros(num_queries, dtype=np.int32)

    for q_idx in prange(num_queries):
        start, end = q_offsets[q_idx], q_offsets[q_idx + 1]
        if start == end:
            continue

        tally = np.zeros(num_targets, dtype=np.uint32)
        anchors = np.zeros(num_targets, dtype=np.int32)
        _tally_single_query(
            start,
            end,
            q_hashes,
            q_positions,
            t_hashes,
            t_seq_indices,
            t_positions,
            tally,
            anchors,
        )

        best_score, best_target = 0, 0
        for t in range(num_targets):
            if tally[t] > best_score:
                best_score = tally[t]
                best_target = t

        best_t[q_idx] = best_target
        max_s[q_idx] = best_score
        offsets[q_idx] = anchors[best_target]

    q_idx_arr = np.arange(num_queries, dtype=np.uint32)
    return q_idx_arr, best_t, max_s, offsets


@njit(parallel=True, cache=True, nogil=True)
def _count_hits_kernel(
    q_hashes: np.ndarray,
    q_positions: np.ndarray,
    q_offsets: np.ndarray[tuple[int], np.dtype[np.uint32]],
    num_queries: int,
    t_hashes: np.ndarray,
    t_seq_indices: np.ndarray,
    t_positions: np.ndarray,
    num_targets: int,
    min_score: int,
) -> np.ndarray[tuple[int], np.dtype[np.uint32]]:
    r"""Numba parallel kernel to count total hits per query above min_score."""
    counts = np.zeros(num_queries, dtype=np.uint32)
    for q_idx in prange(num_queries):
        start, end = q_offsets[q_idx], q_offsets[q_idx + 1]
        if start == end:
            continue
        tally = np.zeros(num_targets, dtype=np.uint32)
        anchors = np.zeros(num_targets, dtype=np.int32)
        _tally_single_query(
            start,
            end,
            q_hashes,
            q_positions,
            t_hashes,
            t_seq_indices,
            t_positions,
            tally,
            anchors,
        )

        count = 0
        for t in range(num_targets):
            if tally[t] >= min_score:
                count += 1
        counts[q_idx] = count
    return counts


@njit(parallel=True, cache=True, nogil=True)
def _populate_hits_kernel(
    q_hashes: np.ndarray,
    q_positions: np.ndarray,
    q_offsets: np.ndarray[tuple[int], np.dtype[np.uint32]],
    num_queries: int,
    t_hashes: np.ndarray,
    t_seq_indices: np.ndarray,
    t_positions: np.ndarray,
    num_targets: int,
    min_score: int,
    out_offsets: np.ndarray[tuple[int], np.dtype[np.uint32]],
    out_q: np.ndarray[tuple[int], np.dtype[np.uint32]],
    out_t: np.ndarray[tuple[int], np.dtype[np.uint32]],
    out_s: np.ndarray[tuple[int], np.dtype[np.uint32]],
    out_o: np.ndarray[tuple[int], np.dtype[np.int32]],
) -> None:
    r"""Numba parallel kernel to populate all hits into flat arrays."""
    for q_idx in prange(num_queries):
        start, end = q_offsets[q_idx], q_offsets[q_idx + 1]
        if start == end:
            continue
        tally = np.zeros(num_targets, dtype=np.uint32)
        anchors = np.zeros(num_targets, dtype=np.int32)
        _tally_single_query(
            start,
            end,
            q_hashes,
            q_positions,
            t_hashes,
            t_seq_indices,
            t_positions,
            tally,
            anchors,
        )

        out_idx = out_offsets[q_idx]
        for t in range(num_targets):
            if tally[t] >= min_score:
                out_q[out_idx] = q_idx
                out_t[out_idx] = t
                out_s[out_idx] = tally[t]
                out_o[out_idx] = anchors[t]
                out_idx += 1
