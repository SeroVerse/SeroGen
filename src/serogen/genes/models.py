r"""Structure-of-Arrays data containers for predicted genes, indels, and HMM parameters."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Self, overload

import numpy as np

from serogen.core.collections import BatchedContainer, RaggedArrayContainer
from serogen.core.interval import Interval, Intervals
from serogen.core.seq.alphabet import GeneticCode
from serogen.core.seq.seq import Sequence, Sequences


class IndelKind(IntEnum):
    r"""Indel mutation type for frameshift tracking."""

    INSERTION = 1
    DELETION = -1


@dataclass(slots=True, frozen=True, eq=False)
class IndelRecord:
    r"""Scalar view of frameshift indel events for a single gene.

    Attributes:
        positions (np.ndarray): 0-based contig coordinates where indels occurred.
        kinds (np.ndarray): Indel mutation types (+1 for Insertion, -1 for Deletion).
    """

    positions: np.ndarray[tuple[int], np.dtype[np.int32]]
    kinds: np.ndarray[tuple[int], np.dtype[np.int8]]

    def __len__(self) -> int:
        r"""Return the number of indel events in this record."""
        return len(self.positions)

    def __eq__(self, other: object) -> bool:
        r"""Compare two IndelRecord views for exact coordinate and kind equality."""
        if not isinstance(other, IndelRecord):
            return NotImplemented
        return bool(np.array_equal(self.positions, other.positions) and np.array_equal(self.kinds, other.kinds))


@dataclass(slots=True, frozen=True)
class Indels(RaggedArrayContainer[IndelRecord, "Indels"]):
    r"""Ragged Structure-of-Arrays container for frameshift indels across a Genes batch.

    Stores variable-length indel event coordinates and mutation kinds in flat contiguous arrays,
    indexed via `offsets` and `lengths` per gene.

    Attributes:
        positions (np.ndarray): 1D array of 0-based contig coordinates across all indels.
        kinds (np.ndarray): 1D array of indel kinds (+1=ins, -1=del).
        offsets (np.ndarray): Starting index into `positions` and `kinds` for each gene.
        lengths (np.ndarray): Number of indels for each gene.
    """

    positions: np.ndarray[tuple[int], np.dtype[np.int32]]
    kinds: np.ndarray[tuple[int], np.dtype[np.int8]]
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]]

    def __len__(self) -> int:
        r"""Return the number of genes in the batch."""
        return len(self.offsets)

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> IndelRecord: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> Indels: ...

    @overload
    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> IndelRecord | Indels: ...

    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> IndelRecord | Indels:
        r"""Access indel records by index, slice, or boolean mask.

        Args:
            item (int | slice | np.ndarray | list): Integer index, slice, or mask/indices array.

        Returns:
            IndelRecord | Indels: A single gene's indel record or a sliced [`Indels`][serogen.bgc.genes.Indels] batch.

        Raises:
            IndexError: If an integer index is out of range.
        """
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError("Batch index out of range")
            off_val = int(self.offsets[idx])
            len_val = int(self.lengths[idx])
            return IndelRecord(
                positions=self.positions[off_val : off_val + len_val],
                kinds=self.kinds[off_val : off_val + len_val],
            )

        indices, new_offsets, new_lengths, (new_pos, new_kinds) = self._slice_ragged(item, self.positions, self.kinds)
        if len(indices) == 0:
            return self.empty()
        return Indels(positions=new_pos, kinds=new_kinds, offsets=new_offsets, lengths=new_lengths)

    @classmethod
    def empty(cls, length: int = 0) -> Indels:
        r"""Create an empty Indels container."""
        if length == 0:
            return cls(
                positions=np.empty(0, dtype=np.int32),
                kinds=np.empty(0, dtype=np.int8),
                offsets=np.empty(0, dtype=np.int32),
                lengths=np.empty(0, dtype=np.int32),
            )
        return cls(
            positions=np.empty(0, dtype=np.int32),
            kinds=np.empty(0, dtype=np.int8),
            offsets=np.zeros(length, dtype=np.int32),
            lengths=np.zeros(length, dtype=np.int32),
        )

    @classmethod
    def concat(cls, batches: Iterable[Indels]) -> Indels:
        r"""Concatenate multiple Indels containers into a single batch."""
        batches_list = list(batches)
        if not batches_list:
            return cls.empty()
        lengths = np.concatenate([b.lengths for b in batches_list])
        offsets = cls.lengths_to_offsets(lengths)
        total_items = sum(len(b.positions) for b in batches_list)
        if total_items == 0:
            return cls(
                positions=np.empty(0, dtype=np.int32),
                kinds=np.empty(0, dtype=np.int8),
                offsets=offsets,
                lengths=lengths,
            )
        positions = np.concatenate([b.positions for b in batches_list if len(b.positions) > 0])
        kinds = np.concatenate([b.kinds for b in batches_list if len(b.kinds) > 0])
        return cls(positions=positions, kinds=kinds, offsets=offsets, lengths=lengths)


@dataclass(slots=True, frozen=True)
class Gene:
    r"""Scalar view of a single predicted gene.

    Attributes:
        interval (Interval): Genomic interval coordinate ([`Interval`][serogen.core.interval.Interval]).
        seq (Sequence | None): Coding nucleotide or protein sequence ([`Sequence`][serogen.core.seq.Sequence]).
        contig_index (int): 0-based contig index offset corresponding to parent contig name.
        indels (IndelRecord): Scalar frameshift indel record ([`IndelRecord`][serogen.genes.models.IndelRecord]).
        is_partial (bool): True if the ORF is truncated at a contig boundary without a start or stop codon.
    """

    interval: Interval
    seq: Sequence | None
    contig_index: int
    indels: IndelRecord
    is_partial: bool = False


@dataclass(slots=True, frozen=True, init=False)
class Genes(BatchedContainer[Gene, "Genes"]):
    r"""Structure-of-Arrays (SoA) container for predicted genes.

    Stores interval coordinates, in-frame coding nucleotide sequences, and contig index
    mappings for a collection of predicted Open Reading Frames (ORFs). Inherits
    vectorised slicing and concatenation interface from
    [`BatchedContainer`][serogen.core.collections.BatchedContainer].

    Attributes:
        intervals (Intervals): Genomic interval coordinates for predicted ORFs
            ([`Intervals`][serogen.core.interval.Intervals]).
        seqs (Sequences): In-frame coding nucleotide sequences oriented 5' to 3'
            ([`Sequences`][serogen.core.seq.Sequences]).
        contig_indices (np.ndarray[tuple[int], np.dtype[np.uint32]]): 1D array of 0-based contig index
            offsets corresponding to parent contig names.
        indels (Indels): Ragged frameshift indel records.
        is_partial (np.ndarray[tuple[int], np.dtype[np.bool_]]): Boolean mask indicating contig-edge partial ORFs.
    """

    intervals: Intervals
    seqs: Sequences
    contig_indices: np.ndarray[tuple[int], np.dtype[np.uint32]]
    indels: Indels
    is_partial: np.ndarray[tuple[int], np.dtype[np.bool_]]
    genetic_code: GeneticCode
    _translations: Sequences | None

    def __init__(
        self,
        intervals: Intervals,
        seqs: Sequences | None = None,
        contig_indices: np.ndarray[tuple[int], np.dtype[np.uint32]] | None = None,
        indels: Indels | None = None,
        translations: Sequences | None = None,
        _translations: Sequences | None = None,
        is_partial: np.ndarray[tuple[int], np.dtype[np.bool_]] | None = None,
        genetic_code: GeneticCode | int | str = GeneticCode.bacterial,
    ) -> None:
        r"""Initialize a [`Genes`][serogen.genes.models.Genes] container."""
        n = len(intervals)
        object.__setattr__(self, "intervals", intervals)
        if seqs is None:
            if translations is not None:
                object.__setattr__(self, "seqs", Sequences.empty())
                object.__setattr__(self, "_translations", translations)
            elif _translations is not None:
                object.__setattr__(self, "seqs", Sequences.empty())
                object.__setattr__(self, "_translations", _translations)
            else:
                object.__setattr__(self, "seqs", Sequences.empty())
                object.__setattr__(self, "_translations", None)
        else:
            object.__setattr__(self, "seqs", seqs)
            object.__setattr__(self, "_translations", translations if translations is not None else _translations)

        object.__setattr__(
            self,
            "contig_indices",
            contig_indices if contig_indices is not None else np.zeros(n, dtype=np.uint32),
        )
        object.__setattr__(
            self,
            "indels",
            indels if indels is not None else Indels.empty(n),
        )
        object.__setattr__(
            self,
            "is_partial",
            np.asarray(is_partial, dtype=np.bool_) if is_partial is not None else np.zeros(n, dtype=np.bool_),
        )
        object.__setattr__(
            self,
            "genetic_code",
            genetic_code if isinstance(genetic_code, GeneticCode) else GeneticCode.from_table(genetic_code),
        )

    @property
    def has_frameshift(self) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        r"""Boolean mask indicating which genes contain at least one frameshift indel."""
        return self.indels.lengths > 0

    @property
    def frameshift_count(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        r"""Array of frameshift indel counts per gene."""
        return self.indels.lengths

    @property
    def translations(self) -> Sequences:
        r"""Translated protein sequences.

        Returns pre-computed translations if provided, or computes them lazily using
        [`self.genetic_code`][serogen.core.seq.alphabet.GeneticCode].

        Returns:
            Sequences: Protein translation sequences.
        """
        if self._translations is not None:
            return self._translations
        if len(self.seqs) == 0:
            return Sequences.empty()
        translated = self.seqs.translate(code=self.genetic_code, to_stop=True)
        object.__setattr__(self, "_translations", translated)
        return translated

    def translate(self, code: GeneticCode | int | str | None = None, to_stop: bool = True) -> Sequences:
        r"""Translate coding nucleotide sequences into protein sequences.

        Args:
            code (GeneticCode | int | str | None): Genetic code translation table to use.
                If None, uses `self.genetic_code`.
            to_stop (bool): If True, truncates translation at the first stop codon (*). Defaults to True.

        Returns:
            Sequences: Translated protein sequences.
        """
        if len(self.seqs) == 0 and self._translations is not None:
            return self._translations
        resolved = (
            self.genetic_code
            if code is None
            else (code if isinstance(code, GeneticCode) else GeneticCode.from_table(code))
        )
        return self.seqs.translate(code=resolved, to_stop=to_stop)

    def __len__(self) -> int:
        r"""Return total number of predicted genes in container.

        Returns:
            int: Number of genes.
        """
        return len(self.intervals)

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> Gene: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> Genes: ...

    @overload
    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> Gene | Genes: ...

    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> Gene | Genes:
        r"""Retrieve gene entry by index or slice container by range or boolean mask."""
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            seq_val = (
                self.seqs[idx]
                if len(self.seqs) > 0
                else (self._translations[idx] if self._translations is not None else None)
            )
            return Gene(
                interval=self.intervals[idx],
                seq=seq_val,
                contig_index=int(self.contig_indices[idx]) if len(self.contig_indices) > 0 else 0,
                indels=self.indels[idx],
                is_partial=bool(self.is_partial[idx]) if len(self.is_partial) > 0 else False,
            )
        return Genes(
            intervals=self.intervals[item],
            seqs=self.seqs[item] if len(self.seqs) > 0 else None,
            contig_indices=self.contig_indices[item] if len(self.contig_indices) > 0 else None,
            indels=self.indels[item],
            _translations=self._translations[item] if self._translations is not None else None,
            is_partial=self.is_partial[item] if len(self.is_partial) > 0 else None,
            genetic_code=self.genetic_code,
        )

    @classmethod
    def empty(cls) -> Self:
        r"""Create an empty [`Genes`][serogen.genes.models.Genes] container."""
        return cls(
            intervals=Intervals.empty(),
            seqs=Sequences.empty(),
            contig_indices=np.empty(0, dtype=np.uint32),
            indels=Indels.empty(),
            is_partial=np.empty(0, dtype=np.bool_),
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        r"""Concatenate multiple [`Genes`][serogen.genes.models.Genes] batches into a single container."""
        batches_list = [b for b in batches if len(b) > 0]
        if not batches_list:
            return cls.empty()

        has_seqs = any(len(b.seqs) > 0 for b in batches_list)
        has_trans = any(b._translations is not None for b in batches_list)

        return cls(
            intervals=Intervals.concat([b.intervals for b in batches_list]),
            seqs=Sequences.concat([b.seqs for b in batches_list]) if has_seqs else Sequences.empty(),
            contig_indices=np.concatenate([b.contig_indices for b in batches_list]),
            indels=Indels.concat([b.indels for b in batches_list]),
            _translations=(
                Sequences.concat([b.translations for b in batches_list]) if has_trans and not has_seqs else None
            ),
            is_partial=np.concatenate([b.is_partial for b in batches_list]),
            genetic_code=batches_list[0].genetic_code,
        )


@dataclass(frozen=True, slots=True)
class HMM:
    r"""Container holding baked FragGeneScan HMM parameters.

    Attributes:
        tr (np.ndarray): Global transition penalties (14,).
        tr_mi (np.ndarray): Match-to-insertion transitions (4, 4).
        tr_ii (np.ndarray): Insertion-to-insertion transitions (4, 4).
        pi (np.ndarray): Initial state log-probabilities (29,).
        e_m (np.ndarray): Forward match state emission log-probabilities (44, 6, 16, 4).
        e_m1 (np.ndarray): Reverse match state emission log-probabilities (44, 6, 16, 4).
        tr_rr (np.ndarray): Non-coding state transitions (44, 4, 4).
        tr_s (np.ndarray): PWM window transitions around forward start (44, 61, 64).
        tr_e (np.ndarray): PWM window transitions around forward stop (44, 61, 64).
        tr_s1 (np.ndarray): PWM window transitions around reverse start (44, 61, 64).
        tr_e1 (np.ndarray): PWM window transitions around reverse stop (44, 61, 64).
        dist_s (np.ndarray): Gaussian parameters for forward start (44, 6).
        dist_e (np.ndarray): Gaussian parameters for forward stop (44, 6).
        dist_s1 (np.ndarray): Gaussian parameters for reverse start (44, 6).
        dist_e1 (np.ndarray): Gaussian parameters for reverse stop (44, 6).
    """

    tr: np.ndarray[tuple[int], np.dtype[np.float64]]
    tr_mi: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    tr_ii: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    pi: np.ndarray[tuple[int], np.dtype[np.float64]]
    tr_rr: np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
    e_match: np.ndarray[tuple[int, int, int, int, int], np.dtype[np.float64]]
    tr_start: np.ndarray[tuple[int, int, int, int], np.dtype[np.float64]]
    tr_stop: np.ndarray[tuple[int, int, int, int], np.dtype[np.float64]]
    dist_start: np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
    dist_stop: np.ndarray[tuple[int, int, int], np.dtype[np.float64]]

    @classmethod
    def load(cls, path: str | Path | None = None) -> HMM:
        r"""Load HMM parameters from compressed NumPy archive (.npz).

        Args:
            path (str | Path | None): Optional path to .npz archive. If None, loads the
                bundled default model located at `serogen/bgc/data/fgs_model.npz`.

        Returns:
            HMM: Instantiated HMM parameter container.
        """
        if path is None:
            path = Path(__file__).resolve().parent / "data" / "fgs_model.npz"
        data = np.load(path)
        return cls(
            tr=data["tr"],
            tr_mi=data["tr_mi"],
            tr_ii=data["tr_ii"],
            pi=data["pi"],
            tr_rr=data["tr_rr"],
            e_match=np.stack([data["e_m"], data["e_m1"]]),
            tr_start=np.stack([data["tr_s"], data["tr_s1"]]),
            tr_stop=np.stack([data["tr_e"], data["tr_e1"]]),
            dist_start=np.stack([data["dist_s"], data["dist_s1"]]),
            dist_stop=np.stack([data["dist_e"], data["dist_e1"]]),
        )


@lru_cache(maxsize=1)
def get_hmm_model() -> HMM:
    r"""Return cached singleton instance of the default bundled HMM model.

    Returns:
        HMM: Default baked model.
    """
    return HMM.load()


def _build_indels(pos_list: list[np.ndarray], kind_list: list[np.ndarray]) -> Indels:  # type: ignore[type-arg]
    if not pos_list:
        return Indels.empty()
    n = len(pos_list)
    lengths = np.array([len(x) for x in pos_list], dtype=np.int32)
    offsets = np.zeros(n, dtype=np.int32)
    if n > 1:
        np.cumsum(lengths[:-1], out=offsets[1:])
    total_indels = int(np.sum(lengths))
    if total_indels == 0:
        return Indels(
            positions=np.empty(0, dtype=np.int32),
            kinds=np.empty(0, dtype=np.int8),
            offsets=offsets,
            lengths=lengths,
        )
    positions = np.concatenate([x for x in pos_list if len(x) > 0])
    kinds = np.concatenate([x for x in kind_list if len(x) > 0])
    return Indels(positions=positions, kinds=kinds, offsets=offsets, lengths=lengths)


def _build_sequences(cds_list: list[np.ndarray]) -> Sequences:
    if not cds_list:
        return Sequences.empty()
    n = len(cds_list)
    ids_arr = np.array([b"%d" % i for i in range(n)], dtype="S")
    out_seqs = np.concatenate(cds_list) if n > 1 else cds_list[0].copy()
    lengths = np.array([len(x) for x in cds_list], dtype=np.int32)
    offsets = np.zeros(n, dtype=np.int32)
    if n > 1:
        np.cumsum(lengths[:-1], out=offsets[1:])
    return Sequences(ids_arr, out_seqs, offsets, lengths)
