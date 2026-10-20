r"""Genomic interval representation, strand orientations, and vectorized Structure-of-Arrays operations.

This module provides discrete coordinate intervals via [`Interval`][serogen.core.interval.Interval],
strand representations via [`Strand`][serogen.core.interval.Strand], and high-performance,
vectorized interval collections via [`Intervals`][serogen.core.interval.Intervals].
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Self, overload

import numpy as np

from serogen.core.collections import BatchedContainer


# Enums ----------------------------------------------------------------------------------------------------------------
class Strand(IntEnum):
    r"""Integer representation of genomic strand orientation.

    Supports conversion from common string formats (`'+'`, `'-'`, `'1'`, `'-1'`).

    Attributes:
        FORWARD (int): Forward strand (`+1`).
        REVERSE (int): Reverse strand (`-1`).
        UNSTRANDED (int): Unstranded or unknown orientation (`0`).
    """

    FORWARD = 1
    REVERSE = -1
    UNSTRANDED = 0

    @classmethod
    def _missing_(cls, value: object) -> Strand:
        r"""Coerce strings, bytes, or integers to a valid [`Strand`][serogen.core.interval.Strand] instance.

        Args:
            value (object): Value to coerce (e.g. `'+'`, `'-'`, `b'+'`, `1`, `-1`).

        Returns:
            Strand: Corresponding [`Strand`][serogen.core.interval.Strand] enum member.
        """
        if isinstance(value, bytes):
            value = value.decode("ascii")
        if isinstance(value, str):
            if value in ("+", "1", "+1"):
                return Strand.FORWARD
            elif value in ("-", "-1"):
                return Strand.REVERSE
        return Strand.UNSTRANDED

    def __str__(self) -> str:
        r"""Return the string representation of the strand (`'+'`, `'-'`, or `'.'`).

        Returns:
            str: Single-character string representation.
        """
        if self == Strand.FORWARD:
            return "+"
        if self == Strand.REVERSE:
            return "-"
        return "."


# Classes --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Interval:
    r"""A single 0-based, half-open genomic interval with strand orientation.

    Represents a discrete sequence segment defined by `start` (inclusive), `end` (exclusive),
    and `strand` orientation.

    Attributes:
        start (int): 0-based starting position (inclusive).
        end (int): 0-based ending position (exclusive).
        strand (Strand): Strand orientation [`Strand`][serogen.core.interval.Strand].
    """

    start: int
    end: int
    strand: Strand = Strand.UNSTRANDED

    def __len__(self) -> int:
        r"""Calculate the length of the interval in base pairs (`end - start`).

        Returns:
            int: Interval length.
        """
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Intervals(BatchedContainer[Interval, "Intervals"]):
    r"""A high-performance Structure-of-Arrays (SoA) collection of genomic intervals.

    Stores starts, ends, and strands as parallel 1D NumPy arrays
    for memory-efficient vectorized operations and Numba spatial algorithms.

    Attributes:
        starts (np.ndarray[tuple[int], np.dtype[np.int32]]): 1D array of 0-based start coordinates.
        ends (np.ndarray[tuple[int], np.dtype[np.int32]]): 1D array of 0-based end coordinates.
        strands (np.ndarray[tuple[int], np.dtype[np.int8]]): 1D array of strand values (+1, -1, 0).
    """

    starts: np.ndarray[tuple[int], np.dtype[np.int32]]
    ends: np.ndarray[tuple[int], np.dtype[np.int32]]
    strands: np.ndarray[tuple[int], np.dtype[np.int8]]

    @classmethod
    def empty(cls) -> Self:
        r"""Create an empty [`Intervals`][serogen.core.interval.Intervals] object with zero-length arrays.

        Returns:
            Intervals: Empty [`Intervals`][serogen.core.interval.Intervals] collection.
        """
        return cls(
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int8),
        )

    def __len__(self) -> int:
        r"""Return the number of intervals in the batch.

        Returns:
            int: Total interval count.
        """
        return len(self.starts)

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> Interval: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> Intervals: ...

    @overload
    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> Interval | Intervals: ...

    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> Interval | Intervals:
        r"""Access intervals by integer index, slice, or boolean/integer NumPy array.

        Args:
            item (int | slice | np.ndarray | list): Integer index, slice, or NumPy mask array.

        Returns:
            Interval | Intervals: A single [`Interval`][serogen.core.interval.Interval] or a
                sliced [`Intervals`][serogen.core.interval.Intervals] collection.

        Raises:
            IndexError: If integer index is out of bounds.
        """
        if isinstance(item, (int, np.integer)):
            item_idx = int(item)
            if item_idx < 0:
                item_idx += len(self)
            if item_idx < 0 or item_idx >= len(self):
                raise IndexError("Batch index out of range")
            return Interval(self.starts[item_idx], self.ends[item_idx], self.strands[item_idx])

        return Intervals(
            self.starts[item],
            self.ends[item],
            self.strands[item],
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        r"""Concatenate multiple [`Intervals`][serogen.core.interval.Intervals] collections into a single batch.

        Args:
            batches (Iterable[Intervals]): Iterable of [`Intervals`][serogen.core.interval.Intervals] objects to merge.

        Returns:
            Intervals: Concatenated [`Intervals`][serogen.core.interval.Intervals] collection.

        Raises:
            ValueError: If `batches` is empty.
        """
        batches_list = list(batches)
        if not batches_list:
            raise ValueError("Cannot concatenate empty list of batches")
        return cls(
            np.concatenate([b.starts for b in batches_list]),
            np.concatenate([b.ends for b in batches_list]),
            np.concatenate([b.strands for b in batches_list]),
        )

    @property
    def lengths(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        r"""Compute the lengths of all intervals in base pairs (`ends - starts`).

        Returns:
            np.ndarray[tuple[int], np.dtype[np.int32]]: 1D array of interval lengths.
        """
        return self.ends - self.starts
