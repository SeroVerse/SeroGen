r"""Protocol contracts and Numba kernels for Structure-of-Arrays (SoA) scalar views and batched containers.

This module defines formal protocols ([`ScalarView`][serogen.core.collections.ScalarView],
[`BatchedContainer`][serogen.core.collections.BatchedContainer], and
[`RaggedArrayContainer`][serogen.core.collections.RaggedArrayContainer]) and shared
offset/ragged-slicing mechanics that govern high-performance, vectorized containers across SeroGen.
"""

from collections.abc import Iterable, Iterator
from dataclasses import Field
from typing import Any, ClassVar, Protocol, Self, overload

import numpy as np
from numba import njit, prange


def lengths_to_offsets(
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
    r"""Compute 0-based exclusive prefix-sum offsets from a 1D array of element lengths.

    Args:
        lengths: 1D array of non-negative record lengths (`int32`).

    Returns:
        np.ndarray[tuple[int], np.dtype[np.int32]]: 1D `int32` array of starting offsets.
    """
    n = len(lengths)
    offsets = np.zeros(n, dtype=np.int32)
    if n > 1:
        np.cumsum(lengths[:-1], out=offsets[1:])
    return offsets


def normalize_indices(
    item: slice | np.ndarray[Any, Any] | list[Any],
    n: int,
) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
    r"""Normalize a slice, boolean mask, or integer index sequence into a 1D `int32` index array.

    Args:
        item: Slice, boolean mask array, or integer index sequence/array.
        n: Total number of records in the target container.

    Returns:
        np.ndarray[tuple[int], np.dtype[np.int32]]: Resolved 0-based `int32` indices.
    """
    if isinstance(item, slice):
        return np.arange(n, dtype=np.int32)[item]
    arr = np.asarray(item)
    if arr.dtype.kind == "b":
        return np.nonzero(arr)[0].astype(np.int32)
    if len(arr) == 0:
        return np.empty(0, dtype=np.int32)
    indices = arr.astype(np.int32, copy=False)
    if np.any(indices < 0):
        indices = np.where(indices < 0, indices + n, indices)
    return indices


@njit(cache=True, nogil=True, parallel=True)
def _gather_ragged_kernel(
    data: np.ndarray[tuple[int], Any],
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    new_offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    new_lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    indices: np.ndarray[tuple[int], np.dtype[np.int32]],
    total_len: int,
) -> np.ndarray[tuple[int], Any]:
    r"""Parallel Numba kernel to gather and compact ragged slices for any 1D NumPy dtype."""
    out = np.empty(total_len, dtype=data.dtype)
    n = len(indices)
    for i in prange(n):
        length_val = new_lengths[i]
        if length_val == 0:
            continue
        src_off = offsets[indices[i]]
        dst_off = new_offsets[i]
        for c in range(length_val):
            out[dst_off + c] = data[src_off + c]
    return out


class ScalarView(Protocol):
    r"""Formal structural contract for immutable scalar views into a [`BatchedContainer`][serogen.core.collections.BatchedContainer].

    All scalar views must be `@dataclass(slots=True, frozen=True)` records representing a
    single element extracted from a Structure-of-Arrays (SoA) batch container.
    """

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


class BatchedContainer[T: ScalarView, S: BatchedContainer[Any, Any]](Protocol):
    r"""A formal contract for Structure-of-Arrays (SoA) and batched containers.

    This protocol ensures that all high-performance vectorized collections provide a standard
    interface for instantiation, iteration, concatenation, and indexing without relying on slow
    dynamic mixins or runtime type introspection.

    Type Parameters:
        T: The [`ScalarView`][serogen.core.collections.ScalarView] dataclass returned when accessing a single index.
        S: The batched container type returned when indexing with slices or array masks.
    """

    def __len__(self) -> int:
        r"""Return the number of records in the batch.

        Returns:
            int: The total count of items in the container.
        """
        ...

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> T: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> S: ...

    @overload
    def __getitem__(self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]) -> T | S: ...

    def __getitem__(self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]) -> T | S:
        r"""Access records by index, slice, or boolean/integer array mask.

        Args:
            item (int | slice | npt.NDArray | list): An integer index, slice, or mask/indices array.

        Returns:
            T | S: A single scalar record (`T`) if an integer index is provided, or a new
                batched collection (`S`) if a slice or mask is provided.

        Raises:
            IndexError: If an integer index is out of bounds.
        """
        ...

    def __iter__(self) -> Iterator[T]:
        r"""Iterate over the container, yielding each scalar view `T` in order."""
        for i in range(len(self)):
            yield self[i]

    def _normalize_indices(
        self, item: slice | np.ndarray[Any, Any] | list[Any]
    ) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        r"""Resolve a slice, boolean mask, or index list into a 1D `int32` index array."""
        return normalize_indices(item, len(self))

    @classmethod
    def empty(cls) -> Self:
        r"""Create an empty, 0-length collection with correctly typed arrays.

        Returns:
            S: An empty instance of the container.
        """
        ...

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        r"""Concatenate multiple collections into a single, larger collection.

        Args:
            batches (Iterable[S]): An iterable of collections of the same type.

        Returns:
            S: A new, combined collection.

        Raises:
            ValueError: If the input iterable is empty or contains incompatible batches.
        """
        ...


class RaggedArrayContainer[T: ScalarView, S: BatchedContainer[Any, Any]](BatchedContainer[T, S], Protocol):
    r"""A formal contract for ragged Structure-of-Arrays (SoA) containers.

    Ragged containers store variable-length data (such as sequences, cluster members, or CIGAR
    operations) in a flat, contiguous memory layout, managing element boundaries using
    `offsets` and `lengths` arrays.
    """

    @property
    def offsets(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        r"""Starting indices into the flat data array for each record."""
        ...

    @property
    def lengths(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        r"""Lengths of each record."""
        ...

    @property
    def indptr(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        r"""CSR-compatible $(N + 1)$-length index pointer array (`[0, *cumsum(lengths)]`)."""
        n = len(self.offsets)
        if n == 0:
            return np.zeros(1, dtype=np.int32)
        out = np.empty(n + 1, dtype=np.int32)
        out[:n] = self.offsets
        out[n] = self.offsets[-1] + self.lengths[-1]
        return out

    @staticmethod
    def lengths_to_offsets(
        lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    ) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        r"""Compute 0-based starting offsets from a 1D `lengths` array."""
        return lengths_to_offsets(lengths)

    def _slice_ragged(
        self,
        item: slice | np.ndarray[Any, Any] | list[Any],
        *arrays: np.ndarray[tuple[int], Any],
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.int32]],
        np.ndarray[tuple[int], np.dtype[np.int32]],
        np.ndarray[tuple[int], np.dtype[np.int32]],
        tuple[np.ndarray[tuple[int], Any], ...],
    ]:
        r"""Slice or gather one or more parallel ragged data arrays while maintaining contiguous packing.

        Uses an $O(1)$ zero-copy view slice on `arrays` when `item` is a contiguous `slice(start, stop, 1)`,
        and falls back to the parallel Numba `_gather_ragged_kernel` for stepped slices, boolean masks,
        or arbitrary index arrays.

        Args:
            item: Slice, boolean mask, or integer index list/array.
            *arrays: Parallel 1D flat data arrays indexed by `self.offsets` and `self.lengths`.

        Returns:
            tuple: `(indices, new_offsets, new_lengths, sliced_arrays)`.
        """
        n = len(self)
        if isinstance(item, slice) and item.step in (None, 1):
            start, stop, _ = item.indices(n)
            if stop <= start:
                empty_i32 = np.empty(0, dtype=np.int32)
                return (
                    empty_i32,
                    empty_i32,
                    empty_i32,
                    tuple(np.empty(0, dtype=a.dtype) for a in arrays),
                )
            indices = np.arange(start, stop, dtype=np.int32)
            new_lengths = self.lengths[start:stop]
            base_offset = int(self.offsets[start])
            end_offset = int(self.offsets[stop - 1]) + int(self.lengths[stop - 1])
            new_offsets = self.offsets[start:stop] - base_offset
            return (
                indices,
                new_offsets,
                new_lengths,
                tuple(a[base_offset:end_offset] for a in arrays),
            )

        indices = self._normalize_indices(item)
        if len(indices) == 0:
            empty_i32 = np.empty(0, dtype=np.int32)
            return (
                indices,
                empty_i32,
                empty_i32,
                tuple(np.empty(0, dtype=a.dtype) for a in arrays),
            )

        new_lengths = self.lengths[indices].astype(np.int32, copy=False)
        new_offsets = lengths_to_offsets(new_lengths)
        total_len = int(new_offsets[-1] + new_lengths[-1])
        if total_len == 0:
            return (
                indices,
                new_offsets,
                new_lengths,
                tuple(np.empty(0, dtype=a.dtype) for a in arrays),
            )

        sliced = tuple(
            _gather_ragged_kernel(a, self.offsets, new_offsets, new_lengths, indices, total_len) for a in arrays
        )
        return indices, new_offsets, new_lengths, sliced
