"""High-performance C-level and zero-copy NumPy FASTA I/O."""

from collections.abc import Iterator
from io import BufferedIOBase, BytesIO, RawIOBase
from typing import IO, Self

import numpy as np

from serogen.core.seq.seq import Sequence, Sequences


class FastaReader(Iterator[Sequence]):
    r"""High-performance stateful FASTA stream iterator and vectorized batch loader.

    Streams binary FASTA handles in chunks, splitting on `\n>` boundaries and applying
    C-level `bytes.translate` to yield [`Sequence`][serogen.core.seq.seq.Sequence] instances
    with $O(1)$ memory overhead, or loads an entire stream at once into a contiguous
    [`Sequences`][serogen.core.seq.seq.Sequences] Structure-of-Arrays container via [`load`][serogen.core.io.fasta.FastaReader.load].

    Args:
        handle: Open binary stream (`IO[bytes]`) containing FASTA sequence data.
        truncate_id: If `True`, truncates header IDs at the first whitespace character.
        uppercase: If `True`, normalizes sequence bytes to uppercase ASCII.
        chunk_size: Read buffer size in bytes for streaming iteration (default 4 MiB).
    """

    # Precomputed 256-byte C translation table for single-pass uppercase + whitespace deletion
    _UPPER_TABLE: bytes = bytes.maketrans(b"abcdefghijklmnopqrstuvwxyz", b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    _WHITESPACE: bytes = b"\r\n\t "

    __slots__ = (
        "_handle",
        "_truncate_id",
        "_uppercase",
        "_chunk_size",
        "_stream",
    )

    def __init__(
        self,
        handle: IO[bytes] | BufferedIOBase | RawIOBase,
        truncate_id: bool = True,
        uppercase: bool = True,
        chunk_size: int = 4 * 1024 * 1024,
    ) -> None:
        """Initialize the stateful FASTA reader."""
        self._handle = handle
        self._truncate_id = truncate_id
        self._uppercase = uppercase
        self._chunk_size = chunk_size
        self._stream: Iterator[Sequence] = self._iter_records()

    @classmethod
    def from_bytes(
        cls,
        data: bytes | bytearray | memoryview,
        truncate_id: bool = True,
        uppercase: bool = True,
        chunk_size: int = 4 * 1024 * 1024,
    ) -> Self:
        """Construct a [`FastaReader`][serogen.core.io.fasta.FastaReader] from an in-memory binary buffer.

        Args:
            data: Raw binary FASTA buffer (`bytes`, `bytearray`, or `memoryview`).
            truncate_id: If `True`, truncates header IDs at the first whitespace character.
            uppercase: If `True`, normalizes sequence bytes to uppercase ASCII.
            chunk_size: Read buffer size in bytes for streaming iteration.

        Returns:
            FastaReader: Reader wrapping the binary buffer.
        """
        return cls(
            BytesIO(data),
            truncate_id=truncate_id,
            uppercase=uppercase,
            chunk_size=chunk_size,
        )

    def __iter__(self) -> Self:
        """Return the stateful iterator object itself."""
        return self

    def __next__(self) -> Sequence:
        """Fetch the next [`Sequence`][serogen.core.seq.seq.Sequence] record from the FASTA stream."""
        return next(self._stream)

    def __enter__(self) -> Self:
        """Enter the runtime context for the reader."""
        return self

    def __exit__(self, *_: object) -> None:
        """Exit the runtime context and close the underlying binary stream."""
        self.close()

    def close(self) -> None:
        """Close the underlying binary stream."""
        self._handle.close()

    def _parse_header(self, header: bytes) -> bytes:
        """Extract sequence identifier from a raw FASTA header line (without leading `>`)."""
        if self._truncate_id:
            parts = header.split(None, 1)
            return parts[0] if parts else b""
        return header.strip()

    def _iter_records(self) -> Iterator[Sequence]:
        """Stateful generator yielding `Sequence` records across chunk boundaries."""
        table = self._UPPER_TABLE if self._uppercase else None
        buf = b""
        started = False

        while True:
            chunk = self._handle.read(self._chunk_size)
            if not chunk:
                break
            buf += chunk

            if not started:
                if buf[0] == 62:  # ord('>')
                    buf = buf[1:]
                    started = True
                else:
                    nl_gt = buf.find(b"\n>")
                    if nl_gt == -1:
                        buf = buf[-1:] if buf.endswith(b"\n") else b""
                        continue
                    buf = buf[nl_gt + 2 :]
                    started = True

            parts = buf.split(b"\n>")
            if len(parts) > 1:
                for rec in parts[:-1]:
                    header, _, seq_raw = rec.partition(b"\n")
                    yield Sequence(
                        id=self._parse_header(header),
                        seq=seq_raw.translate(table, self._WHITESPACE),
                    )
                buf = parts[-1]

        if started and buf:
            header, _, seq_raw = buf.partition(b"\n")
            yield Sequence(
                id=self._parse_header(header),
                seq=seq_raw.translate(table, self._WHITESPACE),
            )

    def load(self) -> Sequences:
        r"""Load the FASTA stream into memory as a contiguous [`Sequences`][serogen.core.seq.seq.Sequences] batch.

        Uses C-level `bytes.split`, `bytes.partition`, and 256-byte lookup table `bytes.translate`
        combined with a single `bytearray.join` allocation and zero-copy `np.frombuffer` view.

        Returns:
            Sequences: Packed Structure-of-Arrays container of all parsed sequences.
        """
        raw = self._handle.read()
        if not raw:
            return Sequences.empty()

        # Locate first valid '>' at start of buffer or immediately following a newline
        if raw[0] == 62:  # ord('>')
            first_gt = 0
        else:
            nl_gt = raw.find(b"\n>")
            if nl_gt == -1:
                return Sequences.empty()
            first_gt = nl_gt + 1

        # Split records at line-initial '>' boundaries using C memmem
        records = raw[first_gt + 1 :].split(b"\n>")
        n = len(records)

        table = self._UPPER_TABLE if self._uppercase else None
        ids_list: list[bytes] = []
        seq_parts: list[bytes] = []

        for rec in records:
            header, _, seq_raw = rec.partition(b"\n")
            ids_list.append(self._parse_header(header))
            seq_parts.append(seq_raw.translate(table, self._WHITESPACE))

        ids = np.array(ids_list, dtype="S")
        lengths = np.fromiter((len(s) for s in seq_parts), dtype=np.int32, count=n)
        offsets = np.zeros(n, dtype=np.int32)
        if n > 1:
            np.cumsum(lengths[:-1], out=offsets[1:])

        # Single C-level buffer allocation + zero-copy writable uint8 NumPy view
        joined_buf = bytearray().join(seq_parts)
        seqs = np.frombuffer(joined_buf, dtype=np.uint8) if len(joined_buf) > 0 else np.empty(0, dtype=np.uint8)
        return Sequences(ids, seqs, offsets, lengths)
