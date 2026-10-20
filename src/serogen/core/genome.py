from collections.abc import Callable, Iterable
from compression import bz2, gzip, lzma, zstd
from dataclasses import dataclass
from pathlib import Path
from re import Pattern
from re import compile as re_compile
from typing import IO, Any, ClassVar, Self

from serogen.core.io.fasta import FastaReader
from serogen.core.seq.seq import Sequence, Sequences


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class GenomeAssembly(Sequences):
    r"""Container for a genome assembly with support for transparent decompression.

    Stores contig sequence data as a contiguous memory block using [`Sequences`][serogen.core.seq.Sequences]
    and supports thread-safe, lazy-cached index creation for high-performance sequence alignment.

    Attributes:
        id (bytes): Unique genome assembly identifier.
    """

    _SEQUENCE_FILE_REGEX: ClassVar[Pattern[str]] = re_compile(
        r"\.(?P<ext>f(asta|a|na|fn|as))(\.(?P<compression>gz|bz2|xz|zst|zstd))?$"
    )
    _OPENERS: ClassVar[dict[str, Callable[..., Any]]] = dict(
        gz=gzip.open, bz2=bz2.open, xz=lzma.open, zst=zstd.open, zstd=zstd.open
    )

    id: bytes = b"unknown"

    @classmethod
    def ensure(cls, genome: Self | str | Path | IO[bytes]) -> Self:
        r"""Ensure the input object is coerced into a [`GenomeAssembly`][serogen.core.genome.GenomeAssembly].

        Args:
            genome (Self | str | Path | IO[bytes]): Existing assembly object, file path,
                or binary FASTA stream.

        Returns:
            GenomeAssembly: Validated or loaded [`GenomeAssembly`][serogen.core.genome.GenomeAssembly] instance.
        """
        if isinstance(genome, cls):
            return genome
        elif isinstance(genome, (str, Path)):
            return cls.from_file(genome)
        return cls.from_stream(genome)  # type: ignore

    @classmethod
    def from_fasta(
        cls,
        data: bytes | bytearray | memoryview,
        id_: bytes | str = b"unknown",
        *,
        truncate_id: bool = True,
        uppercase: bool = True,
    ) -> Self:
        r"""Construct a GenomeAssembly directly from raw binary FASTA data.

        Args:
            data (bytes | bytearray | memoryview): Raw binary FASTA contents.
            id_ (bytes | str): Genome assembly identifier.
            truncate_id (bool): If `True`, truncates header IDs at first whitespace.
            uppercase (bool): If `True`, normalizes sequence bytes to uppercase ASCII.

        Returns:
            GenomeAssembly: Constructed [`GenomeAssembly`][serogen.core.genome.GenomeAssembly] instance.
        """
        seqs = FastaReader.from_bytes(data, truncate_id=truncate_id, uppercase=uppercase).load()
        id_bytes = id_.encode("utf-8") if isinstance(id_, str) else bytes(id_)
        return cls(seqs.ids, seqs.seqs, seqs.offsets, seqs.lengths, id=id_bytes)

    @classmethod
    def from_file(cls, filepath: str | Path) -> Self:
        r"""Load a genome assembly from a FASTA file path.

        Supports plain `.fasta` / `.fa` / `.fna` files as well as compressed
        `.gz`, `.bz2`, and `.xz` files.

        Args:
            filepath (str | Path): Path to the target FASTA file.

        Returns:
            GenomeAssembly: Loaded [`GenomeAssembly`][serogen.core.genome.GenomeAssembly] instance.

        Raises:
            NotImplementedError: If the file format or compression extension is unsupported.
        """
        filepath = Path(filepath)
        if not (m := cls._SEQUENCE_FILE_REGEX.search(filepath.name)):
            raise NotImplementedError(f"Unsupported format: {filepath}")

        with cls._OPENERS.get(m.group("compression"), open)(filepath, mode="rb") as handle:
            stem = filepath.name.removesuffix(m.group()).encode("utf-8")
            return cls.from_stream(handle, id_=stem)

    @classmethod
    def from_stream(cls, handle: IO[bytes], id_: bytes | str | None = None) -> Self:
        r"""Load a genome assembly from an open binary stream.

        Args:
            handle (IO[bytes]): Open binary FASTA stream handle.
            id_ (bytes | str | None): Custom assembly identifier. If None, derived from
                handle name or defaults to `b'unknown'`.

        Returns:
            GenomeAssembly: Loaded [`GenomeAssembly`][serogen.core.genome.GenomeAssembly] instance.
        """
        if id_ is None:
            raw_name = getattr(handle, "name", "unknown")
            id_bytes = raw_name.encode("utf-8") if isinstance(raw_name, str) else bytes(raw_name)
        elif isinstance(id_, str):
            id_bytes = id_.encode("utf-8")
        else:
            id_bytes = bytes(id_)

        seqs = FastaReader(handle).load()
        return cls(seqs.ids, seqs.seqs, seqs.offsets, seqs.lengths, id=id_bytes)

    @classmethod
    def from_records(
        cls,
        records: list[Sequence] | tuple[Sequence, ...] | Iterable[Sequence],
        id_: bytes | str = b"unknown",
    ) -> Self:
        r"""Construct a [`GenomeAssembly`][serogen.core.genome.GenomeAssembly] from sequence records.

        Args:
            records (Iterable[Sequence]): Iterable of [`Sequence`][serogen.core.seq.Sequence] objects.
            id_ (bytes | str): Genome assembly identifier.

        Returns:
            GenomeAssembly: Constructed [`GenomeAssembly`][serogen.core.genome.GenomeAssembly] instance.
        """
        seqs = Sequences.from_records(records)
        id_bytes = id_.encode("utf-8") if isinstance(id_, str) else bytes(id_)
        return cls(seqs.ids, seqs.seqs, seqs.offsets, seqs.lengths, id=id_bytes)
