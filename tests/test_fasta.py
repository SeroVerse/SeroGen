"""Tests for FastaReader streaming and vectorized batch loading."""

from compression import zstd
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from serogen.core.io.fasta import FastaReader


def test_fasta_reader_stream_and_load_consistency(assembly_path: Path) -> None:
    """Streaming iteration and batch `load()` over real genome assemblies must produce identical sequences."""
    with zstd.open(assembly_path, mode="rb") as handle, FastaReader(handle, chunk_size=64 * 1024) as reader:
        streamed = list(reader)

    with zstd.open(assembly_path, mode="rb") as handle, FastaReader(handle) as reader:
        batch = reader.load()

    assert len(streamed) == len(batch)
    assert len(batch) >= 1

    for i, rec in enumerate(streamed):
        assert rec.id == batch.ids[i]
        assert len(rec.seq) == int(batch.lengths[i])
        assert rec.seq == batch[i].seq


def test_fasta_reader_stateful_next_and_small_chunks() -> None:
    """Stateful `next()` calls across tiny chunk boundaries must preserve headers and sequences."""
    raw = (
        b"  # leading comment before first record\n"
        b">contig_1 description text\n"
        b"acgt\r\n"
        b"tgca\n"
        b">contig_2\tsecond header\n"
        b"ggcc>embedded_gt\n"
        b"aatt\n"
    )
    reader = FastaReader(BytesIO(raw), chunk_size=5)

    first = next(reader)
    assert first.id == b"contig_1"
    assert first.seq == b"ACGTTGCA"

    second = next(reader)
    assert second.id == b"contig_2"
    assert second.seq == b"GGCC>EMBEDDED_GTAATT"

    with pytest.raises(StopIteration):
        next(reader)


def test_fasta_reader_from_bytes_options() -> None:
    """`FastaReader.from_bytes` must respect `truncate_id` and `uppercase` flags in both stream and batch modes."""
    raw = b">seqA full description\nacgt\n>seqB second\nTGca\n"

    batch = FastaReader.from_bytes(raw, truncate_id=False, uppercase=False).load()
    assert len(batch) == 2
    assert batch[0].id == b"seqA full description"
    assert batch[0].seq == b"acgt"
    assert batch[1].id == b"seqB second"
    assert batch[1].seq == b"TGca"

    streamed = list(FastaReader.from_bytes(memoryview(raw), truncate_id=True, uppercase=True))
    assert [r.id for r in streamed] == [b"seqA", b"seqB"]
    assert [r.seq for r in streamed] == [b"ACGT", b"TGCA"]


def test_fasta_reader_empty_inputs() -> None:
    """Empty or headerless buffers must return 0 records when streamed and `Sequences.empty()` when loaded."""
    assert list(FastaReader.from_bytes(b"")) == []
    empty_batch = FastaReader.from_bytes(b"no fasta header here\njust text").load()
    assert len(empty_batch) == 0
    assert empty_batch.seqs.dtype == np.uint8
