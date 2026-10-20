"""Tests for GenomeAssembly loading, decompression, and container invariants."""

from compression import zstd
from pathlib import Path

import numpy as np
import pytest

from serogen.core.genome import GenomeAssembly

EXPECTED_ASSEMBLIES: dict[bytes, tuple[bytes, int]] = {
    b"PAO1": (b"NC_002516.2", 6_264_404),
    b"SGH10_chromosome": (b"NZ_CP025080.1", 5_485_114),
    b"STMC125": (b"NZ_CP104816.1", 5_192_560),
}


def test_genome_assembly_fixture_metadata(genome_assembly: GenomeAssembly) -> None:
    """Parsed `GenomeAssembly` fixtures must match expected contig accessions and nucleotide lengths."""
    assert genome_assembly.id in EXPECTED_ASSEMBLIES
    expected_contig_id, expected_len = EXPECTED_ASSEMBLIES[genome_assembly.id]

    assert len(genome_assembly) == 1
    assert genome_assembly.ids[0] == expected_contig_id
    assert int(genome_assembly.lengths[0]) == expected_len
    assert int(genome_assembly.offsets[0]) == 0
    assert len(genome_assembly.seqs) == expected_len
    assert genome_assembly.seqs.dtype == np.uint8


def test_genome_assembly_ensure_and_constructors(pao1_path: Path, pao1_assembly: GenomeAssembly) -> None:
    """`GenomeAssembly.ensure`, `from_stream`, and `from_fasta` must yield identical assembly data."""
    assert GenomeAssembly.ensure(pao1_assembly) is pao1_assembly

    from_path = GenomeAssembly.ensure(pao1_path)
    assert from_path.id == pao1_assembly.id
    assert np.array_equal(from_path.seqs, pao1_assembly.seqs)

    with zstd.open(pao1_path, mode="rb") as handle:
        raw_bytes = handle.read()

    from_bytes = GenomeAssembly.from_fasta(raw_bytes, id_=b"PAO1_custom")
    assert from_bytes.id == b"PAO1_custom"
    assert len(from_bytes) == len(pao1_assembly)
    assert np.array_equal(from_bytes.lengths, pao1_assembly.lengths)
    assert np.array_equal(from_bytes.seqs, pao1_assembly.seqs)


def test_genome_assembly_unsupported_extension(tmp_path: Path) -> None:
    """Unsupported file extensions must raise `NotImplementedError`."""
    bad_file = tmp_path / "invalid.txt"
    bad_file.write_bytes(b">seq1\nACGT\n")
    with pytest.raises(NotImplementedError, match="Unsupported format"):
        GenomeAssembly.from_file(bad_file)
