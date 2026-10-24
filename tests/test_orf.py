"""Tests for OrfPredictor and Genes containers using parsed GenomeAssembly fixtures."""

import numpy as np

from serogen.core.genome import GenomeAssembly
from serogen.core.seq.alphabet import GeneticCode
from serogen.core.seq.seq import Sequence, Sequences
from serogen.genes.models import Gene, Genes
from serogen.genes.orf.predictor import OrfPredictor

EXPECTED_MIN_ORFS: dict[bytes, int] = {
    b"PAO1": 5_200,
    b"SGH10_chromosome": 4_600,
    b"STMC125": 4_400,
}


def test_orf_predictor_on_genome_assembly(
    genome_assembly: GenomeAssembly,
    orf_predictor: OrfPredictor,
) -> None:
    """`OrfPredictor.predict` on each parsed `GenomeAssembly` fixture must return valid `Genes` containers."""
    genes = orf_predictor.predict(genome_assembly)

    assert isinstance(genes, Genes)
    min_expected = EXPECTED_MIN_ORFS[genome_assembly.id]
    assert len(genes) >= min_expected

    # Interval invariants: 0 <= start < end <= contig_length, strand in {-1, +1}
    contig_len = int(genome_assembly.lengths[0])
    assert np.all(genes.intervals.starts >= 0)
    assert np.all(genes.intervals.ends > genes.intervals.starts)
    assert np.all(genes.intervals.ends <= contig_len)
    assert np.all((genes.intervals.strands == 1) | (genes.intervals.strands == -1))

    # Contig index assignment
    assert len(genes.contig_indices) == len(genes)
    assert np.all(genes.contig_indices == 0)

    # Extracted coding sequences must be non-empty and in-frame (multiples of 3 bp)
    assert len(genes.seqs) == len(genes)
    assert np.all(genes.seqs.lengths >= 60)
    assert np.all(genes.seqs.lengths % 3 == 0)

    # Indels container must match gene count
    assert len(genes.indels) == len(genes)
    assert len(genes.has_frameshift) == len(genes)


def test_orf_predictor_translations_and_slicing(
    sgh10_assembly: GenomeAssembly,
    orf_predictor: OrfPredictor,
) -> None:
    """`Genes.translations` and container slicing must work on predicted ORFs from a genome sub-region."""
    sub_seq = sgh10_assembly[0].seq[:100_000]
    genes = orf_predictor.predict(sub_seq)

    assert len(genes) > 50

    proteins = genes.translations
    assert len(proteins) == len(genes)
    assert np.all(proteins.lengths > 0)
    assert np.all(proteins.lengths <= (genes.seqs.lengths // 3))

    subset = genes[:10]
    assert isinstance(subset, Genes)
    assert len(subset) == 10
    assert np.array_equal(subset.intervals.starts, genes.intervals.starts[:10])
    assert len(subset.translations) == 10

    first_gene = subset[0]
    assert isinstance(first_gene, Gene)
    assert first_gene.interval == subset.intervals[0]
    assert first_gene.seq == subset.seqs[0]
    assert first_gene.contig_index == 0
    assert first_gene.is_partial == bool(subset.is_partial[0])
    assert list(subset[:3]) == [subset[0], subset[1], subset[2]]


def test_orf_predictor_mid_orf_contig_split_and_partial_flag(
    sgh10_assembly: GenomeAssembly,
) -> None:
    """Cleaving a chromosome mid-ORF across contigs must flag contig-edge partial ORFs and avoid duplicate stop codons."""
    predictor = OrfPredictor()
    sub_seq = sgh10_assembly[0].seq[:120_000]
    intact_genes = predictor.predict(sub_seq)
    assert len(intact_genes) > 80

    fwd_long = np.nonzero((intact_genes.intervals.strands == 1) & (intact_genes.intervals.lengths >= 900))[0]
    assert len(fwd_long) > 0
    target_idx = int(fwd_long[len(fwd_long) // 2])
    g_start = int(intact_genes.intervals.starts[target_idx])
    g_end = int(intact_genes.intervals.ends[target_idx])
    cut_pos = (g_start + g_end) // 2

    split_batch = Sequences.from_bytes([sub_seq[:cut_pos], sub_seq[cut_pos:]])
    split_genes = predictor.predict(split_batch)

    c0_mask = split_genes.contig_indices == 0
    c1_mask = split_genes.contig_indices == 1
    assert np.any(c0_mask) and np.any(c1_mask)

    c0_genes = split_genes[c0_mask]
    c1_genes = split_genes[c1_mask]

    assert bool(c0_genes[-1].is_partial) is True
    assert bool(c1_genes[0].is_partial) is True

    if len(c0_genes) > 2:
        assert not np.any(c0_genes.is_partial[1:-1])
    if len(c1_genes) > 2:
        assert not np.any(c1_genes.is_partial[1:-1])

    assert np.all(split_genes.seqs.lengths % 3 == 0)
    if len(c0_genes[-1].indels) == 0:
        assert len(c0_genes[-1].seq) == len(c0_genes[-1].interval)


def test_genetic_codes_and_local_gc_orf_prediction(
    sgh10_assembly: GenomeAssembly,
) -> None:
    """Standard and bacterial translation tables and local-GC OrfPredictor must decode and translate accurately."""
    assert GeneticCode.from_table(1) is GeneticCode.standard
    assert GeneticCode.from_table("standard") is GeneticCode.standard
    assert GeneticCode.from_table(11) is GeneticCode.bacterial
    assert GeneticCode.from_table("bacterial") is GeneticCode.bacterial

    cds = Sequence(b"cod", b"ATGAAATAGAAATGA")
    assert bytes(cds.translate(code=GeneticCode.bacterial, to_stop=False).seq) == b"MK*K*"
    assert bytes(cds.translate(code=GeneticCode.standard, to_stop=False).seq) == b"MK*K*"
    assert int(np.sum(GeneticCode.bacterial.stop_mask_64)) == 3
    assert int(np.sum(GeneticCode.standard.stop_mask_64)) == 3

    sub_seq = sgh10_assembly[0].seq[:100_000]
    pred = OrfPredictor(genetic_code=GeneticCode.bacterial, min_gene_len=120, local_gc_window=4_000)
    genes = pred.predict(sub_seq)

    assert genes.genetic_code is GeneticCode.bacterial
    assert len(genes) > 50
    assert np.all(genes.seqs.lengths % 3 == 0)
    assert np.all(genes.seqs.lengths >= 120)
    assert len(genes.translations) == len(genes)


def test_orf_predictor_frameshift_and_stop_readthrough(
    sgh10_assembly: GenomeAssembly,
) -> None:
    """`OrfPredictor` must stitch 1-bp indels and premature stop codons unconditionally."""
    sub_seq = bytes(sgh10_assembly[0].seq[:25_000])
    pred = OrfPredictor()
    intact_genes = pred.predict(sub_seq)

    fwd_long = np.nonzero((intact_genes.intervals.strands == 1) & (intact_genes.intervals.lengths >= 1200))[0]
    assert len(fwd_long) > 0
    g_idx = int(fwd_long[0])
    g_start = int(intact_genes.intervals.starts[g_idx])
    g_end = int(intact_genes.intervals.ends[g_idx])
    intact_aa_len = int(intact_genes.translations.lengths[g_idx])
    mid = (g_start + g_end) // 2
    mid -= (mid - g_start) % 3

    del_seq = sub_seq[:mid] + sub_seq[mid + 1 :]
    ins_seq = sub_seq[:mid] + b"A" + sub_seq[mid:]
    stop_seq = sub_seq[:mid] + b"TAA" + sub_seq[mid + 3 :]

    for mut_seq, expected_len_delta in ((del_seq, -1), (ins_seq, 0), (stop_seq, 0)):
        mut_genes = pred.predict(mut_seq)
        matches = np.nonzero(
            (np.abs(mut_genes.intervals.starts - g_start) <= 30) & (np.abs(mut_genes.intervals.ends - g_end) <= 30)
        )[0]
        assert len(matches) == 1
        m_idx = int(matches[0])
        assert bool(mut_genes.has_frameshift[m_idx]) is True
        assert len(mut_genes.indels[m_idx]) >= 1
        assert abs(int(mut_genes.translations.lengths[m_idx]) - (intact_aa_len + expected_len_delta)) <= 2
        assert b"*" not in bytes(mut_genes.translations[m_idx].seq)
