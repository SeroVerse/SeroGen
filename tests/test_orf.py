"""Tests for OrfPredictor and Genes containers using parsed GenomeAssembly fixtures."""

import numpy as np

from serogen.core.genome import GenomeAssembly
from serogen.genes import Gene, Genes, OrfPredictor

EXPECTED_MIN_ORFS: dict[bytes, int] = {
    b"PAO1": 5_500,
    b"SGH10_chromosome": 5_000,
    b"STMC125": 4_500,
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
    # Extract a 100 kb slice of SGH10 for fast translation and slicing checks
    sub_seq = sgh10_assembly[0].seq[:100_000]
    genes = orf_predictor.predict(sub_seq)

    assert len(genes) > 50

    proteins = genes.translations
    assert len(proteins) == len(genes)
    # Each translated protein length matches its nucleotide CDS codon count (minus terminal stop if stripped)
    assert np.all(proteins.lengths > 0)
    assert np.all(proteins.lengths <= (genes.seqs.lengths // 3))

    # Slicing Genes preserves aligned intervals, sequences, and translations
    subset = genes[:10]
    assert isinstance(subset, Genes)
    assert len(subset) == 10
    assert np.array_equal(subset.intervals.starts, genes.intervals.starts[:10])
    assert len(subset.translations) == 10

    # Scalar indexing and BatchedContainer iteration yield Gene scalar views
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
    from serogen.core.seq import Sequences

    predictor = OrfPredictor(whole_genome=True)
    sub_seq = sgh10_assembly[0].seq[:120_000]
    intact_genes = predictor.predict(sub_seq)
    assert len(intact_genes) > 80

    # Pick a long interior gene (>900 bp) on the forward strand and cleave directly through its midpoint
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

    # The last gene on contig 0 and first gene on contig 1 must be flagged as partial
    assert bool(c0_genes[-1].is_partial) is True
    assert bool(c1_genes[0].is_partial) is True

    # Interior genes away from contig boundaries must not be flagged as partial
    if len(c0_genes) > 2:
        assert not np.any(c0_genes.is_partial[1:-1])
    if len(c1_genes) > 2:
        assert not np.any(c1_genes.is_partial[1:-1])

    # Every extracted CDS (including partial contig-edge ORFs) must be a multiple of 3 bp
    assert np.all(split_genes.seqs.lengths % 3 == 0)
    # And the cleaved 3'-truncated gene on contig 0 must match its genomic interval length (when no frameshifts)
    if len(c0_genes[-1].indels) == 0:
        assert len(c0_genes[-1].seq) == len(c0_genes[-1].interval)


def test_viral_genetic_codes_and_phage_orf_prediction(
    sgh10_assembly: GenomeAssembly,
) -> None:
    """Viral translation tables and single-pass phage-aware OrfPredictor must decode and translate accurately."""
    from serogen.core.seq import GeneticCode, Sequence

    assert GeneticCode.from_table(11) is GeneticCode.bacterial
    assert GeneticCode.from_table("viral") is GeneticCode.viral
    assert GeneticCode.from_table(15) is GeneticCode.viral_amber_q
    assert GeneticCode.from_table(4) is GeneticCode.viral_opal_w
    assert GeneticCode.from_table(25) is GeneticCode.viral_opal_g

    # Verify amber (TAG -> Q) and opal (TGA -> W / G) recoding in translation and 64-codon stop masks
    cds = Sequence(b"cod", b"ATGAAATAGAAATGA")
    assert bytes(cds.translate(code=GeneticCode.viral, to_stop=False).seq) == b"MK*K*"
    assert bytes(cds.translate(code=GeneticCode.viral_amber_q, to_stop=False).seq) == b"MKQK*"
    assert bytes(cds.translate(code=GeneticCode.viral_opal_w, to_stop=False).seq) == b"MK*KW"
    assert bytes(cds.translate(code=GeneticCode.viral_opal_g, to_stop=False).seq) == b"MK*KG"

    # Verify 64-codon stop masks reflect recoded stop codons
    assert int(np.sum(GeneticCode.viral.stop_mask_64)) == 3
    assert int(np.sum(GeneticCode.viral_amber_q.stop_mask_64)) == 2
    assert int(np.sum(GeneticCode.viral_opal_w.stop_mask_64)) == 2
    assert int(np.sum(GeneticCode.viral_opal_g.stop_mask_64)) == 2

    # Single-pass phage-aware prediction vs bacterial-only mode on a 100 kb genomic slice
    sub_seq = sgh10_assembly[0].seq[:100_000]
    pred_phage = OrfPredictor(whole_genome=True, genetic_code=GeneticCode.viral, include_phage=True)
    pred_bact = OrfPredictor(whole_genome=True, genetic_code=GeneticCode.bacterial, include_phage=False)

    genes_phage = pred_phage.predict(sub_seq)
    genes_bact = pred_bact.predict(sub_seq)

    assert genes_phage.genetic_code is GeneticCode.viral
    assert len(genes_phage) >= len(genes_bact)
    assert np.all(genes_phage.seqs.lengths % 3 == 0)
    assert np.all(genes_phage.seqs.lengths >= 90)
    assert len(genes_phage.translations) == len(genes_phage)
