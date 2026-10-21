r"""OrfPredictor high-level execution engine for sequences and assemblies."""

import numpy as np

from serogen.core.genome import GenomeAssembly
from serogen.core.interval import Intervals
from serogen.core.seq.alphabet import Alphabet, GeneticCode
from serogen.core.seq.seq import Sequence, Sequences
from serogen.genes.models import HMM, Genes, Indels, get_hmm_model

from .extract import (
    _build_window_table,
    _decode_and_count_windows,
    _write_windows_parallel,
)


class OrfPredictor:
    r"""High-performance Open Reading Frame (ORF) predictor.

    Executes Numba JIT-compiled FragGeneScan HMM decoding over single sequences,
    [`Sequences`][serogen.core.seq.Sequences], or [`GenomeAssembly`][serogen.core.genome.GenomeAssembly]
    containers using a unified flat window table and Numba `prange` parallelism, returning
    batched [`Genes`][serogen.genes.models.Genes]. Supports bacterial and viral/phage translation
    tables and single-pass co-prediction of bacterial and prophage/phage ORFs.

    Attributes:
        whole_genome (bool): Flag indicating complete genome mode (with start codon refinement)
            or fragmented / short-read mode.
        model (HMM): Baked HMM model parameter container.
        genetic_code (GeneticCode): Translation table defining start/stop codons and amino acid mappings.
        include_phage (bool): If True, enables single-pass phage/prophage ORF co-prediction
            (sliding local GC adaptation, overlapping operon transitions, and small ORF recovery).
        min_gene_len (int): Minimum gene length in nucleotides.
        local_gc_window (int): Sliding window size in bp for local GC bin adaptation.
    """

    def __init__(
        self,
        whole_genome: bool = False,
        model: HMM | None = None,
        window_size: int = 50_000,
        overlap: int = 5_000,
        genetic_code: GeneticCode | int | str = GeneticCode.bacterial,
        include_phage: bool = True,
        min_gene_len: int | None = None,
        local_gc_window: int = 4_000,
    ) -> None:
        r"""Initialize an [`OrfPredictor`][serogen.genes.orf.predictor.OrfPredictor] instance.

        Args:
            whole_genome (bool): If True, enables whole genome gene finder mode. Defaults to False.
            model (HMM | None): Optional custom HMM parameters. If None, uses default baked model.
            window_size (int): Size in base pairs for local GC-adaptive sliding windows on long contigs.
                Set to 0 to disable windowing. Defaults to 50_000.
            overlap (int): Overlap size in base pairs between consecutive windows to ensure seamless
                gene boundary detection. Defaults to 5_000.
            genetic_code (GeneticCode | int | str): Translation table (instance, NCBI table ID, or name).
                Defaults to `GeneticCode.bacterial` (Table 11).
            include_phage (bool): If True, enables single-pass bacterial + phage ORF co-prediction.
                Defaults to True.
            min_gene_len (int | None): Optional minimum ORF length threshold in base pairs. If None,
                defaults to 90 bp in phage-aware whole-genome mode, 120 bp in bacterial-only
                whole-genome mode, or 60 bp in fragmented mode.
            local_gc_window (int): Sliding window size in base pairs for local GC bin adaptation
                when `include_phage=True`. Defaults to 4_000.
        """
        self.whole_genome = whole_genome
        self.model = model if model is not None else get_hmm_model()
        self.window_size = window_size
        self.overlap = overlap
        self.genetic_code = (
            genetic_code if isinstance(genetic_code, GeneticCode) else GeneticCode.from_table(genetic_code)
        )
        self.include_phage = include_phage
        if min_gene_len is not None:
            self.min_gene_len = int(min_gene_len)
        else:
            self.min_gene_len = (90 if include_phage else 120) if whole_genome else 60
        self.local_gc_window = int(local_gc_window)

    def _predict_flat_batch(
        self,
        flat_seqs: np.ndarray,  # type: ignore[type-arg]
        contig_offsets: np.ndarray,  # type: ignore[type-arg]
        contig_lengths: np.ndarray,  # type: ignore[type-arg]
    ) -> Genes:
        r"""Decode all windows across all contigs in parallel via Numba `prange` and build SoA `Genes`."""
        n_contigs = len(contig_lengths)
        if n_contigs == 0 or len(flat_seqs) == 0:
            return Genes.empty()

        min_contig_len = self.min_gene_len if self.whole_genome else min(self.min_gene_len, 60)
        win_table = _build_window_table(
            contig_offsets,
            contig_lengths,
            self.window_size,
            self.overlap,
            min_contig_len,
        )

        n_windows = len(win_table[0])
        if n_windows == 0:
            return Genes.empty()

        m = self.model
        gc = self.genetic_code
        encode_lut = Alphabet.DNA.kmer_lut
        lut_rc = Alphabet.DNA.complement_lut
        viterbi_params = (m.tr, m.tr_mi, m.tr_ii, m.pi)
        score_params = (
            m.e_match,
            m.tr_rr,
            m.tr_start,
            m.tr_stop,
            m.dist_start,
            m.dist_stop,
            gc.start_mask_64,
            gc.stop_mask_64,
            gc.rev_start_mask_64,
            gc.rev_stop_mask_64,
            gc.start_log_priors_64,
            gc.rev_start_log_priors_64,
            gc.stop_log_priors_64,
            gc.rev_stop_log_priors_64,
        )

        (
            encoded_flat,
            vpath_flat,
            cg_bins_flat,
            win_gene_counts,
            win_cds_bytes,
            win_indel_counts,
        ) = _decode_and_count_windows(
            flat_seqs,
            encode_lut,
            lut_rc,
            win_table,
            self.whole_genome,
            self.include_phage,
            self.min_gene_len,
            self.local_gc_window,
            viterbi_params,
            score_params,
        )

        total_genes = int(np.sum(win_gene_counts))
        if total_genes == 0:
            return Genes.empty()

        total_cds = int(np.sum(win_cds_bytes))
        total_indels = int(np.sum(win_indel_counts))

        win_gene_offsets = np.zeros(n_windows, dtype=np.int32)
        win_cds_offsets = np.zeros(n_windows, dtype=np.int32)
        win_indel_offsets = np.zeros(n_windows, dtype=np.int32)
        if n_windows > 1:
            np.cumsum(win_gene_counts[:-1], out=win_gene_offsets[1:])
            np.cumsum(win_cds_bytes[:-1], out=win_cds_offsets[1:])
            np.cumsum(win_indel_counts[:-1], out=win_indel_offsets[1:])

        out_starts = np.empty(total_genes, dtype=np.int32)
        out_ends = np.empty(total_genes, dtype=np.int32)
        out_strands = np.empty(total_genes, dtype=np.int8)
        out_partials = np.empty(total_genes, dtype=np.bool_)
        out_contig_indices = np.empty(total_genes, dtype=np.uint32)

        out_cds_seqs = np.empty(total_cds, dtype=np.uint8)
        out_cds_offsets = np.empty(total_genes, dtype=np.int32)
        out_cds_lengths = np.empty(total_genes, dtype=np.int32)

        out_indel_pos = np.empty(total_indels, dtype=np.int32)
        out_indel_kinds = np.empty(total_indels, dtype=np.int8)
        out_indel_offsets = np.empty(total_genes, dtype=np.int32)
        out_indel_lengths = np.empty(total_genes, dtype=np.int32)

        win_offsets = (win_gene_offsets, win_gene_counts, win_cds_offsets, win_indel_offsets)
        out_bufs = (
            out_starts,
            out_ends,
            out_strands,
            out_partials,
            out_contig_indices,
            out_cds_seqs,
            out_cds_offsets,
            out_cds_lengths,
            out_indel_pos,
            out_indel_kinds,
            out_indel_offsets,
            out_indel_lengths,
        )

        _write_windows_parallel(
            flat_seqs,
            encoded_flat,
            vpath_flat,
            cg_bins_flat,
            lut_rc,
            win_table,
            win_offsets,
            self.whole_genome,
            self.include_phage,
            self.min_gene_len,
            score_params,
            out_bufs,
        )

        ids_arr = np.arange(total_genes).astype("S")
        return Genes(
            intervals=Intervals(starts=out_starts, ends=out_ends, strands=out_strands),
            seqs=Sequences(ids_arr, out_cds_seqs, out_cds_offsets, out_cds_lengths),
            contig_indices=out_contig_indices,
            indels=Indels(
                positions=out_indel_pos,
                kinds=out_indel_kinds,
                offsets=out_indel_offsets,
                lengths=out_indel_lengths,
            ),
            is_partial=out_partials,
            genetic_code=gc,
        )

    def predict(
        self,
        target: bytes | str | Sequence | Sequences | GenomeAssembly,
    ) -> Genes:
        r"""Predict Open Reading Frames across target sequences.

        Args:
            target (bytes | str | Sequence | Sequences | GenomeAssembly): Input genomic sequence(s).

        Returns:
            Genes: Batched [`Genes`][serogen.genes.models.Genes] container containing interval coordinates,
                in-frame coding sequences (5' -> 3'), contig assignments, frameshift indels, and `is_partial` mask.

        Raises:
            TypeError: If target is not a supported sequence type.
        """
        if isinstance(target, (str, bytes)):
            raw_arr = np.frombuffer(target.encode("ascii") if isinstance(target, str) else target, dtype=np.uint8)
            return self._predict_flat_batch(
                raw_arr,
                np.zeros(1, dtype=np.int32),
                np.array([len(raw_arr)], dtype=np.int32),
            )

        if isinstance(target, Sequence):
            raw_arr = np.frombuffer(target.seq, dtype=np.uint8)
            return self._predict_flat_batch(
                raw_arr,
                np.zeros(1, dtype=np.int32),
                np.array([len(raw_arr)], dtype=np.int32),
            )

        if isinstance(target, (Sequences, GenomeAssembly)):
            if len(target) == 0:
                return Genes.empty()
            return self._predict_flat_batch(target.seqs, target.offsets, target.lengths)

        raise TypeError(f"Unsupported target type for ORF prediction: {type(target)}")
