# Release Notes

All notable changes to **`SeroGen`** are documented on this page. This project adheres to [Semantic Versioning](https://semver.org/) and the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) specification.

---

## [Unreleased]

### Added

- **Structure-of-Arrays (`serogen.core`)**:
  - `BatchedContainer` and `RaggedArrayContainer` zero-copy memory protocols (`serogen.core.collections`).
  - 2-bit DNA/RNA (`Alphabet.DNA`) and `MMSEQS12` reduced amino-acid alphabets with ragged translation across NCBI translation tables (`serogen.core.seq.alphabet`).
  - `RandstrobeIndex` ($O(1)$ strobemer seeding) and `FracMinHashSketch` k-mer indices (`serogen.core.seq.kmers`).
  - Vectorized unbanded/banded Smith-Waterman-Gotoh local and semi-global alignment (`PairwiseAligner` in `serogen.core.seq.align`).
  - Hybrid sequence + active-site CNN motif-rescued protein clustering (`Clusterer` supporting greedy incremental and sparse Markov Clustering `MCL` modes in `serogen.core.seq.clustering`).
  - Packed CSR directed multigraphs (`GraphModel`, `GraphBatch`), Consensus Pan-Graph compression (`GraphBatch.as_pan_graph`), Hungarian + L2-QP `GraphAligner`, branch-and-bound `maximum_spanning_arborescence`, `RelationalMPNN`, and Fruchterman-Reingold `LayoutEngine` (`serogen.core.graph`).
- **Single-Pass ORF & CNN Role Engines (`serogen.genes`)**:
  - 29-state frameshift- and operon-overlap-aware profile-HMM Viterbi `OrfPredictor` with sliding 4-kb GC adaptation and single-pass viral/prophage gap rescue (`serogen.genes.orf.predictor`).
  - 30-class `GeneRole` ontology (`29` functional polysaccharide biosynthesis, transport, modification, regulatory, and MGE roles + open-set `UNKNOWN`) and pure NumPy/Numba multi-scale 1D-CNN + 20-D biophysical `GeneRoleClassifier` (`serogen.genes.cnn`).
- **De Novo Polysaccharide Locus Discovery (`serogen.loci`)**:
  -Exact All-Contiguous-Partitions Dynamic Programming (`ACP-DP`) + multi-locus Viterbi `LocusSegmenter` (`serogen.loci.segmenter`).
  - `LocusPredictor` with `TRANSPOSASE`/`INTEGRASE` MGE insertion bridging, optional coordinate healing (`heal_mobile_elements`), and cross-contig boundary stitching (`stitch_contig_splits`).
  - `LocusMatcher` for de novo orthogroup database construction and `MatchType` structural variant bitflag classification (`serogen.loci.matcher`).
- **Interactive Benchmarks & Documentation**:
  - Reproducible 16:9 Marimo slide decks (`benchmark_prophage_orfs.py`, `train_gene_classifier.py`, `benchmark_spneumo_db.py`) and unified SeroVerse Zensical documentation.
