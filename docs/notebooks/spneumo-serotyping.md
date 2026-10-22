---
title: S. pneumoniae De Novo CPS Database
subtitle: Hybrid MCL, Pan-Graph & Contig Stitching
description: End-to-end Streptococcus pneumoniae de novo capsular polysaccharide (CPS) locus discovery, CNN-rescued MCL clustering, pan-locus graph construction, and contig-split stitching.
author: Tom Stanton
icon: lucide/boxes
status: new
comments: true
tags:
  - benchmarks
  - cps
  - mcl
  - pan-graph
---

# S. pneumoniae De Novo CPS Locus Discovery & Database Construction

This presentation validates [`LocusPredictor`][serogen.loci.predictor.LocusPredictor], [`Clusterer`][serogen.core.seq.clustering.Clusterer], [`Loci`][serogen.loci.models.Loci] ([`GraphBatch`][serogen.core.graph.models.GraphBatch]), and [`LocusMatcher`][serogen.loci.matcher.LocusMatcher] across a 4-stage **_Streptococcus pneumoniae_ capsular polysaccharide (CPS) locus discovery and de novo database construction** pipeline.

Rather than relying on pre-trimmed reference databases or global nucleotide alignment, SeroGen extracts CPS biosynthetic operons **de novo** from unannotated sequences and whole chromosomes (including fragmented/contig-split assemblies), classifies divergent polysaccharide genes with [`GeneRoleClassifier`][serogen.genes.cnn.GeneRoleClassifier], clusters them into functional orthogroups via hybrid Markov Clustering ([`Clusterer(mode="mcl")`][serogen.core.seq.clustering.Clusterer]), and compresses the resulting [`Loci`][serogen.loci.models.Loci] batch into an indexed de novo database and directed consensus pan-locus graph ([`Loci.as_pan_graph`][serogen.core.graph.models.GraphBatch.as_pan_graph]).

<div class="serogen-deck-toolbar">
  <span class="deck-meta"><strong>5 Slides</strong> · Interactive 16:9 Marimo Slide Deck · Navigate with ← / → arrow keys inside the deck</span>
  <div class="serogen-deck-actions">
    <a class="serogen-deck-btn" href="../assets/benchmark_spneumo_db.html" target="_blank" rel="noopener">
      Present Fullscreen Slideshow ↗
    </a>
  </div>
</div>

<div class="serogen-slide-frame">
  <iframe src="../assets/benchmark_spneumo_db.html" title="S. pneumoniae De Novo CPS Locus Discovery & Database Slide Deck" loading="lazy" allowfullscreen></iframe>
</div>

---

## Key Takeaways

- **De Novo CPS Locus Discovery Across Mechanisms (`LocusPredictor`)**: Automatically segments both Wzy-dependent (`CPS_WZY`, anchored by `wzg`/`cpsA`–`wze`/`cpsD`, `wzx`, and `wzy`) and synthase-dependent (`CPS_SYNTHASE`, e.g., Type 3 capsule) loci directly from unannotated sequences and complete RefSeq chromosomes (`100%` recall on encapsulated strains and `100%` specificity on the unencapsulated `R6` rough negative control).
- **Hybrid CNN-Rescued MCL Orthogroup Clustering (`Clusterer`)**: [`Loci.cluster_genes`][serogen.loci.models.Loci.cluster_genes] combines [`RandstrobeIndex`][serogen.core.seq.kmers.RandstrobeIndex] strobemer seeding, `MMSEQS12` reduced-alphabet $12^4$ CSR + ungapped BLOSUM62 diagonal prefiltering, SIMD-parallel Smith-Waterman-Gotoh ([`PairwiseAligner`][serogen.core.seq.pairwise.PairwiseAligner]), **role-gated [`GeneRoleClassifier`][serogen.genes.cnn.GeneRoleClassifier] active-site motif rescue**, and sparse Markov Clustering (`mode="mcl"`) to group divergent glycosyltransferases (`GT`), flippases (`Wzx`), and polymerases (`Wzy`) across the sequence twilight zone.
- **Zero-Copy Pan-Locus Graph Data Structures (`Loci` / `GraphBatch`)**: Bit-packed 32-bit [`GeneNode`][serogen.loci.models.GeneNode]s (`CLUSTER_ID`, `ROLE_ID`, `STRAND`, `IS_FRAMESHIFT`) and [`IntergenicEdge`][serogen.loci.models.IntergenicEdge]s (`ORIENTATION`, `GAP_BIN`) compress into a directed consensus pan-CPS CSR network via [`Loci.as_pan_graph()`][serogen.core.graph.models.GraphBatch.as_pan_graph] (visualized in 2D using [`LayoutEngine`][serogen.core.graph.layout.LayoutEngine]).
- **Fragmented Contig-Split Stitching & Novelty Detection**: `LocusPredictor(stitch_contig_splits=True)` achieves **100% recovery** across artificial intergenic and mid-ORF contig breaks (`MatchType.FRAGMENTED_VARIANT`), while [`LocusMatcher`][serogen.loci.matcher.LocusMatcher] flags held-out novel locus architectures via `Match.is_novel` (`MatchType.GENE_VARIANT`) and falls back to the closest intra-serogroup relative.

## Run Locally

```bash
# Present as interactive slideshow
uv run --with marimo --with altair marimo run scripts/benchmark_spneumo_db.py

# Edit notebook cells
uv run --with marimo --with altair marimo edit scripts/benchmark_spneumo_db.py
```
