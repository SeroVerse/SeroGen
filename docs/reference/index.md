# API Reference

The **SeroGen** API is organized into three modules:

- **[`serogen.core`](core.md)** — Structure-of-Arrays (SoA) container protocols, genomic intervals, FASTA I/O, sequence/alphabet kernels, randstrobe/FracMinHash k-mer indices, banded Smith-Waterman-Gotoh pairwise alignment, protein clustering, and bipartite graph models.
- **[`serogen.genes`](genes.md)** — Single-pass bacterial & prophage ORF calling ([`OrfPredictor`][serogen.genes.orf.predictor.OrfPredictor]), functional role ontology ([`GeneRole`][serogen.genes.roles.GeneRole]), and 1D CNN protein classification ([`GeneRoleClassifier`][serogen.genes.cnn.GeneRoleClassifier]).
- **[`serogen.loci`](loci.md)** — Biosynthetic locus archetypes ([`LocusArchetype`][serogen.loci.archetypes.LocusArchetype]), ACP-DP locus segmentation ([`LocusSegmenter`][serogen.loci.segmenter.LocusSegmenter]), high-level locus prediction ([`LocusPredictor`][serogen.loci.predictor.LocusPredictor]), and serotype matching ([`LocusMatcher`][serogen.loci.matcher.LocusMatcher]).

---

::: serogen
    options:
      members: false
