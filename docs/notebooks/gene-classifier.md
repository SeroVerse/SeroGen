---
title: CNN Gene Role Classifier
subtitle: Multi-Scale 1D-CNN & Layer Sweep
description: Polysaccharide-specific multi-scale 1D-CNN + biophysical protein role classifier training, Optuna TPE optimization, and Numba layer representation sweep.
author: Tom Stanton
icon: lucide/cpu
status: new
comments: true
tags:
  - benchmarks
  - cnn
  - gene-roles
  - optuna
---

# Polysaccharide-Specific CNN Gene Role Classifier

This presentation documents the multi-source reference curation, **Optuna TPE hyperparameter optimization**, **Numba-accelerated layer representation sweep**, and held-out validation of [`GeneRoleClassifier`][serogen.genes.cnn.GeneRoleClassifier] across **non-redundant (`<= 80%` CD-HIT) bacterial proteins** spanning all **29 functional [`GeneRole`][serogen.genes.roles.GeneRole]s** plus open-set `UNKNOWN` (`Role 29`) decoy rejection.

## Why Polysaccharide-Specific CNN Classification Beats Remote Homology

Bacterial surface polysaccharide loci (capsular polysaccharides `CPS`, O-antigens `O_AG`, wall teichoic acids `WTA`, and exopolysaccharides `EPS`) evolve under intense diversifying selection from bacteriophages and host immunity:

1. **Sequence Divergence Beyond BLAST / MMseqs2 / Profile-HMM Sensitivity**: Essential membrane machinery — such as `Wzx` flippases (`FLIPPASE`), `Wzy` polymerases (`POLYMERASE`), `WaaL` O-antigen ligases (`LIGASE`), and serotype-defining glycosyltransferases (`GLYCOSYLTRANSFERASE`) — routinely diverge into the sequence "twilight and midnight zones" (`< 20–25%` amino-acid identity) where k-mer seeding and alignment bitscores fail.
2. **Conserved Catalytic Motifs & Biophysical Topology**: Despite extreme global sequence divergence, these enzymes strictly conserve short local catalytic signatures (e.g., `DxD`, `ED`, and `EX7E` glycosyltransferase catalytic triads, Rossmann-fold `GxGxxG` nucleotide-sugar binding loops, and `LytR-CpsA-Psr` motifs) alongside global transmembrane hydropathy, aromaticity, and net charge profiles.
3. **Multi-Scale 1D-CNN + Biophysical Descriptor Head**: [`GeneRoleClassifier`][serogen.genes.cnn.GeneRoleClassifier] combines learnable amino-acid embeddings (`embed_dim=64`), three parallel 1D convolutional motif branches (`num_filters=192`, `kernel_size=(3, 5, 9)`), dual global max + length-normalized mean pooling, and a 20-dimensional biophysical descriptor vector (12-class `MMSEQS12` composition + Kyte-Doolittle hydropathy mean/std, transmembrane helix fraction, net charge, pI proxy, aromaticity, tiny/bulky ratio, and log-length) to classify divergent polysaccharide proteins in milliseconds.
4. **Active-Site Embeddings for Twilight-Zone Clustering Rescue**: Intermediate representations extracted via [`GeneRoleClassifier.transform(seqs, layer="conv_max")`][serogen.genes.cnn.GeneRoleClassifier.transform] capture catalytic motif activations and feed directly into [`Clusterer`][serogen.core.seq.clustering.Clusterer] (`embedding_weight`, `motif_rescue_threshold`) to rescue divergent intra-role homologs during Markov Clustering (MCL).

<div class="serogen-deck-toolbar">
  <span class="deck-meta"><strong>6 Slides</strong> · Interactive 16:9 Marimo Slide Deck · Navigate with ← / → arrow keys inside the deck</span>
  <div class="serogen-deck-actions">
    <a class="serogen-deck-btn" href="../assets/train_gene_classifier.html" target="_blank" rel="noopener">
      Present Fullscreen Slideshow ↗
    </a>
  </div>
</div>

<div class="serogen-slide-frame">
  <iframe src="../assets/train_gene_classifier.html" title="Polysaccharide-Specific CNN Gene Role Classifier Slide Deck" loading="lazy" allowfullscreen></iframe>
</div>

---

## Key Takeaways

- **High-Precision 29-Role Classification**: Curates non-redundant (`<= 80%` identity via [`Clusterer(mode="cdhit")`][serogen.core.seq.clustering.Clusterer]) bacterial proteins across **UniProtKB (Pfam/InterPro)**, **UniRef50**, **NCBI IPG**, and **KEGG Orthology**, achieving **>90% held-out test accuracy and Macro F1** across all 29 biosynthetic, transport, modification, and MGE (`TRANSPOSASE`, `INTEGRASE`) roles (`~3.4s` full refit via LUT-factored Numba SoA backward kernels).
- **Optuna TPE Tuning & Numba Layer Representation Sweep**: Evaluates trial-by-trial validation Macro F1, hyperparameter importance share, and a Numba-parallel (`@njit(parallel=True)`) 8-layer representation sweep (`embed_mean`, `embed_max`, `conv_max`, `conv_mean`, `pooled`, `features_norm`, `task_weighted`, `logits`) scored by 1-NN cosine recall, Top-5 role purity, nearest-centroid Macro F1, and Fisher discriminability.
- **Open-Set `UNKNOWN` Rejection & Automatic Model Export**: Rejects composition-matched shuffled decoys as `GeneRole.UNKNOWN` (`Role 29`) via softmax confidence thresholding and exports the production weights directly to `src/serogen/data/model.npz`.

## Run Locally

```bash
# Present as interactive slideshow
uv run --with marimo --with altair --with optuna marimo run scripts/train_gene_classifier.py

# Edit notebook cells
uv run --with marimo --with altair --with optuna marimo edit scripts/train_gene_classifier.py
```
