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

This presentation documents the multi-source reference curation, **Optuna TPE hyperparameter optimization**, **Numba-accelerated layer representation sweep**, and held-out validation of [`GeneRoleClassifier`][serogen.genes.cnn.GeneRoleClassifier] across **non-redundant (`<= 80%` CD-HIT) bacterial (`taxonomy_id:2`) and bacteriophage/prophage (`taxonomy_id:10239`) proteins** spanning all **29 functional [`GeneRole`][serogen.genes.roles.GeneRole]s**, a joint **1-bit bacterial/phage source head**, and open-set `UNKNOWN` (`Role 29`) biological decoy rejection.

## Why Polysaccharide-Specific CNN Classification Beats Remote Homology

Bacterial surface polysaccharide loci (capsular polysaccharides `CPS`, O-antigens `O_AG`, wall teichoic acids `WTA`, and exopolysaccharides `EPS`) evolve under intense diversifying selection from bacteriophages and host immunity:

1. **Sequence Divergence Beyond BLAST / MMseqs2 / Profile-HMM Sensitivity**: Essential membrane machinery — such as `Wzx`/`GtrA` flippases (`FLIPPASE`), `Wzy` polymerases (`POLYMERASE`), `WaaL` O-antigen ligases (`LIGASE`), and serotype-defining glycosyltransferases (`GLYCOSYLTRANSFERASE`) — routinely diverge into the sequence "twilight and midnight zones" (`< 20–25%` amino-acid identity) where k-mer seeding and alignment bitscores fail.
2. **Conserved Catalytic Motifs & Emergent Biophysical Topology**: Despite extreme global sequence divergence, these enzymes strictly conserve short local catalytic signatures (e.g., `DxD`, `ED`, and `EX7E` glycosyltransferase catalytic triads, Rossmann-fold `GxGxxG` nucleotide-sugar binding loops, and `LytR-CpsA-Psr` motifs) alongside whole-protein transmembrane hydropathy and fold repeat profiles.
3. **Pure Emergent Dual-Pooled 1D-CNN + Joint 1-Bit Source Head**: [`GeneRoleClassifier`][serogen.genes.cnn.GeneRoleClassifier] combines learnable amino-acid embeddings (`embed_dim=64`), parallel multi-scale 1D convolutional branches (`num_filters=192`, `kernel_size=(5, 9, 13)`), and dual global max (`conv_max`, peak local catalytic and viral motifs) + length-normalized mean (`conv_avg`, global fold/transmembrane density) pooling (`feature_dim = 2 * num_filters = 384`), driving both the 29-class `GeneRole` readout (`dense_w_`) and a joint **1-bit genomic source readout** (`source_w_`, `predict_source` / `predict_source_proba`: `0` = bacterial, `1` = bacteriophage/prophage) while keeping [`ArchetypeTensors`][serogen.loci.archetypes.ArchetypeTensors] 32-bit `uint32` role bitmasks untouched.
4. **Active-Site Embeddings for Twilight-Zone Clustering Rescue**: Intermediate representations extracted via [`GeneRoleClassifier.transform(seqs, layer="conv_max")`][serogen.genes.cnn.GeneRoleClassifier.transform] (default in [`Loci.cluster_genes`][serogen.loci.models.Loci.cluster_genes]) capture catalytic motif activations and feed directly into [`Clusterer`][serogen.core.seq.clustering.Clusterer] (`embedding_weight`, `motif_rescue_threshold`) to rescue divergent intra-role homologs during Markov Clustering (MCL).

<div class="serogen-slide-frame">
  <iframe src="assets/train_gene_classifier.html" title="Polysaccharide-Specific CNN Gene Role Classifier Slide Deck" loading="lazy" allowfullscreen></iframe>
</div>

---

## Architectural Assessments & Conclusions

### 1. 11-Layer Representation Sweep (`transform` Benchmark on 1,565 Held-Out Proteins)

Across all 11 representations exposed by [`GeneRoleClassifier.transform`][serogen.genes.cnn.GeneRoleClassifier.transform], **unweighted `penultimate`** (`[conv_max, conv_avg]`, `384-D`) and **unweighted `conv_max`** (`192-D`) achieve the highest overall clustering and retrieval scores, outperforming class-separation-weighted `task_weighted`:

| Layer | Dim | 1-NN Cosine Recall | Top-5 Role Purity | Nearest-Centroid Macro F1 | Fisher Ratio | Composite Score |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`penultimate` (`[conv_max, conv_avg]`)** | 384 | **84.66%** | 82.84% | **87.69%** | 1.339 | **85.06** |
| **`conv_max` (default in `Loci.cluster_genes`)** | 192 | **84.60%** | 82.75% | **87.68%** | 1.309 | **85.01** |
| `task_weighted` | 384 | 83.26% | 83.14% | 87.39% | 1.044 | 84.60 |
| `task_weighted_conv_max` | 192 | 83.51% | 82.65% | 86.90% | 1.031 | 84.35 |
| `conv_k13` (single-kernel max+avg) | 128 | 81.98% | 79.31% | 86.26% | 1.384 | 82.52 |
| `conv_k9` (single-kernel max+avg) | 128 | 81.34% | 79.08% | 85.47% | 1.374 | 81.96 |
| `logits` | 29 | 82.56% | 83.48% | 74.08% | 1.036 | 80.04 |
| `proba` | 29 | 78.79% | 82.77% | 76.29% | 1.883 | 79.28 |
| `conv_k5` (single-kernel max+avg) | 128 | 78.40% | 76.23% | 82.61% | 1.430 | 79.08 |
| `conv_avg` | 192 | 70.99% | 67.39% | 52.81% | 0.631 | 63.73 |
| `embed_mean` | 64 | 61.98% | 55.64% | 34.43% | 0.253 | 50.68 |

- **Why unweighted `conv_max` / `penultimate` beat `task_weighted` for MCL clustering**: Multiplying features by `||dense_w_||_2` emphasizes inter-role boundaries at the expense of intra-role sequence family geometry (`-1.40%` 1-NN recall, Fisher ratio `1.339 -> 1.044`). Using `embedding_layer="conv_max"` in [`Loci.cluster_genes`][serogen.loci.models.Loci.cluster_genes] preserves true sequence-family topology while still achieving `87.68%` Nearest-Centroid Macro F1 at half the dimensionality (`192-D`).

### 2. Dual-Head Weight-Norm Division of Labor (`dense_w_` vs. `source_w_`)

Inspecting the Frobenius squared weight norms (`||W||_F²`) of the trained linear readouts reveals how the shared `[conv_max, conv_avg]` representation serves two distinct biological tasks simultaneously:

| Component / Receptive Field | 29-Role Biosynthetic Head (`dense_w_`) | 1-Bit Phage Source Head (`source_w_`) | Biological Interpretation |
| :--- | ---: | ---: | :--- |
| **`conv_max` (`0:192`)** | **65.75%** | **82.23%** | Peak local catalytic (`DxD`, `EX7E`, Walker A/B) & viral signature motifs |
| **`conv_avg` (`192:384`)** | **34.25%** | **17.77%** | Whole-protein fold & transmembrane helix density (`Wzx`/`Wzy`, β-barrels) |
| **Kernel `k = 5 aa`** | 24.03% | 21.89% | Short active-site turns & catalytic triads |
| **Kernel `k = 9 aa`** | 34.35% | 34.85% | Medium Rossmann loops & HTH DNA-binding helices |
| **Kernel `k = 13 aa`** | **41.63%** | **43.26%** | Extended transmembrane spans & viral structural/tailspike motifs |

### 3. Ablation of the 25-D Length & Amino-Acid Frequency Tail

Removing the hand-crafted 25-D scalar tail (`5` length scalars + `20` global amino-acid frequencies) yielded strictly positive gains across both classification and representation geometry:

- **Eliminated Cosine-Space Scale Domination**: In the previous `409-D` vector, raw length/frequency scalars dominated Euclidean and cosine distances, depressing `conv_avg`. Removing the tail boosted `conv_avg` 1-NN recall by **+12.08%** (`58.91% -> 70.99%`) and Nearest-Centroid Macro F1 by **+19.89%** (`32.92% -> 52.81%`).
- **Higher Validation & Holdout Generalization**: Optuna inner 3-fold validation Macro F1 increased by **+1.86%** (`86.80% -> 88.66%`), with held-out test Macro F1 reaching **87.89%** (`88.2%` Macro Precision, `88.0%` Macro Recall) across `1,565` proteins.
- **Short IS Helper & Transposase ORFs Improved Without Length Scalars**: Widening `TRANSPOSASE` curation to `70–600 aa` and adding IS helper/ORFA domains (`PF01527`, `PF13333`, `PF01548`, `PF02371`) raised `TRANSPOSASE` holdout recall to **92.2%** (`87.4%` F1), while `GLYCOSYLTRANSFERASE` F1 rose **+4.0%** (`79.2% -> 83.2%`) and `LIGASE` (`WaaL`) F1 rose **+11.8%** (`73.9% -> 85.7%`).

### 4. 1-Bit Phage Detection & Real Biological Open-Set `UNKNOWN` Rejection

- **90.4% Bacterial vs. Phage Source Accuracy**: [`GeneRoleClassifier.predict_source`][serogen.genes.cnn.GeneRoleClassifier.predict_source] and [`GeneRoleClassifier.predict_source_proba`][serogen.genes.cnn.GeneRoleClassifier.predict_source_proba] achieve **90.4% accuracy** (`73.8%` Macro F1) across held-out proteins and biological decoys (`995` curated phage/prophage proteins), enabling downstream pipelines to flag prophage seroconversion cassettes (*gtrABC*, *oac*), phage tailspike depolymerases (`LYASE`), and prophage genomic islands.
- **75.3% Biological Decoy Rejection at `p < 0.50`**: Supervising open-set `GeneRole.UNKNOWN` (`Role 29`) with curated bacterial housekeeping and phage structural/packaging/lysis decoys rejects **75.3%** of real biological decoys at `confidence_threshold=0.50` while retaining **92.6%** of true biosynthetic test proteins (`66.9%` biological decoy rejection vs. `42.9%` shuffled decoy rejection at `threshold=0.438`).

---

## Key Takeaways

- **Uncapped Non-Redundant 29-Role + Phage Curation**: Curates all `7,832` non-redundant (`<= 80%` identity via [`Clusterer(mode="cdhit")`][serogen.core.seq.clustering.Clusterer]) bacterial and bacteriophage/prophage proteins across **UniProtKB (Pfam/InterPro)**, **UniRef50**, **NCBI IPG**, and **KEGG Orthology** without artificial per-role truncation caps, using proportional `80% / 20%` stratified train/test splitting.
- **Optuna TPE Tuning & Numba Layer Representation Sweep**: Evaluates trial-by-trial validation Macro F1, hyperparameter importance share, and a Numba-parallel (`@njit(parallel=True)`) 11-layer representation sweep (`penultimate`, `task_weighted`, `task_weighted_conv_max`, `conv_max`, `conv_avg`, `embed_mean`, `logits`, `proba`, `conv_k5`, `conv_k9`, `conv_k13`) scored by 1-NN cosine recall, Top-5 role purity, nearest-centroid Macro F1, and Fisher discriminability.
- **Biological + Shuffled Open-Set `UNKNOWN` Rejection**: Supervises and evaluates open-set `GeneRole.UNKNOWN` (`Role 29`) rejection and 1-bit `source` classification on real non-biosynthetic biological decoys (bacterial housekeeping/flanking proteins + phage terminase/portal/capsid/tail/holin/endolysin proteins) alongside shuffled decoys, exporting the production weights to `src/serogen/data/model.npz`.

## Run Locally

```bash
# Present as interactive slideshow
uv run --with marimo --with altair --with optuna marimo run scripts/train_gene_classifier.py

# Edit notebook cells
uv run --with marimo --with altair --with optuna marimo edit scripts/train_gene_classifier.py
```
