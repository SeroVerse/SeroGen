---
title: Benchmarks & Notebooks
subtitle: Interactive 16:9 Marimo Slide Decks
description: Reproducible Marimo notebooks and interactive 16:9 slide decks benchmarking single-pass bacterial & prophage ORF calling, 1D-CNN gene role classification, and S. pneumoniae de novo CPS database construction.
author: Tom Stanton
icon: lucide/presentation
comments: true
tags:
  - benchmarks
  - marimo
  - notebooks
---

# Benchmarks & Interactive Slide Decks

SeroGen maintains reproducible [Marimo](https://marimo.io) notebooks in [`scripts/`](https://github.com/SeroVerse/serogen/tree/main/scripts) covering every stage of the pipeline: single-pass bacterial & prophage ORF prediction, polysaccharide-specific 1D-CNN gene role training and layer sweeping, and de novo capsular polysaccharide (CPS) locus discovery, hybrid MCL clustering, and pan-locus database construction.

Each notebook is configured with a native **16:9 Marimo slide layout** (`scripts/layouts/*.slides.json`) and a cached session snapshot (`scripts/__marimo__/session/*.py.json`) that exports directly into the interactive slide decks embedded in this section.

<div class="grid cards" markdown>

-   :lucide-dna:{ .lg .middle } **Single-Pass Bacterial & Prophage ORF Benchmark**

    ---

    Evaluates [`OrfPredictor`][serogen.genes.orf.predictor.OrfPredictor] on *Escherichia coli* K-12 MG1655 (`CP014225.1`, `4.66 Mb`) and its 9 embedded cryptic/lambdoid prophages, demonstrating how 29-state frameshift/overlap decoding and sliding 4-kb GC adaptation recover `95.6%` of prophage 3'-stops (`+17` rescued ORFs) and `99.1%` of host CDSs in a single pass at `>60 Mb/s`.

    [:octicons-arrow-right-24: Explore ORF Benchmark](prophage-orfs.md)

-   :lucide-brain-circuit:{ .lg .middle } **Polysaccharide-Specific CNN Gene Role Classifier**

    ---

    Uncapped multi-source reference curation (`7,832` non-redundant `<= 80%` CD-HIT sequences across `UniProtKB`, `UniRef50`, `NCBI IPG`, and `KEGG`), Optuna TPE hyperparameter optimization, Numba-parallel 11-layer representation sweep, and dual-head validation (`87.89%` 29-role Macro F1 + `90.4%` 1-bit bacterial/phage source accuracy, `src/serogen/data/model.npz`), beating remote homology in the `< 25%` identity twilight zone.

    [:octicons-arrow-right-24: Explore Classifier Deck](gene-classifier.md)

-   :lucide-network:{ .lg .middle } **S. pneumoniae De Novo CPS Locus Discovery & Database**

    ---

    4-stage validation of de novo CPS locus discovery ([`LocusPredictor`][serogen.loci.predictor.LocusPredictor]), hybrid CNN-rescued MCL orthogroup clustering ([`Clusterer`][serogen.core.seq.clustering.Clusterer]), directed pan-locus graph construction ([`Loci.as_pan_graph`][serogen.core.graph.models.GraphBatch.as_pan_graph]), contig-split stitching (`100%` recovery across intergenic & mid-ORF breaks), and leave-one-out novelty detection ([`LocusMatcher`][serogen.loci.matcher.LocusMatcher]).

    [:octicons-arrow-right-24: Explore De Novo CPS Database Deck](spneumo-serotyping.md)

</div>

---

## Running Notebooks Locally

All notebooks can be launched interactively as reactive web apps, slide presentations, or editable Marimo notebooks via [`uv`](https://docs.astral.sh/uv/):

=== "Present as Slideshow"

    ```bash
    # Launch any notebook directly in Marimo's fullscreen slide presentation mode
    uv run --with marimo --with altair marimo run scripts/benchmark_prophage_orfs.py
    uv run --with marimo --with altair marimo run scripts/benchmark_spneumo_db.py
    uv run --with marimo --with altair --with optuna marimo run scripts/train_gene_classifier.py
    ```

=== "Interactive Editor"

    ```bash
    # Open in the reactive Marimo notebook editor
    uv run --with marimo --with altair marimo edit scripts/benchmark_prophage_orfs.py
    ```

=== "Refresh Cached Session Snapshot"

    ```bash
    # Re-execute a notebook and update its cached session snapshot in scripts/__marimo__/session/
    uv run --with marimo --with altair marimo export session --no-sandbox --force-overwrite scripts/benchmark_prophage_orfs.py
    ```
