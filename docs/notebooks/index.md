# Benchmarks & Interactive Slide Decks

SeroGen maintains reproducible [Marimo](https://marimo.io) notebooks in [`scripts/`](https://github.com/SeroVerse/serogen/tree/main/scripts) covering every stage of the pipeline: single-pass prophage & bacterial ORF prediction, deep CNN functional gene role training, and whole-database polysaccharide locus matching.

Each notebook is configured with a native **16:9 Marimo slide layout** (`scripts/layouts/*.slides.json`) and a cached session snapshot (`scripts/__marimo__/session/*.py.json`) that exports directly into the interactive slide decks embedded in this section.

<div class="grid cards" markdown>

-   :lucide-dna:{ .lg .middle } **Prophage & Bacterial ORF Benchmark**

    ---

    Head-to-head evaluation of `OrfPredictor` against standard bacterial ORF calling across complete *Klebsiella pneumoniae* chromosomes (`HS11286`, `NTUH-K2044`) and integrated prophage islands (`99.7%` whole-chromosome 3'-stop recall, `95.3%` prophage recall).

    [:octicons-arrow-right-24: Explore ORF Benchmark](prophage-orfs.md)

-   :lucide-brain-circuit:{ .lg .middle } **CNN Functional Gene Role Classifier**

    ---

    Multi-source reference curation (`3,996` non-redundant proteins), 20-trial Optuna TPE hyperparameter optimization, and held-out validation (`92.10%` accuracy / `91.97%` Macro F1) across all 29 functional `GeneRole`s (`src/serogen/data/model.npz`).

    [:octicons-arrow-right-24: Explore Classifier Deck](gene-classifier.md)

-   :lucide-network:{ .lg .middle } **S. pneumoniae CPS Locus Serotyping**

    ---

    4-stage validation of MCL homology clustering, Fruchterman-Reingold pan-locus synteny layout (`LayoutEngine`), whole-genome serotyping (`100%` Top-1 accuracy), contig-split stitching (`FRAGMENTED_VARIANT`), and leave-one-out novelty detection.

    [:octicons-arrow-right-24: Explore CPS Serotyping Deck](spneumo-serotyping.md)

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
