# CNN Functional Gene Role Classifier

This presentation documents the reference dataset curation, **Optuna TPE hyperparameter optimization**, and held-out validation of [`GeneRoleClassifier`][serogen.genes.cnn.GeneRoleClassifier] across **3,996 non-redundant (`<= 80%` CD-HIT) bacterial proteins** spanning all **29 functional [`GeneRole`][serogen.genes.roles.GeneRole]s** plus open-set `UNKNOWN` (`Role 29`) decoy rejection.

<div class="serogen-deck-toolbar">
  <span class="deck-meta"><strong>6 Slides</strong> · Interactive 16:9 Marimo Slide Deck · Navigate with ← / → arrow keys inside the deck</span>
  <div class="serogen-deck-actions">
    <a class="serogen-deck-btn" href="../assets/train_gene_classifier.html" target="_blank" rel="noopener">
      Present Fullscreen Slideshow ↗
    </a>
  </div>
</div>

<div class="serogen-slide-frame">
  <iframe src="../assets/train_gene_classifier.html" title="CNN Functional Gene Role Classifier Slide Deck" loading="lazy" allowfullscreen></iframe>
</div>

---

## Key Takeaways

- **92.10% Held-Out Test Accuracy / 91.97% Macro F1**: Multi-scale 1D-CNN (`embed_dim=64`, `num_filters=192`, `kernel_size=(5, 9, 13)`, `spatial_zones=2`) selected via 20-trial Optuna TPE search achieves 92.3% macro precision and 92.1% macro recall across all 29 functional `GeneRole`s.
- **Dedicated Optuna TPE Optimization Slide**: Visualizes the trial-by-trial validation Macro F1 trajectory, best-so-far frontier, hyperparameter importance share, and Top-5 trial configurations.
- **Automatic Production Model Export**: Running `scripts/train_gene_classifier.py` refits the winning Optuna architecture on the full curated reference set and saves the production weights directly to `src/serogen/data/model.npz` (`545.8 KB`).

## Run Locally

```bash
# Present as interactive slideshow
uv run --with marimo --with altair --with optuna marimo run scripts/train_gene_classifier.py

# Edit notebook cells
uv run --with marimo --with altair --with optuna marimo edit scripts/train_gene_classifier.py
```
