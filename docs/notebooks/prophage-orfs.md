# Prophage & Bacterial ORF Benchmark

This presentation evaluates [`OrfPredictor`][serogen.genes.orf.predictor.OrfPredictor] across **10 diverse bacteriophage genomes** ($\lambda$, T4, T7, P1, P22, Mu, $\phi$X174, MS2, Bxb1, and CTX$\phi$) alongside the complete **_E. coli_ K-12 MG1655** chromosome (`U00096.3`, 4.64 Mb), comparing SeroGen's single-pass prophage-aware decoding against standard bacterial mode and Pyrodigal.

<div class="serogen-deck-toolbar">
  <span class="deck-meta"><strong>5 Slides</strong> · Interactive 16:9 Marimo Slide Deck · Navigate with ← / → arrow keys inside the deck</span>
  <div class="serogen-deck-actions">
    <a class="serogen-deck-btn" href="../assets/benchmark_prophage_orfs.html" target="_blank" rel="noopener">
      Present Fullscreen Slideshow ↗
    </a>
  </div>
</div>

<div class="serogen-slide-frame">
  <iframe src="../assets/benchmark_prophage_orfs.html" title="Prophage & Bacterial ORF Benchmark Slide Deck" loading="lazy" allowfullscreen></iframe>
</div>

---

## Key Takeaways

- **Zero Bacterial Regression**: Enabling `prophage=True` achieves **97.3% exact 3'-end recall** and **98.2% precision** on _E. coli_ K-12 MG1655 (`F1 = 0.9776`), matching standard bacterial mode while adding viral lysis/frameshift detection.
- **Lysis & Microphage Recovery**: Multi-pass hexamer self-training and the curated 60-family viral motif bank recover overlapping lysis genes (`Rz1` in $\lambda$, `lysB` in Bxb1) and ultra-compact ssDNA/ssRNA phages ($\phi$X174, MS2, CTX$\phi$) where single-genome bacterial training degrades.
- **4.2× Faster Whole-Genome Calling**: Single-pass Numba-compiled HMM-Viterbi decoding processes _E. coli_ K-12 in **~118 ms** vs. ~495 ms for Pyrodigal.

## Run Locally

```bash
# Present as interactive slideshow
uv run --with marimo --with altair marimo run scripts/benchmark_prophage_orfs.py

# Edit notebook cells
uv run --with marimo --with altair marimo edit scripts/benchmark_prophage_orfs.py
```
