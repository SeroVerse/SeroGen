# S. pneumoniae CPS Locus Serotyping

This presentation benchmarks [`LocusMatcher`][serogen.loci.matcher.LocusMatcher] and [`LayoutEngine`][serogen.core.graph.layout.LayoutEngine] across **98 complete _Streptococcus pneumoniae_ capsular polysaccharide (CPS) reference loci** (2,163 genes), comparing bipartite gene-level Hungarian matching against global DNA alignment (`MiniMap2`).

<div class="serogen-deck-toolbar">
  <span class="deck-meta"><strong>5 Slides</strong> · Interactive 16:9 Marimo Slide Deck · Navigate with ← / → arrow keys inside the deck</span>
  <div class="serogen-deck-actions">
    <a class="serogen-deck-btn" href="../assets/benchmark_spneumo_db.html" target="_blank" rel="noopener">
      Present Fullscreen Slideshow ↗
    </a>
  </div>
</div>

<div class="serogen-slide-frame">
  <iframe src="../assets/benchmark_spneumo_db.html" title="S. pneumoniae CPS Locus Serotyping Slide Deck" loading="lazy" allowfullscreen></iframe>
</div>

---

## Key Takeaways

- **100% Serogroup Cohesion**: Gene-level bipartite assignment (`LocusMatcher`) resolves structural rearrangements, mobile element insertions (`IS` transposases), and mosaic gene exchanges that fragment global nucleotide alignments.
- **Order-of-Magnitude Speedup**: Three-tier protein hashing + randstrobe seeding + banded SW-Gotoh executes full all-vs-all 98-locus comparison in sub-second wall-clock time on CPU.
- **Synteny Layout Optimization**: [`LayoutEngine`][serogen.core.graph.layout.LayoutEngine] automatically orients and orders reference loci to minimize cross-locus link inversions.

## Run Locally

```bash
# Present as interactive slideshow
uv run --with marimo --with altair marimo run scripts/benchmark_spneumo_db.py

# Edit notebook cells
uv run --with marimo --with altair marimo edit scripts/benchmark_spneumo_db.py
```
