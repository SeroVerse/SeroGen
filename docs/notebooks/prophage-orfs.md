---
title: Prophage & Bacterial ORFs
subtitle: 29-State Viterbi HMM on E. coli K-12
description: Single-pass bacterial and prophage ORF prediction benchmark on Escherichia coli K-12 MG1655 (CP014225.1) and 9 embedded prophage islands.
author: Tom Stanton
icon: lucide/zap
comments: true
tags:
  - benchmarks
  - orf-calling
  - prophages
  - hmm
---

# Single-Pass Bacterial & Prophage ORF Benchmark

This presentation evaluates [`OrfPredictor`][serogen.genes.orf.predictor.OrfPredictor] on the complete **_Escherichia coli_ str. K-12 substr. MG1655** gold-standard reference chromosome (`GenBank: CP014225.1`, `4,656,178 bp`) and its **9 structurally embedded cryptic and lambdoid prophage islands** (`CP4-44`, `CPS-53`, `CPZ-55`, `CP4-57`, `CP4-6`, `DLP12`, `e14`, `Rac`, and `Qin`).

Horizontally acquired prophages frequently encode seroconverting polysaccharide-modification cassettes (such as *gtrABC* glucosyltransferases and *oac* O-acetyltransferases) characterized by sharp local AT-enrichment (`34%–44%` GC vs. `50.8%` host GC), short (`90–120 bp`) accessory ORFs, and tightly coupled `1-bp` (`TGATG`/`TAATG`) or `4-bp` (`ATGA`) stop-start operon overlaps that standard single-genome bacterial ORF callers miss.

<div class="serogen-slide-frame">
  <iframe src="assets/benchmark_prophage_orfs.html" title="Single-Pass Bacterial & Prophage ORF Benchmark Slide Deck" loading="lazy" allowfullscreen></iframe>
</div>

---

## Key Takeaways

- **29-State Frameshift & Operon-Overlap Profile-HMM**: [`OrfPredictor`][serogen.genes.orf.predictor.OrfPredictor] decodes forward and reverse coding frames, Shine-Dalgarno RBS motifs, `1-bp`/`2-bp` sequencing or pseudogene frameshift indels (`IS_FRAMESHIFT`), and direct `1-bp` (`TGATG`/`TAATG`) and `4-bp` (`ATGA`) stop-start operon overlaps (`E → M1` / `Er → M1r`) in a single Numba-compiled Viterbi pass.
- **Sliding 4-kb Prefix-Sum Local GC Adaptation**: Rather than relying on a single chromosome-wide hexamer model, [`OrfPredictor`][serogen.genes.orf.predictor.OrfPredictor] computes a prefix-sum 4-kb local GC bin (`0..43`) at every nucleotide so codon emission log-odds adapt seamlessly across AT-rich prophage islands and polysaccharide synthesis operons.
- **Prophage Recovery With Zero Two-Pass Overhead**: Enabling `OrfPredictor(include_phage=True)` boosts prophage 3'-stop recall to **95.6%** (`+7.4%` over bacterial-only baseline, rescuing `+17` prophage ORFs and `+25` exact 5'+3' boundaries) while simultaneously improving host chromosome recall (`98.7%` → `99.1%`) at **>60 Mb/s** (`~71 ms` for the 4.66 Mb chromosome).

## Run Locally

```bash
# Present as interactive slideshow
uv run --with marimo --with altair marimo run scripts/benchmark_prophage_orfs.py

# Edit notebook cells
uv run --with marimo --with altair marimo edit scripts/benchmark_prophage_orfs.py
```
