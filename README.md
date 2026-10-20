<div align="center">

# 🧬 SeroGen

**High-performance genomic analysis, de novo polysaccharide locus segmentation, and *in silico* serotyping.**

[![CI](https://img.shields.io/github/actions/workflow/status/SeroVerse/serogen/ci.yml?branch=main&label=CI&logo=github)](https://github.com/SeroVerse/serogen/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/SeroVerse/serogen/docs.yml?branch=main&label=Docs&logo=readthedocs)](https://SeroVerse.github.io/serogen/)
[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-0284c7?logo=python&logoColor=white)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-6366f1.svg)](https://www.gnu.org/licenses/gpl-3.0)

[**Documentation**](https://SeroVerse.github.io/serogen/) · [**Architecture**](https://SeroVerse.github.io/serogen/architecture/) · [**API Reference**](https://SeroVerse.github.io/serogen/reference/) · [**Issues**](https://github.com/SeroVerse/serogen/issues)

</div>

---

## Overview

**SeroGen** is the core genomic engine of the **SeroVerse** suite. Built from the ground up on zero-copy Structure-of-Arrays (SoA) containers (`numpy`) and SIMD-parallel JIT kernels (`numba`), SeroGen transforms raw bacterial or prophage genome assemblies into annotated polysaccharide synthesis loci and high-precision serotype calls in tens of milliseconds — **with zero external CLI dependencies**.

### Key Highlights

- ⚡ **Single-Pass Bacterial & Prophage ORF Calling (`OrfPredictor`)**: Decodes a 4.66 Mb *E. coli* chromosome in **`~70 ms` (>65 Mb/s)** via a 21-state frameshift-aware profile-HMM Viterbi decoder with Shine-Dalgarno RBS scoring and single-pass viral/prophage gap rescue (`GeneticCode.bacterial` + `GeneticCode.viral`).
- 🧠 **Multi-Scale 1D CNN Role Classification (`GeneRoleClassifier`)**: Classifies translated proteins across **36 biosynthetic, regulatory, transport, and MGE roles** and extracts multi-scale convolutional embeddings in pure NumPy/Numba.
- 🧩 **Exact ACP-DP Locus Segmentation (`LocusSegmenter`)**: Partitions chromosomes into capsule (`CPS`), O-antigen (`O_AG`), wall teichoic acid (`WTA`), and ectopic synthase islands using All-Contiguous-Partitions Dynamic Programming and cross-contig boundary stitching.
- 🔍 **Randstrobe + Banded Gotoh + Bipartite Graph Matching (`LocusMatcher`)**: Combines `RandstrobeIndex` k-mer seeding, parallel banded Smith-Waterman-Gotoh (`PairwiseAligner`), MCL/Leiden orthogroup clustering (`Clusterer`), and Hungarian bipartite synteny alignment (`GraphAligner`).

---

## Installation

```bash
uv add serogen
# or with pip:
pip install serogen
```

---

## Quickstart

```python
from serogen import GenomeAssembly, LocusDatabase, LocusMatcher, LocusPredictor

# 1. Load a genome assembly and predict biosynthetic loci de novo
assembly = GenomeAssembly.from_fasta("genome.fasta")
predictor = LocusPredictor()
loci = predictor.predict(assembly)

for locus in loci:
    print(f"{locus.archetype.name}: {locus.start}..{locus.end} ({len(locus.genes)} genes)")

# 2. Match predicted loci against a compiled SeroGen reference database
db = LocusDatabase.load("spneumo_cps.npz")
matcher = LocusMatcher.from_database(db)
matches = matcher.match(loci)

best = matches[0]
print(f"Serotype: {best.serotype} (score={best.score:.3f}, AA identity={best.mean_aa_identity:.1%})")
```

---

## Development

```bash
# Install all development, test, lint, typing, and docs groups
uv sync --all-groups

# Run linter and static type checker
uv run ruff check .
uv run ty check src/

# Run test suite
uv run pytest

# Build and serve documentation locally with Zensical
uv run zensical serve
```

---

## License

Released under the [GNU General Public License v3.0 (GPLv3)](LICENSE).
