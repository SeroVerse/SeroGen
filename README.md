<div align="center">

# 🧬 SeroGen

**De novo bacterial polysaccharide antigen locus discovery, twilight-zone CNN gene role classification, hybrid MCL orthogroup clustering, and graph-based pan-locus analysis.**

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

**SeroGen** is the core genomic engine of the **SeroVerse** suite. Built from the ground up on zero-copy Structure-of-Arrays (SoA) containers (`numpy`) and SIMD-parallel JIT kernels (`numba`), SeroGen transforms raw bacterial or prophage genome assemblies into annotated polysaccharide synthesis loci, orthogroup pan-graphs, and de novo reference databases in tens of milliseconds — **with zero external CLI dependencies**.

### Core Capabilities

- ⚡ **Single-Pass 29-State Bacterial & Prophage ORF Calling (`OrfPredictor`)**: Decodes a 4.66 Mb *E. coli* chromosome in **`~71 ms` (>65 Mb/s)** via a 29-state frameshift- and operon-overlap-aware (`TGATG`/`ATGA`) profile-HMM Viterbi decoder with sliding 4-kb local GC adaptation, Shine-Dalgarno RBS scoring, and single-pass viral/prophage gap rescue (`include_phage=True`).
- 🧠 **Polysaccharide-Specific 1D-CNN Role Classification (`GeneRoleClassifier`)**: Beats remote homology detection (`BLAST`/`MMseqs2`) in the `< 25%` sequence identity twilight zone by combining multi-scale 1D convolutional motif filters (`kernels=(3, 5, 9)`) with a 20-D biophysical composition/hydropathy/charge head across **29 functional `GeneRole`s** (+ open-set `UNKNOWN`).
- 🔬 **Smart Hybrid Clustering & CNN-Rescued MCL (`Clusterer`)**: Combines `FNV-1a` exact deduplication, `RandstrobeIndex` strobemer seeding, `MMSEQS12` reduced-alphabet $12^4$ CSR + ungapped BLOSUM62 diagonal prefiltering, SIMD-parallel Smith-Waterman-Gotoh (`PairwiseAligner`), **role-gated CNN active-site motif rescue** (`conv_max`), and sparse Markov Clustering (`mode="mcl"`) or greedy `CD-HIT` (`mode="cdhit"`).
- 🕸️ **Zero-Copy Graph Data Structures & Solvers (`serogen.core.graph`)**: Bit-packed 32-bit `IntNode`s (`GeneNode`, `IntergenicEdge`), `GraphBatch` (`Loci`) containers with `.as_pan_graph()` consensus compression, Weisfeiler-Lehman + Hungarian `GraphAligner`, Chu-Liu/Edmonds + constrained $K$-best `maximum_spanning_arborescence`, and Numba `RelationalMPNN`.
- 🧩 **De Novo Fragmented & MGE-Tolerant Locus Discovery (`LocusPredictor` & `LocusMatcher`)**: Segments `CPS`, `O_AG`, `WTA`, `LTA`, `EPS`, `LOS_CORE`, and `PGA` loci *de novo* using All-Contiguous-Partitions Dynamic Programming (`LocusSegmenter`), bridges/heals `IS` transposase/integrase insertions (`heal_mobile_elements=True`), stitches fragmented loci split across draft contigs (`stitch_contig_splits=True`), and builds de novo searchable databases (`LocusMatcher`).

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
from serogen.core.genome import GenomeAssembly
from serogen.loci.matcher import LocusMatcher
from serogen.loci.models import GeneNode, Loci
from serogen.loci.predictor import LocusPredictor

# 1. Load a genome assembly and discover polysaccharide loci de novo
assembly = GenomeAssembly.from_fasta("genome.fasta")
predictor = LocusPredictor(stitch_contig_splits=True, heal_mobile_elements=False)
loci = predictor.predict(assembly)

for locus in loci:
    print(
        f"{locus.archetype.name}: {locus.start}..{locus.end} "
        f"({locus.num_genes} genes, completeness={locus.completeness:.2f})"
    )

# 2. Build a de novo MCL-clustered locus database and consensus pan-locus graph
db_loci = Loci.from_list(reference_loci).canonicalize_strands()
matcher = LocusMatcher(db_loci, cluster_mode="mcl", min_identity=80.0, min_coverage=0.80)
pan_adj, pan_nodes, pan_counts = matcher.targets.as_pan_graph(node_mask=GeneNode.mask("ROLE_ID", "CLUSTER_ID"))

# 3. Match newly discovered loci and inspect structural variant bitflags
matches = matcher.match(loci)
best = matches[0]
print(f"Matched locus: {best.target_name} (score={best.score:.3f}, flags={best.match_type!r})")
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
