r"""SeroGen: High-performance genomic analysis and *in silico* serotyping engine.

Provides Structure-of-Arrays (SoA) sequence and graph containers ([`serogen.core`][serogen.core]),
single-pass bacterial and prophage ORF prediction plus multi-channel protein role classification
([`serogen.genes`][serogen.genes]), and exact dynamic-programming locus segmentation and bipartite
synteny matching ([`serogen.loci`][serogen.loci]).
"""

from serogen import core, genes, loci
from serogen.core import FastaReader, GenomeAssembly, Interval, Intervals, Strand
from serogen.genes import Gene, GeneRole, GeneRoleClassifier, GeneRoles, Genes, OrfPredictor
from serogen.loci import (
    ArchetypeClassifier,
    Loci,
    Locus,
    LocusArchetype,
    LocusMatch,
    LocusMatcher,
    LocusMatches,
    LocusPredictor,
    LocusSegmenter,
    Match,
    Matches,
)

__all__ = [
    "ArchetypeClassifier",
    "FastaReader",
    "Gene",
    "GeneRole",
    "GeneRoleClassifier",
    "GeneRoles",
    "Genes",
    "GenomeAssembly",
    "Interval",
    "Intervals",
    "Loci",
    "Locus",
    "LocusArchetype",
    "LocusMatch",
    "LocusMatcher",
    "LocusMatches",
    "LocusPredictor",
    "LocusSegmenter",
    "Match",
    "Matches",
    "OrfPredictor",
    "Strand",
    "core",
    "genes",
    "loci",
]
