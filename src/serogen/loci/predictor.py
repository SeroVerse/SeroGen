"""End-to-end biosynthetic locus prediction pipeline (`LocusPredictor` / `LocusFinder`)."""

from collections.abc import Iterable

import numpy as np

from serogen.core.genome import GenomeAssembly
from serogen.core.seq.seq import Sequence, Sequences
from serogen.genes.classifier import GeneRoleClassifier
from serogen.genes.models import Genes
from serogen.genes.predictor import OrfPredictor
from serogen.genes.roles import GeneRole
from serogen.loci.archetypes import ArchetypeCategory, LocusArchetype
from serogen.loci.classifier import ArchetypeClassifier
from serogen.loci.models import Loci
from serogen.loci.segmenter import LocusSegmenter


class LocusPredictor:
    """High-level biosynthetic gene cluster (BGC) discovery and segmentation pipeline.

    Orchestrates:
    1. Open Reading Frame (ORF) prediction via [`OrfPredictor`][serogen.genes.predictor.OrfPredictor],
    2. Protein sequence functional role classification via [`GeneRoleClassifier`][serogen.genes.classifier.GeneRoleClassifier], and
    3. Archetype-constrained chromosomal segmentation via [`LocusSegmenter`][serogen.loci.segmenter.LocusSegmenter].
    """

    def __init__(
        self,
        orf_predictor: OrfPredictor | None = None,
        role_classifier: GeneRoleClassifier | None = None,
        segmenter: LocusSegmenter | None = None,
        *,
        whole_genome: bool = False,
        min_completeness: float = 0.80,
        max_incompatibility: float = 0.25,
        unknown_temperature: float = 0.35,
        stitch_contig_splits: bool = True,
    ) -> None:
        self.orf_predictor = orf_predictor if orf_predictor is not None else OrfPredictor(whole_genome=whole_genome)
        self.role_classifier = role_classifier if role_classifier is not None else GeneRoleClassifier.pretrained()
        self.segmenter = (
            segmenter
            if segmenter is not None
            else LocusSegmenter(
                min_completeness=min_completeness,
                max_incompatibility=max_incompatibility,
                stitch_contig_splits=stitch_contig_splits,
            )
        )
        self.unknown_temperature = unknown_temperature

    @property
    def archetype_classifier(self) -> ArchetypeClassifier:
        """Return the underlying `ArchetypeClassifier` used by `self.segmenter`."""
        return self.segmenter.classifier

    def predict_role_probabilities(self, genes: Genes) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Predict full `(N, 30)` `GeneRole` probability distributions (including `UNKNOWN`) for `genes`."""
        n_genes = len(genes)
        n_roles = len(GeneRole)
        if n_genes == 0:
            return np.empty((0, n_roles), dtype=np.float64)

        raw_probs = self.role_classifier.predict_proba(genes).astype(np.float64)
        classes = self.role_classifier.classes_

        full_probs = np.full((n_genes, n_roles), 1e-12, dtype=np.float64)
        if int(GeneRole.UNKNOWN) in classes:
            full_probs[:, classes] = raw_probs
        else:
            logits = self.role_classifier.decision_function(genes).astype(np.float64)
            # Machinery roles with weak/negative raw logits (< 1.0) are never genuine transport/assembly proteins
            mach_set = set(int(r) for r in self.archetype_classifier.tensors.machinery_roles)
            mach_col_mask = np.array([int(c) in mach_set for c in classes], dtype=np.bool_)
            neg_mach_mask = (logits < 1.0) & mach_col_mask[None, :]
            cleaned_probs = np.where(neg_mach_mask, 1e-12, raw_probs)

            max_l = np.max(logits, axis=1, keepdims=True)
            max_p = np.max(cleaned_probs, axis=1, keepdims=True)

            base_conf = 1.0 - np.clip(
                (1.0 - max_p) ** (1.0 / max(self.unknown_temperature, 1e-3)),
                1e-6,
                0.99,
            )
            logit_factor = 0.75 + 0.25 / (1.0 + np.exp(-np.clip((max_l - 4.5) / 1.0, -30.0, 30.0)))
            active_mass = np.clip(base_conf * logit_factor, 1e-4, 0.999999)

            row_sum = np.maximum(cleaned_probs.sum(axis=1, keepdims=True), 1e-12)
            norm_cleaned = np.where(max_p > 1e-6, cleaned_probs / row_sum, cleaned_probs)
            full_probs[:, classes] = norm_cleaned * active_mass
            full_probs[:, int(GeneRole.UNKNOWN)] = np.clip(1.0 - full_probs[:, classes].sum(axis=1), 1e-12, 1.0)

        # Enforce biological minimum length constraints for machinery (< 90 aa) and biosynthetic (< 70 aa) ORFs
        bp_lengths = genes.intervals.lengths
        if len(bp_lengths) == n_genes and np.any(bp_lengths > 0):
            short_mach_mask = (bp_lengths > 0) & (bp_lengths < 270)
            if np.any(short_mach_mask):
                mach_roles = self.archetype_classifier.tensors.machinery_roles
                shifted = full_probs[np.ix_(short_mach_mask, mach_roles)].sum(axis=1)
                full_probs[np.ix_(short_mach_mask, mach_roles)] = 1e-12
                full_probs[short_mach_mask, int(GeneRole.UNKNOWN)] += shifted

            tiny_orf_mask = (bp_lengths > 0) & (bp_lengths < 210)
            if np.any(tiny_orf_mask):
                full_probs[tiny_orf_mask, :] = 1e-12
                full_probs[tiny_orf_mask, int(GeneRole.UNKNOWN)] = 0.99

        full_probs /= full_probs.sum(axis=1, keepdims=True)
        return full_probs

    def predict(
        self,
        target: bytes | str | Sequence | Sequences | GenomeAssembly | Genes,
        *,
        categories: ArchetypeCategory | str | Iterable[ArchetypeCategory | str] | None = None,
        archetypes: LocusArchetype | str | int | Iterable[LocusArchetype | str | int] | None = None,
        min_completeness: float = 0.0,
        min_score: float = 0.0,
        max_frameshift_fraction: float = 1.0,
        only_intact: bool = False,
    ) -> Loci:
        """Discover, classify, and segment biosynthetic [`Loci`][serogen.loci.models.Loci] from genomic sequences or `Genes`."""
        genes = target if isinstance(target, Genes) else self.orf_predictor.predict(target)
        if len(genes) == 0:
            return Loci.empty()

        role_probs = self.predict_role_probabilities(genes)
        loci = self.segmenter.segment(genes, role_probs)
        if (
            categories is not None
            or archetypes is not None
            or min_completeness > 0.0
            or min_score > 0.0
            or max_frameshift_fraction < 1.0
            or only_intact
        ):
            return loci.filter(
                categories=categories,
                archetypes=archetypes,
                min_completeness=min_completeness,
                min_score=min_score,
                max_frameshift_fraction=max_frameshift_fraction,
                only_intact=only_intact,
            )
        return loci


LocusFinder = LocusPredictor
