"""End-to-end biosynthetic locus prediction pipeline (`LocusPredictor` / `LocusFinder`)."""

from collections.abc import Iterable

import numpy as np

from serogen.core.genome import GenomeAssembly
from serogen.core.seq.seq import Sequence, Sequences
from serogen.genes.cnn import GeneRoleClassifier
from serogen.genes.models import Genes
from serogen.genes.orf.predictor import OrfPredictor
from serogen.genes.roles import GeneRole
from serogen.loci.archetypes import ArchetypeCategory, LocusArchetype
from serogen.loci.classifier import ArchetypeClassifier
from serogen.loci.models import Loci
from serogen.loci.segmenter import LocusSegmenter


class LocusPredictor:
    """High-level biosynthetic gene cluster (BGC) discovery and segmentation pipeline.

    Orchestrates:
    1. Open Reading Frame (ORF) prediction via [`OrfPredictor`][serogen.genes.orf.predictor.OrfPredictor],
    2. Protein sequence functional role classification via [`GeneRoleClassifier`][serogen.genes.cnn.GeneRoleClassifier], and
    3. Archetype-constrained chromosomal segmentation via [`LocusSegmenter`][serogen.loci.segmenter.LocusSegmenter].
    """

    def __init__(
        self,
        orf_predictor: OrfPredictor | None = None,
        role_classifier: GeneRoleClassifier | None = None,
        segmenter: LocusSegmenter | None = None,
        *,
        min_completeness: float = 0.80,
        max_incompatibility: float = 0.25,
        unknown_temperature: float = 0.35,
        stitch_contig_splits: bool = True,
    ) -> None:
        self.orf_predictor = orf_predictor if orf_predictor is not None else OrfPredictor()
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

        bp_lengths = genes.intervals.lengths
        aa_lengths = (
            genes.translations.lengths if len(genes.translations) == n_genes else np.maximum(0, bp_lengths // 3 - 1)
        )
        partials = genes.is_partial if len(genes.is_partial) == n_genes else np.zeros(n_genes, dtype=np.bool_)

        # Suppress unpaired ABC_PERMEASE calls when no ABC_ATPASE partner is present within +-15 genes on the same contig
        unpaired_perm = np.zeros(n_genes, dtype=np.bool_)
        perm_col = -1
        if int(GeneRole.ABC_PERMEASE) in classes and int(GeneRole.ABC_ATPASE) in classes:
            perm_col = int(np.flatnonzero(classes == int(GeneRole.ABC_PERMEASE))[0])
            atp_col = int(np.flatnonzero(classes == int(GeneRole.ABC_ATPASE))[0])
            cids = genes.contig_indices if len(genes.contig_indices) == n_genes else np.zeros(n_genes, dtype=np.uint32)
            atp_hits = raw_probs[:, atp_col] >= 0.15
            for idx in np.flatnonzero(raw_probs[:, perm_col] >= 0.15):
                i = int(idx)
                cid_i = cids[i]
                lo = max(0, i - 15)
                hi = min(n_genes, i + 16)
                while lo < i and cids[lo] != cid_i:
                    lo += 1
                while hi > i + 1 and cids[hi - 1] != cid_i:
                    hi -= 1
                near_truncated_edge = (
                    i == lo
                    or i == hi - 1
                    or (i == lo + 1 and bool(partials[lo]) and int(aa_lengths[lo]) < 200)
                    or (i == hi - 2 and bool(partials[hi - 1]) and int(aa_lengths[hi - 1]) < 200)
                )
                if not np.any(atp_hits[lo:hi]) and not near_truncated_edge:
                    unpaired_perm[i] = True

        full_probs = np.full((n_genes, n_roles), 1e-12, dtype=np.float64)
        if int(GeneRole.UNKNOWN) in classes:
            full_probs[:, classes] = raw_probs
            if perm_col >= 0 and np.any(unpaired_perm):
                shifted_perm = full_probs[unpaired_perm, int(GeneRole.ABC_PERMEASE)].copy()
                full_probs[unpaired_perm, int(GeneRole.ABC_PERMEASE)] = 1e-12
                full_probs[unpaired_perm, int(GeneRole.UNKNOWN)] += shifted_perm
        else:
            logits = self.role_classifier.decision_function(genes).astype(np.float64)
            if perm_col >= 0 and np.any(unpaired_perm):
                raw_probs = raw_probs.copy()
                raw_probs[unpaired_perm, perm_col] = 1e-12
                logits[unpaired_perm, perm_col] = -10.0

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
        heal_mobile_elements: bool = False,
    ) -> Loci:
        """Discover, classify, and segment biosynthetic [`Loci`][serogen.loci.models.Loci] from genomic sequences or `Genes`."""
        if isinstance(target, Genes):
            if (
                len(target) > 0
                and len(target.intervals.lengths) == len(target)
                and np.any(target.intervals.lengths > 0)
            ):
                partials = (
                    target.is_partial
                    if len(target.is_partial) == len(target)
                    else np.zeros(len(target), dtype=np.bool_)
                )
                keep_mask = partials | (target.intervals.lengths >= self.orf_predictor.min_gene_len)
                genes = target[keep_mask] if not bool(np.all(keep_mask)) else target
            else:
                genes = target
        else:
            genes = self.orf_predictor.predict(target)
        if len(genes) == 0:
            return Loci.empty()

        role_probs = self.predict_role_probabilities(genes)
        is_phage = self.role_classifier.predict_source(genes) == 1
        loci = self.segmenter.segment(genes, role_probs, is_phage=is_phage)
        if heal_mobile_elements and len(loci) > 0:
            loci = loci.heal_mobile_elements(remove_nodes=True)
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
