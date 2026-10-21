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

# Biological minimum amino-acid lengths for intact (non-partial) ORFs per GeneRole
_ROLE_MIN_AA_LEN: dict[GeneRole, int] = {
    GeneRole.FLIPPASE: 240,
    GeneRole.POLYMERASE: 240,
    GeneRole.COPOLYMERASE: 185,
    GeneRole.KINASE: 185,
    GeneRole.PHOSPHATASE: 110,
    GeneRole.ABC_PERMEASE: 185,
    GeneRole.ABC_ATPASE: 185,
    GeneRole.LIGASE: 250,
    GeneRole.TRANSLOCON: 250,
    GeneRole.CAPSULE_ANCHOR: 280,
    GeneRole.PYRUVYLTRANSFERASE: 180,
    GeneRole.METHYLTRANSFERASE: 130,
    GeneRole.ACETYLTRANSFERASE: 105,
    GeneRole.PHOSPHOTRANSFERASE: 150,
    GeneRole.FORMYLTRANSFERASE: 160,
    GeneRole.ALANINE_TRANSFERASE: 180,
    GeneRole.AMINOTRANSFERASE: 240,
    GeneRole.ALDOLASE: 180,
    GeneRole.DEACETYLASE: 150,
    GeneRole.EPIMERASE: 180,
    GeneRole.OXIDOREDUCTASE: 180,
    GeneRole.TRANSMEMBRANE_SYNTHASE: 310,
    GeneRole.NUCLEOTIDYLTRANSFERASE: 160,
    GeneRole.NUCLEOTIDASE: 130,
    GeneRole.INITIATOR: 175,
    GeneRole.GLYCOSYLHYDROLASE: 180,
    GeneRole.GLYCOSYLTRANSFERASE: 150,
}


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

        bp_lengths = genes.intervals.lengths
        aa_lengths = np.array(
            [
                len(genes.translations[i].seq)
                if len(genes.translations) == n_genes and len(genes.translations[i].seq) > 0
                else max(0, int(bp_lengths[i]) // 3 - 1)
                for i in range(n_genes)
            ],
            dtype=np.int32,
        )
        partials = genes.is_partial if len(genes.is_partial) == n_genes else np.zeros(n_genes, dtype=np.bool_)

        full_probs = np.full((n_genes, n_roles), 1e-12, dtype=np.float64)
        if int(GeneRole.UNKNOWN) in classes:
            full_probs[:, classes] = raw_probs
        else:
            logits = self.role_classifier.decision_function(genes).astype(np.float64).copy()
            raw_logits = logits.copy()
            raw_top_cols = np.argmax(raw_logits, axis=1)
            t_arch = self.archetype_classifier.tensors
            mach_set = set(int(r) for r in t_arch.machinery_roles)

            # 1. Enforce per-role biological minimum AA lengths on intact (non-partial) ORFs
            valid_role_mask = np.ones_like(logits, dtype=np.bool_)
            for col_idx, cls_val in enumerate(classes):
                role_enum = GeneRole(int(cls_val))
                min_aa = _ROLE_MIN_AA_LEN.get(role_enum, 70)
                short_col = (~partials) & (aa_lengths > 0) & (aa_lengths < min_aa)
                if np.any(short_col):
                    valid_role_mask[short_col, col_idx] = False
                    logits[short_col, col_idx] = -30.0

            # 2. Co-directional operon context: suppress spurious MGE calls & disambiguate weak cargo
            starts = genes.intervals.starts
            ends = genes.intervals.ends
            strands = genes.intervals.strands
            contigs = (
                genes.contig_indices if len(genes.contig_indices) == n_genes else np.zeros(n_genes, dtype=np.uint32)
            )
            in_tight_operon = np.zeros(n_genes, dtype=np.bool_)
            for i in range(n_genes):
                same_left = (
                    i > 0
                    and contigs[i] == contigs[i - 1]
                    and strands[i] == strands[i - 1]
                    and (int(starts[i]) - int(ends[i - 1])) <= 350
                )
                same_right = (
                    i + 1 < n_genes
                    and contigs[i + 1] == contigs[i]
                    and strands[i + 1] == strands[i]
                    and (int(starts[i + 1]) - int(ends[i])) <= 350
                )
                if same_left and same_right:
                    in_tight_operon[i] = True
                elif (same_left or same_right) and i > 0 and i + 1 < n_genes and strands[i - 1] == strands[i + 1]:
                    in_tight_operon[i] = True

            mge_cols = [
                col_idx
                for col_idx, c in enumerate(classes)
                if int(c) in (int(GeneRole.TRANSPOSASE), int(GeneRole.INTEGRASE))
            ]
            gt_cols = [col_idx for col_idx, c in enumerate(classes) if int(c) == int(GeneRole.GLYCOSYLTRANSFERASE)]
            tms_cols = [col_idx for col_idx, c in enumerate(classes) if int(c) == int(GeneRole.TRANSMEMBRANE_SYNTHASE)]
            wzx_cols = [col_idx for col_idx, c in enumerate(classes) if int(c) == int(GeneRole.FLIPPASE)]
            wzy_cols = [col_idx for col_idx, c in enumerate(classes) if int(c) == int(GeneRole.POLYMERASE)]
            at_cols = [col_idx for col_idx, c in enumerate(classes) if int(c) == int(GeneRole.ACETYLTRANSFERASE)]
            wzm_cols = [col_idx for col_idx, c in enumerate(classes) if int(c) == int(GeneRole.ABC_PERMEASE)]
            wzt_cols = [col_idx for col_idx, c in enumerate(classes) if int(c) == int(GeneRole.ABC_ATPASE)]
            wzb_cols = [col_idx for col_idx, c in enumerate(classes) if int(c) == int(GeneRole.PHOSPHATASE)]
            waal_cols = [col_idx for col_idx, c in enumerate(classes) if int(c) == int(GeneRole.LIGASE)]
            remnant_roles = {
                int(GeneRole.FLIPPASE),
                int(GeneRole.POLYMERASE),
                int(GeneRole.LIGASE),
                int(GeneRole.TRANSLOCON),
                int(GeneRole.CAPSULE_ANCHOR),
            }

            for i in range(n_genes):
                rtc = int(raw_top_cols[i])
                rt_role = int(classes[rtc])
                if (
                    not partials[i]
                    and aa_lengths[i] < 215
                    and rt_role in remnant_roles
                    and not valid_role_mask[i, rtc]
                    and raw_logits[i, rtc] >= 4.0
                ):
                    valid_role_mask[i, :] = False
                    logits[i, :] = -10.0
                    continue

                if in_tight_operon[i] and aa_lengths[i] >= 180:
                    for mc in mge_cols:
                        if logits[i, mc] < 2.5:
                            logits[i, mc] -= 3.5
                    if gt_cols and 215 <= aa_lengths[i] <= 660:
                        gc_col = gt_cols[0]
                        max_cur = float(np.max(logits[i]))
                        if max_cur < 2.2 and logits[i, gc_col] > -2.5:
                            logits[i, gc_col] += 1.6
                    if gt_cols and tms_cols:
                        gc_col = gt_cols[0]
                        tms_col = tms_cols[0]
                        if logits[i, gc_col] > logits[i, tms_col] and logits[i, gc_col] >= 1.0:
                            logits[i, tms_col] = -10.0
                            valid_role_mask[i, tms_col] = False

            # Orphan Wzm (ABC_PERMEASE) without adjacent Wzt (ABC_ATPASE) vs cpsACP (PHOSPHATASE) disambiguation
            if wzm_cols and wzb_cols:
                wzm_c = wzm_cols[0]
                wzb_c = wzb_cols[0]
                wzt_c = wzt_cols[0] if wzt_cols else -1
                for i in range(n_genes):
                    if (
                        valid_role_mask[i, wzm_c]
                        and aa_lengths[i] < 225
                        and logits[i, wzm_c] >= logits[i, wzb_c]
                        and logits[i, wzb_c] >= 2.0
                    ):
                        has_wzt = False
                        if wzt_c >= 0:
                            for j in range(n_genes):
                                if (
                                    i != j
                                    and contigs[i] == contigs[j]
                                    and abs(int(starts[i]) - int(starts[j])) <= 5_000
                                    and valid_role_mask[j, wzt_c]
                                    and logits[j, wzt_c] >= 3.5
                                ):
                                    has_wzt = True
                                    break
                        if not has_wzt:
                            logits[i, wzm_c] = -10.0
                            valid_role_mask[i, wzm_c] = False
                            for lc in waal_cols:
                                logits[i, lc] = -10.0
                                valid_role_mask[i, lc] = False

            # Single-copy Wzx / Wzy disambiguation within a tight same-strand operon (< 15 kb)
            at_c = at_cols[0] if at_cols else -1
            for single_cols in (wzx_cols, wzy_cols):
                if not single_cols:
                    continue
                sc = single_cols[0]
                top_cls_cur = np.argmax(np.where(valid_role_mask, logits, -1e9), axis=1)
                cand_idxs = [
                    idx
                    for idx in range(n_genes)
                    if top_cls_cur[idx] == sc and valid_role_mask[idx, sc] and logits[idx, sc] >= 1.0
                ]
                if len(cand_idxs) > 1:
                    for idx_a in cand_idxs:
                        for idx_b in cand_idxs:
                            if (
                                idx_a != idx_b
                                and contigs[idx_a] == contigs[idx_b]
                                and strands[idx_a] == strands[idx_b]
                                and abs(int(starts[idx_a]) - int(starts[idx_b])) <= 15_000
                                and logits[idx_a, sc] > logits[idx_b, sc]
                            ):
                                is_acyl3 = (
                                    at_c >= 0
                                    and valid_role_mask[idx_b, at_c]
                                    and logits[idx_b, at_c] >= 2.5
                                    and (at_c < 0 or logits[idx_b, at_c] > logits[idx_a, at_c] + 1.5)
                                )
                                is_weak_dup = (logits[idx_a, sc] - logits[idx_b, sc] >= 2.5) and (
                                    logits[idx_b, sc] < 4.5
                                )
                                if is_acyl3 or is_weak_dup:
                                    logits[idx_b, sc] = -10.0
                                    valid_role_mask[idx_b, sc] = False
                                    if is_acyl3:
                                        for oc in wzx_cols + wzy_cols:
                                            logits[idx_b, oc] = -10.0
                                            valid_role_mask[idx_b, oc] = False

            # Recompute calibrated softmax probabilities from context-adjusted logits
            shift_logits = logits - np.max(logits, axis=1, keepdims=True)
            exp_logits = np.where(valid_role_mask, np.exp(np.clip(shift_logits, -60.0, 0.0)), 0.0)
            exp_sum = np.maximum(exp_logits.sum(axis=1, keepdims=True), 1e-12)
            adj_probs = exp_logits / exp_sum

            # Machinery roles with weak/negative raw logits (< 1.5) are never genuine transport/assembly proteins
            mach_col_mask = np.array([int(c) in mach_set for c in classes], dtype=np.bool_)
            neg_mach_mask = ((logits < 1.5) & mach_col_mask[None, :]) | (~valid_role_mask)
            cleaned_probs = np.where(neg_mach_mask, 1e-12, adj_probs)

            valid_logits = np.where(~neg_mach_mask, logits, -30.0)
            max_l = np.max(valid_logits, axis=1, keepdims=True)
            max_p = np.max(cleaned_probs, axis=1, keepdims=True)

            base_conf = 1.0 - np.clip(
                (1.0 - max_p) ** (1.0 / max(self.unknown_temperature, 1e-3)),
                1e-6,
                0.99,
            )
            logit_factor = np.where(
                max_l < 0.35,
                0.15 / (1.0 + np.exp(-np.clip((max_l - 0.0) / 0.5, -30.0, 30.0))),
                0.75 + 0.25 / (1.0 + np.exp(-np.clip((max_l - 4.5) / 1.0, -30.0, 30.0))),
            )
            active_mass = np.clip(base_conf * logit_factor, 1e-4, 0.999999)

            row_sum = np.maximum(cleaned_probs.sum(axis=1, keepdims=True), 1e-12)
            norm_cleaned = np.where(max_p > 1e-6, cleaned_probs / row_sum, cleaned_probs)
            full_probs[:, classes] = norm_cleaned * active_mass
            full_probs[:, int(GeneRole.UNKNOWN)] = np.clip(1.0 - full_probs[:, classes].sum(axis=1), 1e-12, 1.0)

        # Enforce biological minimum length constraints for machinery (< 90 aa) and biosynthetic (< 70 aa) ORFs
        if len(bp_lengths) == n_genes and np.any(bp_lengths > 0):
            for role_enum, min_aa in _ROLE_MIN_AA_LEN.items():
                short_r_mask = (~partials) & (aa_lengths > 0) & (aa_lengths < min_aa)
                if np.any(short_r_mask):
                    r_idx = int(role_enum)
                    shifted = full_probs[short_r_mask, r_idx].copy()
                    full_probs[short_r_mask, r_idx] = 1e-12
                    full_probs[short_r_mask, int(GeneRole.UNKNOWN)] += shifted

            tiny_orf_mask = (~partials) & (bp_lengths > 0) & (bp_lengths < 210)
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
