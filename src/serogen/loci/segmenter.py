"""Archetype-Constrained Exact Partitioning Dynamic Programming (ACP-DP) locus segmenter."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from serogen.core.interval import Interval, Strand
from serogen.genes.models import Genes
from serogen.genes.roles import GeneRole
from serogen.loci.archetypes import LocusArchetype
from serogen.loci.classifier import ArchetypeClassifier
from serogen.loci.models import GeneNode, IntergenicEdge, IntergenicType, Loci, Locus


@dataclass(slots=True)
class _BoundaryIsland:
    """Contig-boundary candidate island (or stitched chain of boundary islands) in canonical 5' -> 3' frame."""

    contig_ids: set[int]
    pieces: tuple[tuple[int, int, np.ndarray[tuple[int], np.dtype[np.intp]]], ...]
    canon_g_indices: np.ndarray[tuple[int], np.dtype[np.intp]]
    open_5prime: bool
    open_3prime: bool
    v_probs: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    v_starts: np.ndarray[tuple[int], np.dtype[np.int32]]
    v_ends: np.ndarray[tuple[int], np.dtype[np.int32]]
    v_strands: np.ndarray[tuple[int], np.dtype[np.int8]]
    dp_segments: list[tuple[int, int, LocusArchetype, float, float]]
    total_score: float


class LocusSegmenter:
    """Archetype-Constrained Exact Partitioning Dynamic Programming (ACP-DP) locus segmenter.

    Segments chromosomal [`Genes`][serogen.genes.models.Genes] and their `GeneRole` probability
    distributions (or discrete role labels) into non-overlapping [`Loci`][serogen.loci.models.Loci],
    disambiguating tandem loci (e.g., adjacent K-locus and O-locus in *Klebsiella* / *E. coli*),
    bridging internal unannotated (`UNKNOWN`) and mobile element (`TRANSPOSASE`/`INTEGRASE`) gaps,
    and stitching loci split across contig boundaries in fragmented assemblies.
    """

    def __init__(
        self,
        classifier: ArchetypeClassifier | None = None,
        *,
        min_completeness: float = 0.80,
        max_incompatibility: float = 0.25,
        max_gap_genes: int = 3,
        max_gap_bp: int = 5_000,
        min_locus_genes: int = 3,
        max_locus_genes: int = 45,
        locus_open_penalty: float = 6.0,
        strand_switch_penalty: float = 5.5,
        min_mean_llr: float = 1.0,
        min_biosynthetic_prob: float = 0.35,
        min_machinery_prob: float = 0.40,
        max_operon_snap_bp: int = 150,
        allow_single_gene_synthase: bool = True,
        min_synthase_prob: float = 0.70,
        stitch_contig_splits: bool = True,
    ) -> None:
        self.classifier = (
            classifier
            if classifier is not None
            else ArchetypeClassifier(
                min_completeness=min_completeness,
                max_incompatibility=max_incompatibility,
            )
        )
        self.min_completeness = self.classifier.min_completeness
        self.max_incompatibility = self.classifier.max_incompatibility
        self.max_gap_genes = max_gap_genes
        self.max_gap_bp = max_gap_bp
        self.min_locus_genes = min_locus_genes
        self.max_locus_genes = max_locus_genes
        self.locus_open_penalty = locus_open_penalty
        self.strand_switch_penalty = strand_switch_penalty
        self.min_mean_llr = min_mean_llr
        self.min_biosynthetic_prob = min_biosynthetic_prob
        self.min_machinery_prob = min_machinery_prob
        self.max_operon_snap_bp = max_operon_snap_bp
        self.allow_single_gene_synthase = allow_single_gene_synthase
        self.min_synthase_prob = min_synthase_prob
        self.stitch_contig_splits = stitch_contig_splits

    def _find_candidate_islands(
        self,
        starts: np.ndarray[tuple[int], np.dtype[np.int32]],
        ends: np.ndarray[tuple[int], np.dtype[np.int32]],
        probs: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        *,
        allow_boundary_partials: bool = False,
    ) -> list[tuple[int, int]]:
        """Group active biosynthetic genes into coarse candidate islands separated by large gaps."""
        n = len(starts)
        if n == 0:
            return []

        t = self.classifier.tensors
        biosyn_prob = probs[:, t.biosynthetic_mask].sum(axis=1)
        mge_prob = probs[:, int(GeneRole.TRANSPOSASE)] + probs[:, int(GeneRole.INTEGRASE)]
        is_mge = mge_prob >= self.min_biosynthetic_prob
        mach_prob = probs[:, t.machinery_roles].sum(axis=1)
        synth_prob = probs[:, int(GeneRole.TRANSMEMBRANE_SYNTHASE)]

        active_indices = np.nonzero(biosyn_prob >= self.min_biosynthetic_prob)[0]
        if len(active_indices) == 0:
            return []

        contig_min_start = int(starts[0])
        contig_max_end = int(ends[-1])

        def _keep_run(c_start: int, c_end: int, has_mach: bool, has_synth: bool) -> bool:
            if has_mach and (c_end - c_start + 1) >= self.min_locus_genes:
                return True
            if self.allow_single_gene_synthase and has_synth:
                return True
            if allow_boundary_partials:
                touches_start = (c_start <= 1 and int(starts[c_start]) - contig_min_start <= 2500) or (
                    int(starts[c_start]) - contig_min_start <= 1200
                )
                touches_end = (c_end >= n - 2 and contig_max_end - int(ends[c_end]) <= 2500) or (
                    contig_max_end - int(ends[c_end]) <= 1200
                )
                if touches_start or touches_end:
                    return True
            return False

        def _snap_operon_run(c_start: int, c_end: int) -> tuple[int, int]:
            while (
                c_start > 0
                and int(starts[c_start]) - int(ends[c_start - 1]) <= self.max_operon_snap_bp
                and biosyn_prob[c_start - 1] >= self.min_biosynthetic_prob
            ):
                c_start -= 1
            while (
                c_end + 1 < n
                and int(starts[c_end + 1]) - int(ends[c_end]) <= self.max_operon_snap_bp
                and biosyn_prob[c_end + 1] >= self.min_biosynthetic_prob
            ):
                c_end += 1
            return c_start, c_end

        islands: list[tuple[int, int]] = []
        cur_start = int(active_indices[0])
        cur_end = int(active_indices[0])
        has_machinery = bool(mach_prob[cur_start] >= self.min_machinery_prob)
        has_synthase = bool(synth_prob[cur_start] >= self.min_synthase_prob)

        for idx in active_indices[1:]:
            i = int(idx)
            raw_gene_gap = i - cur_end - 1
            mge_in_gap = int(np.sum(is_mge[cur_end + 1 : i])) if raw_gene_gap > 0 else 0
            if mge_in_gap > 0:
                gene_gap = max(0, raw_gene_gap - mge_in_gap - 1)
                mge_idx = np.nonzero(is_mge[cur_end + 1 : i])[0] + (cur_end + 1)
                mge_span = max(0, int(ends[mge_idx[-1]]) - int(starts[mge_idx[0]]))
                bp_gap = max(0, int(starts[i]) - int(ends[cur_end]) - mge_span)
            else:
                gene_gap = raw_gene_gap
                bp_gap = int(starts[i]) - int(ends[cur_end])
            if gene_gap <= self.max_gap_genes and bp_gap <= self.max_gap_bp:
                cur_end = i
                if mach_prob[i] >= self.min_machinery_prob:
                    has_machinery = True
                if synth_prob[i] >= self.min_synthase_prob:
                    has_synthase = True
            else:
                if _keep_run(cur_start, cur_end, has_machinery, has_synthase):
                    s_run, e_run = _snap_operon_run(cur_start, cur_end)
                    islands.append((s_run, e_run + 1))
                cur_start = i
                cur_end = i
                has_machinery = bool(mach_prob[i] >= self.min_machinery_prob)
                has_synthase = bool(synth_prob[i] >= self.min_synthase_prob)

        if _keep_run(cur_start, cur_end, has_machinery, has_synthase):
            s_run, e_run = _snap_operon_run(cur_start, cur_end)
            islands.append((s_run, e_run + 1))

        return islands

    def _trim_boundaries(
        self,
        i_start: int,
        i_end: int,
        archetype: LocusArchetype,
        top_roles: np.ndarray[tuple[int], np.dtype[np.int32]],
        biosyn_prob: np.ndarray[tuple[int], np.dtype[np.float64]],
        strands: np.ndarray[tuple[int], np.dtype[np.int8]],
        prefixes: tuple[np.ndarray, np.ndarray, np.ndarray],
        starts: np.ndarray[tuple[int], np.dtype[np.int32]] | None = None,
        ends: np.ndarray[tuple[int], np.dtype[np.int32]] | None = None,
        min_bound: int = 0,
        max_bound: int | None = None,
    ) -> tuple[int, int]:
        """Snap segment boundaries inward and outward to archetype-relevant genes on the dominant operon strand."""
        left = i_start
        right = i_end - 1
        upper_bound = len(top_roles) if max_bound is None else min(len(top_roles), max_bound)
        min_genes = (
            1
            if (self.allow_single_gene_synthase and archetype == LocusArchetype.CPS_SYNTHASE)
            else self.min_locus_genes
        )

        def _is_operon_coupled_boundary(idx: int, neighbor_idx: int) -> bool:
            if starts is None or ends is None:
                return False
            if int(strands[idx]) != int(strands[neighbor_idx]):
                return False
            gap = (
                int(starts[neighbor_idx]) - int(ends[idx])
                if neighbor_idx > idx
                else int(starts[idx]) - int(ends[neighbor_idx])
            )
            return gap <= self.max_operon_snap_bp and float(biosyn_prob[idx]) >= self.min_biosynthetic_prob

        def _snap_relevant(l_idx: int, r_idx: int) -> tuple[int, int]:
            while l_idx <= r_idx:
                r = GeneRole(int(top_roles[l_idx]))
                is_rel = self.classifier.is_boundary_relevant(archetype, r)
                if is_rel and (
                    biosyn_prob[l_idx] >= self.min_biosynthetic_prob
                    or (l_idx < r_idx and _is_operon_coupled_boundary(l_idx, l_idx + 1))
                ):
                    break
                l_idx += 1
            while r_idx >= l_idx:
                r = GeneRole(int(top_roles[r_idx]))
                is_rel = self.classifier.is_boundary_relevant(archetype, r)
                if is_rel and (
                    biosyn_prob[r_idx] >= self.min_biosynthetic_prob
                    or (r_idx > l_idx and _is_operon_coupled_boundary(r_idx, r_idx - 1))
                ):
                    break
                r_idx -= 1
            return l_idx, r_idx

        left, right = _snap_relevant(left, right)
        if right - left + 1 < min_genes:
            return i_start, i_end

        # Determine dominant strand of core module genes inside [left, right]
        t = self.classifier.tensors
        a_id = int(archetype)
        n_m = int(t.num_modules[a_id])
        mod_mask = 0
        for m_i in range(n_m):
            mod_mask |= int(t.module_masks[a_id, m_i])

        mach_set = {int(r) for r in t.machinery_roles}
        arch_mach_mask = mod_mask & sum(1 << r for r in mach_set)

        # Trim across large inter-operon gaps (>= 2,000 bp) if the outer piece has no archetype machinery
        # and trimming preserves 100% of the segment's module completeness
        mge_roles = (int(GeneRole.TRANSPOSASE), int(GeneRole.INTEGRASE))
        if starts is not None and ends is not None and right - left + 1 > min_genes:
            c_cur, _, _ = self.classifier.score_segments(
                np.array([left], dtype=np.intp),
                np.array([right + 1], dtype=np.intp),
                prefixes=prefixes,
            )
            cur_comp = float(c_cur[0, a_id])

            for k in range(left + 1, right - min_genes + 2):
                if (
                    int(starts[k]) - int(ends[k - 1]) >= 2000
                    and int(top_roles[k - 1]) not in mge_roles
                    and int(top_roles[k]) not in mge_roles
                ):
                    has_outer_mach = any((arch_mach_mask & (1 << int(top_roles[p]))) != 0 for p in range(left, k))
                    if not has_outer_mach:
                        c_re, inc_re, _ = self.classifier.score_segments(
                            np.array([k], dtype=np.intp),
                            np.array([right + 1], dtype=np.intp),
                            prefixes=prefixes,
                        )
                        if (
                            float(c_re[0, a_id]) >= cur_comp - 1e-6
                            and float(c_re[0, a_id]) >= self.min_completeness
                            and float(inc_re[0, a_id]) <= self.max_incompatibility
                        ):
                            left, right = _snap_relevant(k, right)

            for k in range(right, left + min_genes - 1, -1):
                if (
                    int(starts[k]) - int(ends[k - 1]) >= 2000
                    and int(top_roles[k - 1]) not in mge_roles
                    and int(top_roles[k]) not in mge_roles
                ):
                    has_outer_mach = any((arch_mach_mask & (1 << int(top_roles[p]))) != 0 for p in range(k, right + 1))
                    if not has_outer_mach:
                        c_re, inc_re, _ = self.classifier.score_segments(
                            np.array([left], dtype=np.intp),
                            np.array([k], dtype=np.intp),
                            prefixes=prefixes,
                        )
                        if (
                            float(c_re[0, a_id]) >= cur_comp - 1e-6
                            and float(c_re[0, a_id]) >= self.min_completeness
                            and float(inc_re[0, a_id]) <= self.max_incompatibility
                        ):
                            left, right = _snap_relevant(left, k - 1)

        mod_strands = [
            int(strands[k])
            for k in range(left, right + 1)
            if biosyn_prob[k] >= self.min_biosynthetic_prob and (mod_mask & (1 << int(top_roles[k]))) != 0
        ]
        dom_strand = (
            (1 if sum(mod_strands) >= 0 else -1)
            if mod_strands
            else (1 if int(np.sum(strands[left : right + 1])) >= 0 else -1)
        )

        c_cur_st, _, _ = self.classifier.score_segments(
            np.array([left], dtype=np.intp),
            np.array([right + 1], dtype=np.intp),
            prefixes=prefixes,
        )
        cur_comp_st = float(c_cur_st[0, a_id])

        # Trim leading opposite-strand genes if completeness is preserved and remains >= min_completeness
        while right - left + 1 > min_genes and int(strands[left]) != dom_strand:
            c_re, inc_re, _ = self.classifier.score_segments(
                np.array([left + 1], dtype=np.intp),
                np.array([right + 1], dtype=np.intp),
                prefixes=prefixes,
            )
            if (
                float(c_re[0, a_id]) >= cur_comp_st - 1e-6
                and float(c_re[0, a_id]) >= self.min_completeness
                and float(inc_re[0, a_id]) <= self.max_incompatibility
            ):
                left, right = _snap_relevant(left + 1, right)
            else:
                break

        # Trim trailing opposite-strand genes if completeness is preserved and remains >= min_completeness
        while right - left + 1 > min_genes and int(strands[right]) != dom_strand:
            c_re, inc_re, _ = self.classifier.score_segments(
                np.array([left], dtype=np.intp),
                np.array([right], dtype=np.intp),
                prefixes=prefixes,
            )
            if (
                float(c_re[0, a_id]) >= cur_comp_st - 1e-6
                and float(c_re[0, a_id]) >= self.min_completeness
                and float(inc_re[0, a_id]) <= self.max_incompatibility
            ):
                left, right = _snap_relevant(left, right - 1)
            else:
                break

        # Outward operon boundary-snapping pass: include immediately adjacent same-strand genes
        # (< max_operon_snap_bp gap, P(active) >= min_biosynthetic_prob, archetype-relevant & non-MGE/non-forbidden)
        if starts is not None and ends is not None:
            forb_mask = int(archetype.forbidden_mask)
            while left > min_bound:
                cand = left - 1
                if int(strands[cand]) != dom_strand:
                    break
                if int(starts[left]) - int(ends[cand]) > self.max_operon_snap_bp:
                    break
                if float(biosyn_prob[cand]) < self.min_biosynthetic_prob:
                    break
                cand_role = GeneRole(int(top_roles[cand]))
                if not self.classifier.is_boundary_relevant(archetype, cand_role):
                    break
                if (forb_mask & (1 << int(cand_role))) != 0:
                    break
                c_re, inc_re, _ = self.classifier.score_segments(
                    np.array([cand], dtype=np.intp),
                    np.array([right + 1], dtype=np.intp),
                    prefixes=prefixes,
                )
                if float(c_re[0, a_id]) >= self.min_completeness and float(inc_re[0, a_id]) <= self.max_incompatibility:
                    left = cand
                else:
                    break

            while right + 1 < upper_bound:
                cand = right + 1
                if int(strands[cand]) != dom_strand:
                    break
                if int(starts[cand]) - int(ends[right]) > self.max_operon_snap_bp:
                    break
                if float(biosyn_prob[cand]) < self.min_biosynthetic_prob:
                    break
                cand_role = GeneRole(int(top_roles[cand]))
                if not self.classifier.is_boundary_relevant(archetype, cand_role):
                    break
                if (forb_mask & (1 << int(cand_role))) != 0:
                    break
                c_re, inc_re, _ = self.classifier.score_segments(
                    np.array([left], dtype=np.intp),
                    np.array([cand + 1], dtype=np.intp),
                    prefixes=prefixes,
                )
                if float(c_re[0, a_id]) >= self.min_completeness and float(inc_re[0, a_id]) <= self.max_incompatibility:
                    right = cand
                else:
                    break

        if right - left + 1 < min_genes:
            return i_start, i_end
        return left, right + 1

    def _solve_island_dp(
        self,
        island_probs: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        island_starts: np.ndarray[tuple[int], np.dtype[np.int32]],
        island_ends: np.ndarray[tuple[int], np.dtype[np.int32]],
        island_strands: np.ndarray[tuple[int], np.dtype[np.int8]],
    ) -> list[tuple[int, int, LocusArchetype, float, float]]:
        """Run exact ACP-DP interval partitioning on a single candidate island."""
        m = len(island_probs)
        if m == 0:
            return []

        synth_probs = island_probs[:, int(GeneRole.TRANSMEMBRANE_SYNTHASE)]
        has_island_synth = self.allow_single_gene_synthase and bool(np.any(synth_probs >= self.min_synthase_prob))
        if m < self.min_locus_genes and not has_island_synth:
            return []

        t = self.classifier.tensors
        prefixes = self.classifier.precompute_prefixes(island_probs)
        non_unk = island_probs.copy()
        non_unk[:, int(GeneRole.UNKNOWN)] = -1.0
        best_non_unk = np.argmax(non_unk, axis=1).astype(np.int32)
        raw_top = np.argmax(island_probs, axis=1).astype(np.int32)
        biosyn_prob = island_probs[:, t.biosynthetic_mask].sum(axis=1)
        mge_prob = island_probs[:, int(GeneRole.TRANSPOSASE)] + island_probs[:, int(GeneRole.INTEGRASE)]
        is_mge = mge_prob >= self.min_biosynthetic_prob
        top_roles = np.where(
            (raw_top == int(GeneRole.UNKNOWN)) & ((biosyn_prob + mge_prob) >= self.min_biosynthetic_prob),
            best_non_unk,
            raw_top,
        ).astype(np.int32)

        # Precompute internal intergenic transition penalties (strand switches & large gaps),
        # propagating host operon strand across mobile element (TRANSPOSASE/INTEGRASE) insertions
        pref_trans = np.zeros(m, dtype=np.float64)
        if m > 1:
            eff_strands = island_strands.copy()
            for k_s in range(1, m):
                if is_mge[k_s]:
                    eff_strands[k_s] = eff_strands[k_s - 1]
            for k_s in range(m - 2, -1, -1):
                if is_mge[k_s]:
                    eff_strands[k_s] = eff_strands[k_s + 1]
            strand_diff = (eff_strands[:-1] != eff_strands[1:]).astype(np.float64)
            mge_edge = is_mge[:-1] | is_mge[1:]
            gaps_bp = np.where(mge_edge, 0.0, np.maximum(0, island_starts[1:] - island_ends[:-1]).astype(np.float64))
            gap_pen = np.maximum(0.0, (gaps_bp - 500.0) / 1000.0) * 2.0
            edge_pen = strand_diff * self.strand_switch_penalty + gap_pen
            np.cumsum(edge_pen, out=pref_trans[1:])

        min_seg_len = 1 if has_island_synth else self.min_locus_genes
        pair_starts: list[int] = []
        pair_ends: list[int] = []
        for i in range(m):
            max_j = min(m, i + self.max_locus_genes)
            for j in range(i + min_seg_len, max_j + 1):
                if j - i < self.min_locus_genes and not bool(np.any(synth_probs[i:j] >= self.min_synthase_prob)):
                    continue
                pair_starts.append(i)
                pair_ends.append(j)

        if not pair_starts:
            return []

        s_arr = np.array(pair_starts, dtype=np.intp)
        e_arr = np.array(pair_ends, dtype=np.intp)
        comp_mat, incomp_mat, llr_mat = self.classifier.score_segments(s_arr, e_arr, prefixes=prefixes)

        trans_pen_vec = pref_trans[e_arr - 1] - pref_trans[s_arr]
        adj_llr_mat = llr_mat - trans_pen_vec[:, None]
        seg_len = (e_arr - s_arr)[:, None].astype(np.float64)
        pref_mge = np.zeros(m + 1, dtype=np.float64)
        np.cumsum(is_mge.astype(np.float64), out=pref_mge[1:])
        mge_cnt = (pref_mge[e_arr] - pref_mge[s_arr])[:, None]
        eff_seg_len = np.maximum(1.0, seg_len - mge_cnt)

        valid_mask = (
            (comp_mat >= self.min_completeness)
            & (incomp_mat <= self.max_incompatibility)
            & (adj_llr_mat >= self.min_mean_llr * eff_seg_len)
        )
        valid_mask[:, int(LocusArchetype.UNKNOWN)] = False

        # Restrict 1-2 gene segments exclusively to CPS_SYNTHASE
        short_rows = (e_arr - s_arr) < self.min_locus_genes
        if np.any(short_rows):
            cps_synth_id = int(LocusArchetype.CPS_SYNTHASE)
            short_arch_mask = np.zeros(len(LocusArchetype), dtype=np.bool_)
            short_arch_mask[cps_synth_id] = True
            valid_mask[short_rows] &= short_arch_mask[None, :]

        open_pen = np.full(seg_len.shape, self.locus_open_penalty, dtype=np.float64)
        if np.any(short_rows):
            scale = np.minimum(1.0, seg_len[short_rows, 0] / max(self.min_locus_genes, 1))
            open_pen[short_rows, 0] = self.locus_open_penalty * scale

        mod_bonus = (1.5 * t.num_modules[None, :] + 0.02 * t.num_modules[None, :] * seg_len) * (comp_mat**2)
        eff_mat = np.where(
            valid_mask,
            adj_llr_mat * comp_mat - 4.0 * incomp_mat + mod_bonus - open_pen,
            -np.inf,
        )

        best_arch_per_pair = np.argmax(eff_mat, axis=1)
        best_util_per_pair = eff_mat[np.arange(len(s_arr)), best_arch_per_pair]
        best_comp_per_pair = comp_mat[np.arange(len(s_arr)), best_arch_per_pair]

        seg_lookup: dict[tuple[int, int], tuple[float, int, float]] = {}
        for k in range(len(s_arr)):
            u = float(best_util_per_pair[k])
            if u > 0.0:
                seg_lookup[(int(s_arr[k]), int(e_arr[k]))] = (
                    u,
                    int(best_arch_per_pair[k]),
                    float(best_comp_per_pair[k]),
                )

        if not seg_lookup:
            return []

        dp = np.zeros(m + 1, dtype=np.float64)
        back_prev = np.full(m + 1, -1, dtype=np.int32)
        back_arch = np.full(m + 1, int(LocusArchetype.UNKNOWN), dtype=np.int32)
        back_comp = np.zeros(m + 1, dtype=np.float64)
        back_score = np.zeros(m + 1, dtype=np.float64)

        for j in range(1, m + 1):
            dp[j] = dp[j - 1]
            back_prev[j] = j - 1
            back_arch[j] = int(LocusArchetype.UNKNOWN)

            min_i = max(0, j - self.max_locus_genes)
            for i in range(min_i, j - min_seg_len + 1):
                entry = seg_lookup.get((i, j))
                if entry is not None:
                    u, a_id, c_val = entry
                    cand = dp[i] + u
                    if cand > dp[j]:
                        dp[j] = cand
                        back_prev[j] = i
                        back_arch[j] = a_id
                        back_comp[j] = c_val
                        back_score[j] = u

        raw_segments: list[tuple[int, int, LocusArchetype, float, float]] = []
        curr = m
        max_snap_bound = m
        while curr > 0:
            prev = int(back_prev[curr])
            a_id = int(back_arch[curr])
            if a_id != int(LocusArchetype.UNKNOWN):
                arch = LocusArchetype(a_id)
                min_genes_arch = (
                    1
                    if (self.allow_single_gene_synthase and arch == LocusArchetype.CPS_SYNTHASE)
                    else self.min_locus_genes
                )
                min_snap_bound = prev
                while min_snap_bound > 0 and int(back_arch[min_snap_bound]) == int(LocusArchetype.UNKNOWN):
                    min_snap_bound = int(back_prev[min_snap_bound])
                t_start, t_end = self._trim_boundaries(
                    prev,
                    curr,
                    arch,
                    top_roles,
                    biosyn_prob,
                    island_strands,
                    prefixes,
                    starts=island_starts,
                    ends=island_ends,
                    min_bound=min_snap_bound,
                    max_bound=max_snap_bound,
                )
                if t_end - t_start >= min_genes_arch:
                    c_re, inc_re, llr_re = self.classifier.score_segments(
                        np.array([t_start], dtype=np.intp),
                        np.array([t_end], dtype=np.intp),
                        prefixes=prefixes,
                    )
                    comp_val = float(c_re[0, a_id])
                    incomp_val = float(inc_re[0, a_id])
                    t_pen = float(pref_trans[t_end - 1] - pref_trans[t_start])
                    score_val = float(max(0.0, llr_re[0, a_id] - t_pen) * comp_val * (1.0 - incomp_val))
                    if comp_val >= self.min_completeness and incomp_val <= self.max_incompatibility:
                        raw_segments.append((t_start, t_end, arch, comp_val, score_val))
                        max_snap_bound = t_start
                        curr = min(prev, t_start)
                        continue
                    raw_segments.append((prev, curr, arch, float(back_comp[curr]), float(back_score[curr])))
                    max_snap_bound = prev
                else:
                    raw_segments.append((prev, curr, arch, float(back_comp[curr]), float(back_score[curr])))
                    max_snap_bound = prev
            curr = prev

        raw_segments.reverse()
        return raw_segments

    def _compile_virtual_island(
        self,
        contig_ids: set[int],
        pieces: tuple[tuple[int, int, np.ndarray[tuple[int], np.dtype[np.intp]]], ...],
        open_5prime: bool,
        open_3prime: bool,
        probs: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        starts: np.ndarray[tuple[int], np.dtype[np.int32]],
        ends: np.ndarray[tuple[int], np.dtype[np.int32]],
        strands: np.ndarray[tuple[int], np.dtype[np.int8]],
    ) -> _BoundaryIsland:
        """Build virtual coordinates and run `_solve_island_dp` on a sequence of oriented contig pieces."""
        canon_g = np.concatenate([p[2] for p in pieces])
        m = len(canon_g)
        v_probs = probs[canon_g]
        v_starts = np.empty(m, dtype=np.int32)
        v_ends = np.empty(m, dtype=np.int32)
        v_strands = np.empty(m, dtype=np.int8)

        cur_coord = 0
        offset = 0
        for _, step_sign, p_idx in pieces:
            p_len = len(p_idx)
            for k in range(p_len):
                g = int(p_idx[k])
                g_len = max(1, int(ends[g]) - int(starts[g]))
                if k == 0:
                    gap = 50 if offset > 0 else 0
                else:
                    g_prev = int(p_idx[k - 1])
                    gap = int(starts[g]) - int(ends[g_prev]) if step_sign > 0 else int(starts[g_prev]) - int(ends[g])
                cur_coord = max(0, cur_coord + gap)
                v_starts[offset + k] = cur_coord
                cur_coord += g_len
                v_ends[offset + k] = cur_coord
                v_strands[offset + k] = int(strands[g]) * step_sign
            offset += p_len

        dp_segs = self._solve_island_dp(v_probs, v_starts, v_ends, v_strands)
        tot_score = sum(float(s[4]) for s in dp_segs)

        return _BoundaryIsland(
            contig_ids=contig_ids,
            pieces=pieces,
            canon_g_indices=canon_g,
            open_5prime=open_5prime,
            open_3prime=open_3prime,
            v_probs=v_probs,
            v_starts=v_starts,
            v_ends=v_ends,
            v_strands=v_strands,
            dp_segments=dp_segs,
            total_score=tot_score,
        )

    def _machinery_mask(
        self,
        g_indices: np.ndarray[tuple[int], np.dtype[np.intp]],
        probs: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        top_roles: np.ndarray[tuple[int], np.dtype[np.int32]],
    ) -> int:
        """Compute bitmask of strict single-copy transport/polymerization roles present in `g_indices`."""
        single_copy_roles = {
            int(GeneRole.TRANSLOCON),
            int(GeneRole.FLIPPASE),
            int(GeneRole.POLYMERASE),
            int(GeneRole.ABC_PERMEASE),
            int(GeneRole.ABC_ATPASE),
            int(GeneRole.TRANSMEMBRANE_SYNTHASE),
        }
        mask = 0
        for g in g_indices:
            r = int(top_roles[g])
            if r in single_copy_roles and float(probs[g, r]) >= self.min_machinery_prob:
                mask |= 1 << r
        return mask

    def _stitch_boundary_islands(
        self,
        islands: list[_BoundaryIsland],
        probs: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        top_roles: np.ndarray[tuple[int], np.dtype[np.int32]],
        starts: np.ndarray[tuple[int], np.dtype[np.int32]],
        ends: np.ndarray[tuple[int], np.dtype[np.int32]],
        strands: np.ndarray[tuple[int], np.dtype[np.int8]],
        partials: np.ndarray[tuple[int], np.dtype[np.bool_]] | None = None,
    ) -> list[_BoundaryIsland]:
        """Greedily stitch complementary boundary islands (`open_3prime` + `open_5prime`) via virtual ACP-DP."""
        if len(islands) <= 1:
            return islands

        t = self.classifier.tensors
        biosyn_prob_all = probs[:, t.biosynthetic_mask].sum(axis=1)

        active = list(islands)
        while len(active) > 1:
            best_gain = 0.0
            best_pair: tuple[int, int, _BoundaryIsland] | None = None

            for u_i, isl_u in enumerate(active):
                if not isl_u.open_3prime:
                    continue
                m_u = len(isl_u.canon_g_indices)
                for d_i, isl_d in enumerate(active):
                    if u_i == d_i or not isl_d.open_5prime:
                        continue
                    if not isl_u.contig_ids.isdisjoint(isl_d.contig_ids):
                        continue

                    combined = self._compile_virtual_island(
                        contig_ids=isl_u.contig_ids | isl_d.contig_ids,
                        pieces=isl_u.pieces + isl_d.pieces,
                        open_5prime=isl_u.open_5prime,
                        open_3prime=isl_d.open_3prime,
                        probs=probs,
                        starts=starts,
                        ends=ends,
                        strands=strands,
                    )

                    # Find whether DP produced a segment crossing the contig break `m_u`
                    cross_segs = [s for s in combined.dp_segments if s[0] < m_u < s[1]]
                    if not cross_segs:
                        continue

                    seg_s, seg_e, arch, comp_cross, _ = cross_segs[0]
                    a_id = int(arch)
                    left_g = combined.canon_g_indices[seg_s:m_u]
                    right_g = combined.canon_g_indices[m_u:seg_e]

                    # Single-copy core machinery disjointness (excluding a gene sliced in half right at the break)
                    u_last_role = int(top_roles[left_g[-1]])
                    d_first_role = int(top_roles[right_g[0]])
                    is_split_break_gene = (u_last_role == d_first_role) or (
                        partials is not None and (bool(partials[left_g[-1]]) or bool(partials[right_g[0]]))
                    )
                    left_inner = left_g[:-1] if (u_last_role == d_first_role and len(left_g) > 1) else left_g
                    right_inner = right_g[1:] if (u_last_role == d_first_role and len(right_g) > 1) else right_g
                    if (
                        self._machinery_mask(left_inner, probs, top_roles)
                        & self._machinery_mask(right_inner, probs, top_roles)
                    ) != 0:
                        continue

                    # Require active biosynthetic genes on the canonical +1 strand immediately flanking the break `m_u`
                    left_edge_g = left_g[-min(len(left_g), 2) :]
                    right_edge_g = right_g[: min(len(right_g), 2)]
                    left_edge_v = combined.v_strands[m_u - len(left_edge_g) : m_u]
                    right_edge_v = combined.v_strands[m_u : m_u + len(right_edge_g)]
                    if not bool(np.any((biosyn_prob_all[left_edge_g] >= 0.25) & (left_edge_v > 0))):
                        continue
                    if not bool(np.any((biosyn_prob_all[right_edge_g] >= 0.25) & (right_edge_v > 0))):
                        continue

                    # Verify canonical strand coherence and biosynthetic density on both sides of the break
                    strand_frac_l = float(np.mean(combined.v_strands[seg_s:m_u] > 0))
                    strand_frac_r = float(np.mean(combined.v_strands[m_u:seg_e] > 0))
                    if strand_frac_l < 0.70 or strand_frac_r < 0.70:
                        continue

                    active_frac_l = float(np.mean(biosyn_prob_all[left_g] >= self.min_biosynthetic_prob))
                    active_frac_r = float(np.mean(biosyn_prob_all[right_g] >= self.min_biosynthetic_prob))
                    if active_frac_l < 0.50 or active_frac_r < 0.50:
                        continue

                    # Verify both sides of the break are individually compatible with `arch`
                    len_l = len(left_g)
                    len_r = len(right_g)
                    tot_lr = len_l + len_r
                    pref_lr = self.classifier.precompute_prefixes(probs[combined.canon_g_indices[seg_s:seg_e]])
                    comp_mat, inc_mat, llr_mat = self.classifier.score_segments(
                        np.array([0, 0, len_l], dtype=np.intp),
                        np.array([tot_lr, len_l, tot_lr], dtype=np.intp),
                        prefixes=pref_lr,
                    )
                    comp_l = float(comp_mat[1, a_id])
                    comp_r = float(comp_mat[2, a_id])
                    inc_l = float(inc_mat[1, a_id])
                    inc_r = float(inc_mat[2, a_id])
                    llr_l = float(llr_mat[1, a_id])
                    llr_r = float(llr_mat[2, a_id])

                    if inc_l > self.max_incompatibility or inc_r > self.max_incompatibility:
                        continue
                    if llr_l < 0.50 * self.min_mean_llr * len_l or llr_r < 0.50 * self.min_mean_llr * len_r:
                        continue

                    # Complementary module gating: check whether combining both halves strictly increases `arch` completeness
                    has_complementary_modules = comp_cross > max(comp_l, comp_r) + 1e-4

                    # Check standalone segments overlapping left_g in isl_u and right_g in isl_d
                    u_overlaps = [s for s in isl_u.dp_segments if s[1] > seg_s]
                    d_overlaps = [s for s in isl_d.dp_segments if s[0] < (seg_e - m_u)]
                    if u_overlaps and d_overlaps and not has_complementary_modules:
                        continue

                    # When one side already meets min_completeness for `arch`, require either complementary modules
                    # OR a pure sub-threshold cargo continuation (without its own standalone locus) / split boundary gene
                    if comp_l >= self.min_completeness:
                        if (
                            inc_r > 0.15
                            or llr_r < 0.90 * self.min_mean_llr * len_r
                            or active_frac_r < 0.65
                            or strand_frac_r < 0.80
                        ):
                            continue
                        if not has_complementary_modules:
                            if d_overlaps:
                                continue
                            if not is_split_break_gene and (inc_r > 0.05 or llr_r < 1.25 * self.min_mean_llr * len_r):
                                continue
                    if comp_r >= self.min_completeness:
                        if (
                            inc_l > 0.15
                            or llr_l < 0.90 * self.min_mean_llr * len_l
                            or active_frac_l < 0.65
                            or strand_frac_l < 0.80
                        ):
                            continue
                        if not has_complementary_modules:
                            if u_overlaps:
                                continue
                            if not is_split_break_gene and (inc_l > 0.05 or llr_l < 1.25 * self.min_mean_llr * len_l):
                                continue

                    # Compute archetype-consistent standalone baseline under `a_id` so sub-archetype halves
                    # (e.g. a 3' CPS_WZY half that resembles LPS_WZY in isolation) are not double-penalized
                    u_non_cross = sum(float(s[4]) for s in isl_u.dp_segments if s[1] <= seg_s)
                    d_non_cross = sum(float(s[4]) for s in isl_d.dp_segments if s[0] >= (seg_e - m_u))
                    base_l = max(0.0, llr_l) * comp_l * (1.0 - inc_l) if comp_l >= self.min_completeness else 0.0
                    base_r = max(0.0, llr_r) * comp_r * (1.0 - inc_r) if comp_r >= self.min_completeness else 0.0
                    raw_gain = combined.total_score - (u_non_cross + d_non_cross + base_l + base_r)
                    if raw_gain <= 0.0:
                        continue

                    mean_density = 0.5 * ((llr_l / len_l) + (llr_r / len_r))
                    gain = (
                        raw_gain * (0.5 + 0.5 * active_frac_l * active_frac_r) * (strand_frac_l * strand_frac_r)
                        + 2.0 * mean_density * comp_cross
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (u_i, d_i, combined)

            if best_pair is None:
                break

            u_idx, d_idx, merged_isl = best_pair
            next_active = [isl for idx, isl in enumerate(active) if idx != u_idx and idx != d_idx]
            next_active.append(merged_isl)
            active = next_active

        return active

    def _build_single_contig_locus(
        self,
        c_id: int,
        g_indices: np.ndarray[tuple[int], np.dtype[np.intp]],
        arch: LocusArchetype,
        comp: float,
        score: float,
        is_tandem: bool,
        genes: Genes,
        top_roles: np.ndarray[tuple[int], np.dtype[np.int32]],
        frameshifts: np.ndarray[tuple[int], np.dtype[np.bool_]],
        cluster_arr: np.ndarray[tuple[int], np.dtype[np.int32]],
        starts: np.ndarray[tuple[int], np.dtype[np.int32]],
        ends: np.ndarray[tuple[int], np.dtype[np.int32]],
        strands: np.ndarray[tuple[int], np.dtype[np.int8]],
    ) -> Locus:
        """Construct a scalar `Locus` from gene indices on a single contig."""
        n_seg = len(g_indices)
        seg_genes = genes[g_indices]
        seg_roles = top_roles[g_indices]
        seg_fs = frameshifts[g_indices]
        seg_cids = cluster_arr[g_indices]
        seg_starts = starts[g_indices]
        seg_ends = ends[g_indices]
        seg_strands = strands[g_indices]

        nodes = tuple(
            GeneNode(int(seg_roles[k]), cluster_id=int(seg_cids[k]), is_frameshift=bool(seg_fs[k]))
            for k in range(n_seg)
        )
        edges = tuple(
            IntergenicEdge(
                source=k,
                target=k + 1,
                edge_type=IntergenicType.from_adjacent(
                    int(seg_strands[k]),
                    int(seg_strands[k + 1]),
                    gap_bp=int(seg_starts[k + 1]) - int(seg_ends[k]),
                ),
            )
            for k in range(n_seg - 1)
        )

        dom_strand = Strand.FORWARD if int(np.sum(seg_strands)) >= 0 else Strand.REVERSE
        locus_interval = Interval(int(seg_starts.min()), int(seg_ends.max()), dom_strand)

        return Locus(
            nodes=nodes,
            edges=edges,
            genes=seg_genes,
            interval=locus_interval,
            contig_index=int(c_id),
            archetype=arch,
            completeness=comp,
            score=score,
            is_tandem_split=is_tandem,
        )

    def _emit_boundary_island_loci(
        self,
        isl: _BoundaryIsland,
        genes: Genes,
        top_roles: np.ndarray[tuple[int], np.dtype[np.int32]],
        frameshifts: np.ndarray[tuple[int], np.dtype[np.bool_]],
        cluster_arr: np.ndarray[tuple[int], np.dtype[np.int32]],
        starts: np.ndarray[tuple[int], np.dtype[np.int32]],
        ends: np.ndarray[tuple[int], np.dtype[np.int32]],
        strands: np.ndarray[tuple[int], np.dtype[np.int8]],
    ) -> list[Locus]:
        """Convert each DP segment of a `_BoundaryIsland` into a single-contig or multi-contig `Locus`."""
        if not isl.dp_segments:
            return []

        m = len(isl.canon_g_indices)
        piece_ids = np.empty(m, dtype=np.int32)
        contig_ids = np.empty(m, dtype=np.int32)
        step_signs = np.empty(m, dtype=np.int8)
        offset = 0
        for p_idx, (c_id, s_sign, p_arr) in enumerate(isl.pieces):
            p_len = len(p_arr)
            piece_ids[offset : offset + p_len] = p_idx
            contig_ids[offset : offset + p_len] = c_id
            step_signs[offset : offset + p_len] = s_sign
            offset += p_len

        is_tandem = len(isl.dp_segments) > 1
        out_loci: list[Locus] = []

        for seg_s, seg_e, arch, comp, score in isl.dp_segments:
            g_indices = isl.canon_g_indices[seg_s:seg_e]
            n_seg = len(g_indices)
            if n_seg == 0:
                continue

            seg_pieces = piece_ids[seg_s:seg_e]
            seg_steps = step_signs[seg_s:seg_e]
            seg_contigs = contig_ids[seg_s:seg_e]

            # If this segment lies wholly within a single contig piece, emit a normal single-contig Locus
            if bool(np.all(seg_pieces == seg_pieces[0])):
                asc_idx = g_indices if int(seg_steps[0]) > 0 else g_indices[::-1]
                out_loci.append(
                    self._build_single_contig_locus(
                        c_id=int(seg_contigs[0]),
                        g_indices=asc_idx,
                        arch=arch,
                        comp=comp,
                        score=score,
                        is_tandem=is_tandem,
                        genes=genes,
                        top_roles=top_roles,
                        frameshifts=frameshifts,
                        cluster_arr=cluster_arr,
                        starts=starts,
                        ends=ends,
                        strands=strands,
                    )
                )
                continue

            # Otherwise, emit a stitched multi-contig Locus in canonical 5' -> 3' operon order
            seg_genes = genes[g_indices]
            seg_roles = top_roles[g_indices]
            seg_fs = frameshifts[g_indices]
            seg_cids = cluster_arr[g_indices]
            seg_starts = starts[g_indices]
            seg_ends = ends[g_indices]

            nodes = tuple(
                GeneNode(int(seg_roles[k]), cluster_id=int(seg_cids[k]), is_frameshift=bool(seg_fs[k]))
                for k in range(n_seg)
            )
            edges_list: list[IntergenicEdge] = []
            for k in range(n_seg - 1):
                g_u = int(g_indices[k])
                g_v = int(g_indices[k + 1])
                s_u = int(strands[g_u]) * int(seg_steps[k])
                s_v = int(strands[g_v]) * int(seg_steps[k + 1])
                if seg_pieces[k] == seg_pieces[k + 1]:
                    gap_bp = (
                        int(starts[g_v]) - int(ends[g_u]) if seg_steps[k] > 0 else int(starts[g_u]) - int(ends[g_v])
                    )
                    etype = IntergenicType.from_adjacent(s_u, s_v, gap_bp=gap_bp)
                else:
                    orient = IntergenicType.from_adjacent(s_u, s_v, gap_bp=0).orientation
                    etype = IntergenicType(orient, gap_bin=14)
                edges_list.append(IntergenicEdge(source=k, target=k + 1, edge_type=etype))

            locus_interval = Interval(int(seg_starts.min()), int(seg_ends.max()), Strand.FORWARD)
            out_loci.append(
                Locus(
                    nodes=nodes,
                    edges=tuple(edges_list),
                    genes=seg_genes,
                    interval=locus_interval,
                    contig_index=int(seg_contigs[0]),
                    archetype=arch,
                    completeness=comp,
                    score=score,
                    is_tandem_split=is_tandem,
                )
            )

        return out_loci

    def segment(
        self,
        genes: Genes,
        role_probs: np.ndarray | Sequence[int | GeneRole],
        *,
        clusters: Sequence[int] | np.ndarray | None = None,
    ) -> Loci:
        """Segment a [`Genes`][serogen.genes.models.Genes] container and role predictions into [`Loci`][serogen.loci.models.Loci]."""
        n_genes = len(genes)
        if n_genes == 0:
            return Loci.empty()

        probs = self.classifier.to_probability_matrix(role_probs)
        if len(probs) != n_genes:
            raise ValueError(f"Length of role_probs ({len(probs)}) does not match len(genes) ({n_genes}).")

        t = self.classifier.tensors
        non_unk_probs = probs.copy()
        non_unk_probs[:, int(GeneRole.UNKNOWN)] = -1.0
        best_non_unk_roles = np.argmax(non_unk_probs, axis=1).astype(np.int32)

        top_roles = np.argmax(probs, axis=1).astype(np.int32)
        biosyn_prob_all = probs[:, t.biosynthetic_mask].sum(axis=1)
        mge_prob_all = probs[:, int(GeneRole.TRANSPOSASE)] + probs[:, int(GeneRole.INTEGRASE)]
        use_non_unk = (top_roles == int(GeneRole.UNKNOWN)) & (
            (biosyn_prob_all + mge_prob_all) >= self.min_biosynthetic_prob
        )
        top_roles = np.where(use_non_unk, best_non_unk_roles, top_roles).astype(np.int32)

        starts = genes.intervals.starts
        ends = genes.intervals.ends
        strands = genes.intervals.strands
        contig_indices = (
            genes.contig_indices if len(genes.contig_indices) == n_genes else np.zeros(n_genes, dtype=np.uint32)
        )
        frameshifts = genes.has_frameshift
        partials = genes.is_partial if len(genes.is_partial) == n_genes else np.zeros(n_genes, dtype=np.bool_)
        cluster_arr = (
            np.asarray(clusters, dtype=np.int32) if clusters is not None else np.zeros(n_genes, dtype=np.int32)
        )

        unique_contigs = np.unique(contig_indices)
        multi_contig_mode = self.stitch_contig_splits and len(unique_contigs) > 1

        discovered_loci: list[Locus] = []
        boundary_islands: list[_BoundaryIsland] = []

        for c_id in unique_contigs:
            c_mask = contig_indices == c_id
            c_global_idx = np.nonzero(c_mask)[0]
            n_c = len(c_global_idx)
            if n_c == 0 or (
                not multi_contig_mode and not self.allow_single_gene_synthase and n_c < self.min_locus_genes
            ):
                continue

            c_starts = starts[c_global_idx]
            c_ends = ends[c_global_idx]
            c_strands = strands[c_global_idx]
            c_probs = probs[c_global_idx]
            c_partials = partials[c_global_idx]

            order = np.argsort(c_starts, kind="stable")
            c_global_idx = c_global_idx[order]
            c_starts = c_starts[order]
            c_ends = c_ends[order]
            c_strands = c_strands[order]
            c_probs = c_probs[order]
            c_partials = c_partials[order]

            contig_min_start = int(c_starts[0])
            contig_max_end = int(c_ends[-1])

            islands = self._find_candidate_islands(
                c_starts,
                c_ends,
                c_probs,
                allow_boundary_partials=multi_contig_mode,
            )
            for raw_isl_start, raw_isl_end in islands:
                raw_touches_start = (
                    (raw_isl_start <= 2 and int(c_starts[raw_isl_start]) - contig_min_start <= 2500)
                    or (int(c_starts[raw_isl_start]) - contig_min_start <= 1200)
                    or bool(c_partials[raw_isl_start])
                )
                raw_touches_end = (
                    (raw_isl_end - 1 >= n_c - 3 and contig_max_end - int(c_ends[raw_isl_end - 1]) <= 2500)
                    or (contig_max_end - int(c_ends[raw_isl_end - 1]) <= 1200)
                    or bool(c_partials[raw_isl_end - 1])
                )

                # If a contig island touches BOTH contig ends on opposite strands, split at the internal strand/gap switch
                sub_islands: list[tuple[int, int, bool, bool]] = [
                    (raw_isl_start, raw_isl_end, raw_touches_start, raw_touches_end)
                ]
                if multi_contig_mode and raw_touches_start and raw_touches_end and (raw_isl_end - raw_isl_start) >= 6:
                    raw_g = c_global_idx[raw_isl_start:raw_isl_end]
                    raw_act = raw_g[biosyn_prob_all[raw_g] >= self.min_biosynthetic_prob]
                    if len(raw_act) >= 4:
                        k_e = min(len(raw_act), 5)
                        s_g = raw_act[:k_e]
                        e_g = raw_act[-k_e:]
                        s_w = np.arange(k_e, 0, -1, dtype=np.float64) * biosyn_prob_all[s_g]
                        e_w = np.arange(1, k_e + 1, dtype=np.float64) * biosyn_prob_all[e_g]
                        s_dir = 1 if float(np.dot(strands[s_g], s_w)) >= 0.0 else -1
                        e_dir = 1 if float(np.dot(strands[e_g], e_w)) >= 0.0 else -1
                        if s_dir != e_dir:
                            best_k = raw_isl_start + (raw_isl_end - raw_isl_start) // 2
                            best_sep = -1e9
                            for k_split in range(raw_isl_start + 2, raw_isl_end - 1):
                                coh_l = float(np.mean(c_strands[raw_isl_start:k_split] == s_dir))
                                coh_r = float(np.mean(c_strands[k_split:raw_isl_end] == e_dir))
                                gap_k = max(0, int(c_starts[k_split]) - int(c_ends[k_split - 1]))
                                sep = coh_l + coh_r + 0.0005 * gap_k
                                if sep > best_sep:
                                    best_sep = sep
                                    best_k = k_split
                            sub_islands = [
                                (raw_isl_start, best_k, True, False),
                                (best_k, raw_isl_end, False, True),
                            ]

                for isl_start, isl_end, touches_start, touches_end in sub_islands:
                    if multi_contig_mode and (touches_start or touches_end):
                        isl_g_idx = c_global_idx[isl_start:isl_end]
                        active_g = isl_g_idx[biosyn_prob_all[isl_g_idx] >= self.min_biosynthetic_prob]
                        if len(active_g) == 0:
                            active_g = isl_g_idx
                        if touches_end and not touches_start:
                            prox_g = active_g[-min(len(active_g), 8) :]
                            weights = np.arange(1, len(prox_g) + 1, dtype=np.float64) * biosyn_prob_all[prox_g]
                        elif touches_start and not touches_end:
                            prox_g = active_g[: min(len(active_g), 8)]
                            weights = np.arange(len(prox_g), 0, -1, dtype=np.float64) * biosyn_prob_all[prox_g]
                        else:
                            prox_g = active_g
                            weights = biosyn_prob_all[prox_g]

                        dom_strand = 1 if float(np.dot(strands[prox_g], weights)) >= 0.0 else -1
                        if dom_strand >= 0:
                            canon_idx = isl_g_idx.copy()
                            open_5p = touches_start
                            open_3p = touches_end
                            step_sign = 1
                        else:
                            canon_idx = isl_g_idx[::-1].copy()
                            open_5p = touches_end
                            open_3p = touches_start
                            step_sign = -1

                        boundary_islands.append(
                            self._compile_virtual_island(
                                contig_ids={int(c_id)},
                                pieces=((int(c_id), step_sign, canon_idx),),
                                open_5prime=open_5p,
                                open_3prime=open_3p,
                                probs=probs,
                                starts=starts,
                                ends=ends,
                                strands=strands,
                            )
                        )
                        continue

                    isl_probs = c_probs[isl_start:isl_end]
                    isl_starts = c_starts[isl_start:isl_end]
                    isl_ends = c_ends[isl_start:isl_end]
                    isl_strands = c_strands[isl_start:isl_end]

                    segments = self._solve_island_dp(isl_probs, isl_starts, isl_ends, isl_strands)
                    is_tandem = len(segments) > 1

                    for seg_s, seg_e, arch, comp, score in segments:
                        g_indices = c_global_idx[isl_start + seg_s : isl_start + seg_e]
                        if len(g_indices) == 0:
                            continue
                        discovered_loci.append(
                            self._build_single_contig_locus(
                                c_id=int(c_id),
                                g_indices=g_indices,
                                arch=arch,
                                comp=comp,
                                score=score,
                                is_tandem=is_tandem,
                                genes=genes,
                                top_roles=top_roles,
                                frameshifts=frameshifts,
                                cluster_arr=cluster_arr,
                                starts=starts,
                                ends=ends,
                                strands=strands,
                            )
                        )

        if boundary_islands:
            stitched_islands = self._stitch_boundary_islands(
                boundary_islands,
                probs=probs,
                top_roles=top_roles,
                starts=starts,
                ends=ends,
                strands=strands,
                partials=partials,
            )
            for b_isl in stitched_islands:
                discovered_loci.extend(
                    self._emit_boundary_island_loci(
                        b_isl,
                        genes=genes,
                        top_roles=top_roles,
                        frameshifts=frameshifts,
                        cluster_arr=cluster_arr,
                        starts=starts,
                        ends=ends,
                        strands=strands,
                    )
                )

        if not discovered_loci:
            return Loci.empty()
        return Loci.from_list(discovered_loci)
