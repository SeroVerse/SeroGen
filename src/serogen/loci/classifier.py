"""Probabilistic and bitmask-based classifier for biosynthetic locus archetypes."""

from collections.abc import Sequence
from typing import Any, Self

import numpy as np
from scipy.optimize import minimize

from serogen.genes.roles import GeneRole
from serogen.loci.archetypes import LocusArchetype
from serogen.loci.models import ArchetypeTensors, Loci, Locus, get_archetype_tensors

ArchetypeInput = (
    Locus | Loci | np.ndarray | Sequence[int | GeneRole] | Sequence[Locus | np.ndarray | Sequence[int | GeneRole]]
)


class ArchetypeClassifier:
    """Scikit-learn compatible classifier and segment scorer for `LocusArchetype`s.

    Encapsulates all discrete bitmask evaluations and continuous noisy-OR log-likelihood-ratio
    (LLR) scoring over [`ArchetypeTensors`][serogen.loci.models.ArchetypeTensors]. Works out-of-the-box
    with domain-calibrated weights (`w_comp`, `w_incomp`, `w_llr`, `unknown_logit`), and optionally
    supports empirical weight calibration via `.fit(X, y)` on labeled loci.
    """

    def __init__(
        self,
        min_completeness: float = 0.80,
        max_incompatibility: float = 0.25,
        w_comp: float = 10.0,
        w_incomp: float = 15.0,
        w_llr: float = 0.15,
        unknown_logit: float = 4.5,
        tensors: ArchetypeTensors | None = None,
    ) -> None:
        self.min_completeness = min_completeness
        self.max_incompatibility = max_incompatibility
        self.w_comp = w_comp
        self.w_incomp = w_incomp
        self.w_llr = w_llr
        self.unknown_logit = unknown_logit
        self.tensors = tensors if tensors is not None else get_archetype_tensors()

    @property
    def classes_(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """1D array of archetype class labels (`0..10`)."""
        return np.arange(len(LocusArchetype), dtype=np.int32)

    @property
    def active_weights(self) -> tuple[float, float, float, float]:
        """Return `(w_comp, w_incomp, w_llr, unknown_logit)`, preferring fitted parameters if present."""
        w_c = getattr(self, "w_comp_", self.w_comp)
        w_i = getattr(self, "w_incomp_", self.w_incomp)
        w_l = getattr(self, "w_llr_", self.w_llr)
        u_l = getattr(self, "unknown_logit_", self.unknown_logit)
        return float(w_c), float(w_i), float(w_l), float(u_l)

    # ------------------------------------------------------------------------------------------------------------------
    # Discrete Bitmask Operations
    # ------------------------------------------------------------------------------------------------------------------
    def check_modules(self, mask: int) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """Count how many required functional modules are satisfied for each archetype (`(11,)`)."""
        m = np.uint32(mask)
        satisfied = (self.tensors.module_masks & m) != 0
        satisfied &= self.tensors.active_module_mask
        return np.sum(satisfied, axis=1, dtype=np.int32)

    def check_forbidden(self, mask: int) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        """Return a boolean array `(11,)` indicating whether any forbidden role is present."""
        return (self.tensors.forbidden_masks & np.uint32(mask)) != 0

    def is_complete(
        self, mask: int, min_completeness: float | None = None
    ) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        """Evaluate whether a 32-bit role presence mask satisfies each archetype."""
        thresh = self.min_completeness if min_completeness is None else min_completeness
        mod_counts = self.check_modules(mask)
        no_forbidden = ~self.check_forbidden(mask)
        has_modules = self.tensors.num_modules > 0
        denom = np.maximum(self.tensors.num_modules, 1)
        completeness = np.where(has_modules, mod_counts / denom, 0.0)
        return has_modules & no_forbidden & (completeness >= thresh)

    def classify_mask(
        self,
        mask: int,
        min_completeness: float | None = None,
    ) -> tuple[LocusArchetype, float]:
        """Classify a single 32-bit role presence mask into its best-matching `LocusArchetype`."""
        thresh = self.min_completeness if min_completeness is None else min_completeness
        mod_counts = self.check_modules(mask)
        no_forbidden = ~self.check_forbidden(mask)
        has_modules = self.tensors.num_modules > 0
        denom = np.maximum(self.tensors.num_modules, 1)
        completeness = np.where(has_modules & no_forbidden, mod_counts / denom, 0.0)

        cargo_overlap = np.array(
            [(int(c_mask) & int(mask)).bit_count() for c_mask in self.tensors.all_relevant_masks],
            dtype=np.float64,
        )
        ranking = completeness + 1e-3 * mod_counts + 1e-5 * cargo_overlap
        ranking[int(LocusArchetype.UNKNOWN)] = -1.0

        best_idx = int(np.argmax(ranking))
        best_comp = float(completeness[best_idx])
        if best_comp < thresh or best_comp == 0.0:
            return LocusArchetype.UNKNOWN, best_comp
        return LocusArchetype(best_idx), best_comp

    def is_boundary_relevant(self, archetype: LocusArchetype | int, role: GeneRole | int) -> bool:
        """Return True if `role` is a valid boundary/anchor/cargo gene for `archetype` (excluding MGEs/UNKNOWN)."""
        r_idx = int(role)
        if r_idx in (int(GeneRole.TRANSPOSASE), int(GeneRole.INTEGRASE), int(GeneRole.UNKNOWN)):
            return False
        a_idx = int(archetype)
        return (int(self.tensors.all_relevant_masks[a_idx]) & (1 << r_idx)) != 0

    # ------------------------------------------------------------------------------------------------------------------
    # Prefix-Sum Vectorized Soft Probability Scoring
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def to_probability_matrix(
        roles_or_probs: np.ndarray | Sequence[int | GeneRole],
        confidence: float = 0.98,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Convert 1D discrete role indices or a 2D probability matrix into a clean `(L, 30)` float64 matrix."""
        arr = np.asarray(roles_or_probs)
        n_roles = len(GeneRole)
        if arr.ndim == 2:
            if arr.shape[1] != n_roles:
                raise ValueError(f"Expected 2D probability matrix with {n_roles} columns, got shape {arr.shape}.")
            probs = np.clip(arr.astype(np.float64, copy=False), 1e-12, 1.0)
            row_sums = probs.sum(axis=1, keepdims=True)
            return np.where(row_sums > 0.0, probs / row_sums, 1.0 / n_roles)

        if arr.ndim != 1:
            raise ValueError(f"Expected 1D role index array or 2D probability matrix, got {arr.ndim}D.")

        length = len(arr)
        if length == 0:
            return np.empty((0, n_roles), dtype=np.float64)

        smooth = (1.0 - confidence) / (n_roles - 1)
        probs = np.full((length, n_roles), smooth, dtype=np.float64)
        valid_mask = (arr >= 0) & (arr < n_roles)
        rows = np.nonzero(valid_mask)[0]
        cols = arr[valid_mask].astype(np.intp)
        probs[rows, cols] = confidence
        if not np.all(valid_mask):
            probs[~valid_mask, int(GeneRole.UNKNOWN)] = confidence
        return probs

    def precompute_prefixes(
        self,
        probs: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> tuple[
        np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ]:
        """Precompute $O(L)$ prefix sums of log-complement and log-odds probabilities across a contig."""
        t = self.tensors
        length = len(probs)
        n_arch, max_mods, _ = t.module_indicators.shape

        mod_probs = np.where(probs >= 0.12, probs, 0.0)
        mod_hit = np.clip(np.einsum("lr,amr->lam", mod_probs, t.module_indicators), 0.0, 1.0)
        log_not_mod = np.log(np.clip(1.0 - mod_hit, 1e-12, 1.0))

        forb_probs = np.where(probs >= 0.05, probs, 0.0)
        forb_hit = np.clip(forb_probs @ t.forbidden_indicators.T, 0.0, 1.0)
        log_not_forb = np.log(np.clip(1.0 - forb_hit, 1e-12, 1.0))

        null_mass = probs @ t.null_priors
        rel_mass = probs @ t.relevant_indicators.T
        arch_emission = 0.85 * rel_mass + 0.15 * null_mass[:, None]
        forb_penalty = np.log(np.clip(1.0 - 0.95 * forb_hit, 1e-6, 1.0))
        step_llr = np.log(np.clip(arch_emission, 1e-12, None)) - np.log(np.clip(null_mass[:, None], 1e-12, None))
        step_llr += forb_penalty
        step_llr[:, int(LocusArchetype.UNKNOWN)] = 0.0

        pref_mod = np.zeros((length + 1, n_arch, max_mods), dtype=np.float64)
        pref_forb = np.zeros((length + 1, n_arch), dtype=np.float64)
        pref_llr = np.zeros((length + 1, n_arch), dtype=np.float64)

        if length > 0:
            np.cumsum(log_not_mod, axis=0, out=pref_mod[1:])
            np.cumsum(log_not_forb, axis=0, out=pref_forb[1:])
            np.cumsum(step_llr, axis=0, out=pref_llr[1:])

        return pref_mod, pref_forb, pref_llr

    def score_segments(
        self,
        starts: np.ndarray[tuple[int], np.dtype[np.intp]],
        ends: np.ndarray[tuple[int], np.dtype[np.intp]],
        *,
        probs: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None,
        prefixes: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> tuple[
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ]:
        """Evaluate completeness, incompatibility, and LLR scores for $K$ candidate segments in $O(K \\cdot |\\mathcal{A}|)$."""
        if prefixes is None:
            if probs is None:
                raise ValueError("Either `probs` or `prefixes` must be provided to `score_segments`.")
            prefixes = self.precompute_prefixes(probs)

        t = self.tensors
        pref_mod, pref_forb, pref_llr = prefixes
        seg_log_not_mod = pref_mod[ends] - pref_mod[starts]
        raw_mod_probs = 1.0 - np.exp(seg_log_not_mod)
        mod_probs = np.minimum(1.0, raw_mod_probs / 0.95) * t.active_module_mask[None, :, :]
        denom = np.maximum(t.num_modules, 1)[None, :]
        has_mods = (t.num_modules > 0)[None, :]
        completeness = np.where(has_mods, mod_probs.sum(axis=2) / denom, 0.0)

        seg_log_not_forb = pref_forb[ends] - pref_forb[starts]
        incompatibility = 1.0 - np.exp(seg_log_not_forb)

        llr_scores = pref_llr[ends] - pref_llr[starts]
        return completeness, incompatibility, llr_scores

    # ------------------------------------------------------------------------------------------------------------------
    # Scikit-Learn Estimator Interface & Locus/Loci Classification
    # ------------------------------------------------------------------------------------------------------------------
    def _score_single(
        self,
        item: Locus | np.ndarray | Sequence[int | GeneRole],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(item, Locus):
            if item.num_genes == 0:
                z = np.zeros(len(LocusArchetype), dtype=np.float64)
                return z, z, z
            probs = self.to_probability_matrix(item.roles)
        else:
            probs = self.to_probability_matrix(item)

        if len(probs) == 0:
            z = np.zeros(len(LocusArchetype), dtype=np.float64)
            return z, z, z

        comp, incomp, llr = self.score_segments(
            np.array([0], dtype=np.intp),
            np.array([len(probs)], dtype=np.intp),
            probs=probs,
        )
        return llr[0], comp[0], incomp[0]

    def score_components(
        self,
        X: ArchetypeInput,
    ) -> tuple[
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ]:
        """Compute `(llr_scores, completeness, incompatibility)` matrices of shape `(N, 11)` across a batch."""
        items: list[Any]
        if isinstance(X, Loci):
            items = [X[i] for i in range(len(X))]
        elif isinstance(X, Locus):
            items = [X]
        elif isinstance(X, np.ndarray):
            if X.ndim in (1, 2):
                items = [X]
            else:
                items = [X[i] for i in range(len(X))]
        else:
            seq: list[Any] = list(X)
            if len(seq) == 0:
                e = np.empty((0, len(LocusArchetype)), dtype=np.float64)
                return e, e, e
            if isinstance(seq[0], (int, np.integer, GeneRole)):
                items = [seq]
            else:
                items = seq

        n = len(items)
        n_arch = len(LocusArchetype)
        scores = np.zeros((n, n_arch), dtype=np.float64)
        comps = np.zeros((n, n_arch), dtype=np.float64)
        incomps = np.zeros((n, n_arch), dtype=np.float64)

        for i, it in enumerate(items):
            s, c, inc = self._score_single(it)
            scores[i] = s
            comps[i] = c
            incomps[i] = inc
        return scores, comps, incomps

    def decision_function(self, X: ArchetypeInput) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Compute unnormalized archetype logits of shape `(N, 11)` for input segments."""
        scores, comps, incomps = self.score_components(X)
        w_c, w_i, w_l, u_l = self.active_weights
        logits = w_c * comps - w_i * incomps + w_l * np.maximum(scores, 0.0)
        logits[:, int(LocusArchetype.UNKNOWN)] = u_l
        return logits

    def predict_proba(self, X: ArchetypeInput) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Compute calibrated softmax posterior probabilities `(N, 11)` across archetypes."""
        logits = self.decision_function(X)
        if len(logits) == 0:
            return logits
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(shifted)
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)

    def predict(self, X: ArchetypeInput) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """Predict the integer `LocusArchetype` index (`0..10`) for each input locus or segment."""
        scores, comps, incomps = self.score_components(X)
        n = len(scores)
        if n == 0:
            return np.empty(0, dtype=np.int32)

        w_c, w_i, w_l, _ = self.active_weights
        eff = comps * (1.0 - incomps) + 1e-2 * (w_l * np.maximum(scores, 0.0) / max(w_c, 1.0))
        eff[:, int(LocusArchetype.UNKNOWN)] = -1.0

        valid = (comps >= self.min_completeness) & (incomps <= self.max_incompatibility)
        valid[:, int(LocusArchetype.UNKNOWN)] = False

        masked_eff = np.where(valid, eff, -1.0)
        best_idx = np.argmax(masked_eff, axis=1).astype(np.int32)
        has_valid = np.any(valid, axis=1)
        return np.where(has_valid, best_idx, int(LocusArchetype.UNKNOWN)).astype(np.int32)

    def classify(
        self,
        X: Locus | np.ndarray | Sequence[int | GeneRole],
    ) -> tuple[LocusArchetype, float, float]:
        """Classify a single `Locus`, role sequence, or `(L, 30)` probability matrix into `(archetype, completeness, score)`."""
        scores, comps, incomps = self.score_components([X])
        s = scores[0]
        c = comps[0]
        inc = incomps[0]

        eff = c * (1.0 - inc) + 1e-2 * np.maximum(s, 0.0)
        eff[int(LocusArchetype.UNKNOWN)] = -1.0
        best_idx = int(np.argmax(eff))
        best_comp = float(c[best_idx])
        best_incomp = float(inc[best_idx])
        best_score = float(s[best_idx])

        if best_comp < self.min_completeness or best_incomp > self.max_incompatibility or best_score <= 0.0:
            return LocusArchetype.UNKNOWN, best_comp, best_score
        return LocusArchetype(best_idx), best_comp, best_score

    def fit(self, X: ArchetypeInput, y: Sequence[int | LocusArchetype] | np.ndarray) -> Self:
        """Calibrate `w_comp_`, `w_incomp_`, `w_llr_`, and `unknown_logit_` via L-BFGS-B cross-entropy minimization."""
        scores, comps, incomps = self.score_components(X)
        y_arr = np.asarray([int(lbl) for lbl in y], dtype=np.int32)
        n = len(y_arr)
        if n == 0:
            self.w_comp_ = self.w_comp
            self.w_incomp_ = self.w_incomp
            self.w_llr_ = self.w_llr
            self.unknown_logit_ = self.unknown_logit
            return self

        pos_scores = np.maximum(scores, 0.0)
        unk_idx = int(LocusArchetype.UNKNOWN)

        def _loss(params: np.ndarray) -> float:
            wc, wi, wl, ul = params
            logits = wc * comps - wi * incomps + wl * pos_scores
            logits[:, unk_idx] = ul
            logits = logits - np.max(logits, axis=1, keepdims=True)
            log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
            nll = -float(np.mean(log_probs[np.arange(n), y_arr]))
            reg = 1e-3 * float(
                (wc - self.w_comp) ** 2
                + (wi - self.w_incomp) ** 2
                + (wl - self.w_llr) ** 2
                + (ul - self.unknown_logit) ** 2
            )
            return nll + reg

        init = np.array([self.w_comp, self.w_incomp, self.w_llr, self.unknown_logit], dtype=np.float64)
        bounds = [(1.0, 50.0), (1.0, 50.0), (0.01, 5.0), (0.0, 20.0)]
        res = minimize(_loss, init, method="L-BFGS-B", bounds=bounds)
        self.w_comp_ = float(res.x[0])
        self.w_incomp_ = float(res.x[1])
        self.w_llr_ = float(res.x[2])
        self.unknown_logit_ = float(res.x[3])
        return self

    def score(self, X: ArchetypeInput, y: Sequence[int | LocusArchetype] | np.ndarray) -> float:
        """Compute classification accuracy on labeled segments `(X, y)`."""
        y_arr = np.asarray([int(lbl) for lbl in y], dtype=np.int32)
        if len(y_arr) == 0:
            return 1.0
        preds = self.predict(X)
        return float(np.mean(preds == y_arr))
