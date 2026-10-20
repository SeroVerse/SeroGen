"""Classifying gene roles from amino acid sequences using a 1D Convolutional
Neural Network, trained via compiled Numba AdamW backpropagation with exact
dual-pooling (ReLU max-pooling + ReLU average-pooling) gradient routing.
"""

from collections.abc import Sequence
from functools import cache
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Self

import numpy as np
from numba import njit, prange

from serogen.core.seq.alphabet import Alphabet
from serogen.core.seq.seq import Sequence as SeqRecord
from serogen.core.seq.seq import Sequences
from serogen.genes.models import Genes
from serogen.genes.roles import GeneRole


# ----------------------------------------------------------------------------------------------------------------------
# GeneRoleClassifier Class
# ----------------------------------------------------------------------------------------------------------------------
class GeneRoleClassifier:
    r"""Classifies `GeneRole`s from amino acid sequences using a Multi-Scale 1D-CNN with Dual Pooling.

    Learns representations from variable-length sequence token arrays using an embedding layer,
    multi-scale 1D convolutions (e.g. receptive fields 5, 9, 13), ReLU activation, dual pooling
    (global max pooling + global average pooling) concatenated with normalized log-length, and a
    dense classification layer. Optimization is performed via a parallel compiled Numba AdamW
    training kernel with exact gradient routing.

    Attributes:
        vocab_size: Number of discrete tokens in sequence vocabulary (default 23: 22 AMINO tokens + 1 unknown/pad).
        embed_dim: Dimensionality of token embedding vectors.
        num_filters: Number of 1D convolutional feature channels across all kernel sizes.
        kernel_size: Width of 1D convolution receptive field window (int), or tuple of multi-scale
            kernel sizes (e.g. `(5, 9, 13)`).
        learning_rate: AdamW step size hyperparameter.
        epochs: Number of complete training epochs over dataset.
        batch_size: Mini-batch size for parallel AdamW gradient accumulation and updates.
        weight_decay: Decoupled L2 weight decay coefficient applied to model parameters.
        random_state: Optional seed for reproducible parameter initialization and sample shuffling.
        class_weight: Optional `"balanced"` inverse-frequency class weighting or `None`.
    """

    classes_: np.ndarray[tuple[int], np.dtype[Any]]
    embedding_: np.ndarray[tuple[int, int], np.dtype[np.float32]]
    conv_w_: np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
    conv_b_: np.ndarray[tuple[int], np.dtype[np.float32]]
    dense_w_: np.ndarray[tuple[int, int], np.dtype[np.float32]]
    dense_b_: np.ndarray[tuple[int], np.dtype[np.float32]]
    loss_history_: list[float]
    lut_: np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
    k_sizes_: np.ndarray[tuple[int], np.dtype[np.int32]]
    unique_ks_: np.ndarray[tuple[int], np.dtype[np.int32]]
    k_indices_: np.ndarray[tuple[int], np.dtype[np.int32]]

    def __init__(
        self,
        vocab_size: int = 23,
        embed_dim: int = 16,
        num_filters: int = 64,
        kernel_size: int | tuple[int, ...] = (5, 9, 13),
        learning_rate: float = 0.01,
        epochs: int = 20,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        random_state: int | None = None,
        class_weight: str | None = "balanced",
    ) -> None:
        """Initialize the `GeneRoleClassifier` with architecture and optimizer hyperparameters."""
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.class_weight = class_weight

    @property
    def is_fitted(self) -> bool:
        """Return `True` if all model parameters have been fitted or loaded."""
        return all(
            getattr(self, attr, None) is not None
            for attr in ("classes_", "embedding_", "conv_w_", "conv_b_", "dense_w_", "dense_b_", "k_sizes_")
        )

    @property
    def num_classes(self) -> int:
        """Number of target classes learned by the fitted model."""
        return len(self.classes_) if getattr(self, "classes_", None) is not None else 0

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the penultimate 1D-CNN + global composition feature vector."""
        if getattr(self, "dense_w_", None) is not None:
            return int(self.dense_w_.shape[0])
        return 2 * int(self.num_filters) + 2 + int(self.vocab_size)

    def _pack_ragged(
        self,
        X: Genes | Sequences | Sequence[Any] | tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray[tuple[int], np.dtype[np.uint8]], np.ndarray[tuple[int], np.dtype[np.int64]]]:
        """Pack sequence inputs into a flat `uint8` token array (`0..vocab_size-1`) and `int64` CSR `offsets`."""
        unk_tok = np.uint8(max(0, self.vocab_size - 1))

        # 1. Genes container -> extract .translations Sequences
        if isinstance(X, Genes):
            X = X.translations

        # 2. Native Sequences (RaggedArrayContainer) -> zero-loop vectorized encoding
        if isinstance(X, Sequences):
            raw = X.seqs
            if len(raw) == 0:
                return np.empty(0, dtype=np.uint8), X.indptr.astype(np.int64, copy=False)
            if int(np.max(raw)) >= self.vocab_size:
                encoded = Alphabet.AMINO.encode(raw)
                flat_u8 = np.minimum(encoded, unk_tok)
            else:
                flat_u8 = raw.astype(np.uint8, copy=False)
            return flat_u8, X.indptr.astype(np.int64, copy=False)

        # 3. Pre-packed (flat_tokens, offsets) tuple
        if isinstance(X, tuple) and len(X) == 2 and isinstance(X[0], np.ndarray) and isinstance(X[1], np.ndarray):
            raw_flat = X[0]
            if len(raw_flat) > 0 and raw_flat.dtype == np.uint8 and int(np.max(raw_flat)) >= self.vocab_size:
                flat_u8 = np.minimum(Alphabet.AMINO.encode(raw_flat), unk_tok)
            else:
                flat_u8 = np.clip(raw_flat, 0, int(unk_tok)).astype(np.uint8, copy=False)
            offsets = X[1].astype(np.int64, copy=False)
            if len(offsets) > 0 and offsets[-1] != len(flat_u8):
                offsets = np.append(offsets, len(flat_u8))
            return flat_u8, offsets

        # 4. Sequence of token arrays, byte strings, strings, or Sequence scalar objects
        seq_list = list(X)
        n_seqs = len(seq_list)
        if n_seqs == 0:
            return np.empty(0, dtype=np.uint8), np.zeros(1, dtype=np.int64)

        first = seq_list[0]
        if isinstance(first, (bytes, bytearray, str, SeqRecord)):
            bytes_list = [
                s.seq if isinstance(s, SeqRecord) else (s.encode("ascii") if isinstance(s, str) else bytes(s))
                for s in seq_list
            ]
            seqs_obj = Sequences.from_bytes(bytes_list)
            encoded = Alphabet.AMINO.encode(seqs_obj.seqs)
            return np.minimum(encoded, unk_tok), seqs_obj.indptr.astype(np.int64, copy=False)

        offsets = np.zeros(n_seqs + 1, dtype=np.int64)
        total_len = sum(len(s) for s in seq_list)
        flat = np.empty(total_len, dtype=np.uint8)
        curr = 0
        for i, s in enumerate(seq_list):
            n = len(s)
            if n > 0:
                arr = np.asarray(s)
                if arr.dtype == np.uint8 and int(np.max(arr)) >= self.vocab_size:
                    flat[curr : curr + n] = np.minimum(Alphabet.AMINO.encode(arr), unk_tok)
                else:
                    flat[curr : curr + n] = np.clip(arr, 0, int(unk_tok)).astype(np.uint8, copy=False)
            curr += n
            offsets[i + 1] = curr
        return flat, offsets

    def fit(
        self,
        X: Genes | Sequences | Sequence[Any] | tuple[np.ndarray, np.ndarray],
        y: Sequence[int | GeneRole] | np.ndarray,
        sample_weight: Sequence[float] | np.ndarray | None = None,
        *,
        init_from: GeneRoleClassifier | None = None,
        backbone_lr_scale: float = 1.0,
    ) -> Self:
        """Fit the 1D-CNN using parallel mini-batch AdamW optimization over ragged token sequences.

        Args:
            X: [`Genes`][serogen.genes.models.Genes], [`Sequences`][serogen.core.seq.seq.Sequences],
                sequence of 1D integer token arrays, or `(flat_tokens, offsets)` tuple.
            y: 1D array-like of integer class labels (e.g. `GeneRole` indices, `cluster_id`s,
                or bit-packed 32-bit `LinkageType` / `ResidueType` integers).
            sample_weight: Optional 1D array-like of per-sample confidence weights in `[0, inf)`.
            init_from: Optional fitted `GeneRoleClassifier` from which to warm-start the
                `embedding_`, `conv_w_`, and `conv_b_` 1D-CNN backbone weights.
            backbone_lr_scale: Multiplier in `[0.0, 1.0]` applied to the learning rate of the
                `embedding_` and `conv_w_` / `conv_b_` layers (useful when fine-tuning `init_from`).

        Returns:
            Self: Fitted classifier instance.
        """
        y_raw = np.asarray(y)
        if len(y_raw) == 0:
            raise ValueError("Training dataset cannot be empty.")

        y_i64 = y_raw.astype(np.int64, copy=False)
        if int(np.min(y_i64)) >= -128 and int(np.max(y_i64)) <= 127:
            y_arr = y_i64.astype(np.int8)
            self.classes_ = np.unique(y_arr).astype(np.int8)
        else:
            y_arr = y_i64
            self.classes_ = np.unique(y_arr).astype(np.int64)

        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError(f"{self.__class__.__name__} requires at least 2 distinct classes, found {n_classes}.")

        y_indexed = np.searchsorted(self.classes_, y_arr).astype(np.int32)

        if sample_weight is not None:
            sw_arr = np.asarray(sample_weight, dtype=np.float32)
            if len(sw_arr) != len(y_arr):
                raise ValueError(
                    f"Length of sample_weight ({len(sw_arr)}) does not match y ({len(y_arr)})."
                )
            sw_arr = np.maximum(sw_arr, np.float32(0.0))
            mean_sw = float(np.mean(sw_arr))
            if mean_sw > 0.0:
                sw_arr = (sw_arr / np.float32(mean_sw)).astype(np.float32)
        else:
            sw_arr = np.ones(len(y_arr), dtype=np.float32)

        if self.class_weight == "balanced":
            counts = np.bincount(y_indexed, weights=sw_arr, minlength=n_classes).astype(np.float32)
            counts_safe = np.maximum(counts, 1.0)
            weights = float(np.sum(sw_arr)) / (float(n_classes) * counts_safe)
            class_weights = (weights / np.mean(weights)).astype(np.float32)
        else:
            class_weights = np.ones(n_classes, dtype=np.float32)

        # Multi-scale kernel allocation and He-normal initialization
        if isinstance(self.kernel_size, int):
            k_tuple = (self.kernel_size,)
        else:
            k_tuple = tuple(int(k) for k in self.kernel_size)

        n_k = len(k_tuple)
        self.k_sizes_ = np.empty(self.num_filters, dtype=np.int32)
        base_count = self.num_filters // n_k
        remainder = self.num_filters % n_k
        idx = 0
        for i, k in enumerate(k_tuple):
            count = base_count + (1 if i < remainder else 0)
            self.k_sizes_[idx : idx + count] = k
            idx += count

        self.unique_ks_ = np.unique(self.k_sizes_).astype(np.int32)
        self.k_indices_ = np.array(
            [np.where(self.unique_ks_ == k)[0][0] for k in self.k_sizes_],
            dtype=np.int32,
        )
        max_k = int(np.max(self.unique_ks_))

        rng = np.random.default_rng(self.random_state)
        if (
            init_from is not None
            and init_from.is_fitted
            and init_from.vocab_size == self.vocab_size
            and init_from.embed_dim == self.embed_dim
            and init_from.num_filters == self.num_filters
            and np.array_equal(init_from.k_sizes_, self.k_sizes_)
        ):
            self.embedding_ = init_from.embedding_.astype(np.float32, copy=True)
            self.conv_w_ = init_from.conv_w_.astype(np.float32, copy=True)
            self.conv_b_ = init_from.conv_b_.astype(np.float32, copy=True)
        else:
            self.embedding_ = rng.normal(0, 0.1, (self.vocab_size, self.embed_dim)).astype(np.float32)
            self.conv_w_ = np.zeros((self.num_filters, self.embed_dim, max_k), dtype=np.float32)
            for oc in range(self.num_filters):
                k_width = int(self.k_sizes_[oc])
                std = float(np.sqrt(2.0 / (self.embed_dim * k_width)))
                self.conv_w_[oc, :, :k_width] = rng.normal(0.0, std, (self.embed_dim, k_width)).astype(np.float32)
            self.conv_b_ = np.zeros(self.num_filters, dtype=np.float32)

        feat_dim = 2 * self.num_filters + 2 + self.vocab_size
        cnn_feat_dim = 2 * self.num_filters
        self.dense_w_ = np.zeros((feat_dim, n_classes), dtype=np.float32)
        self.dense_w_[:cnn_feat_dim, :] = rng.normal(
            0.0,
            np.sqrt(2.0 / cnn_feat_dim),
            (cnn_feat_dim, n_classes),
        ).astype(np.float32)
        self.dense_b_ = np.zeros(n_classes, dtype=np.float32)

        x_flat, offsets = self._pack_ragged(X)
        if len(offsets) - 1 != len(y_arr):
            raise ValueError(f"Number of samples in X ({len(offsets) - 1}) does not match y ({len(y_arr)}).")

        seed = -1 if self.random_state is None else int(self.random_state)

        # Train via compiled parallel Numba AdamW backprop kernel
        (
            self.embedding_,
            self.conv_w_,
            self.conv_b_,
            self.dense_w_,
            self.dense_b_,
            loss_hist,
        ) = _train_adam(
            x_flat,
            offsets,
            y_indexed,
            class_weights,
            sw_arr,
            self.embedding_,
            self.conv_w_,
            self.conv_b_,
            self.dense_w_,
            self.dense_b_,
            self.k_sizes_,
            int(self.epochs),
            int(self.batch_size),
            float(self.learning_rate),
            float(self.weight_decay),
            seed,
            float(backbone_lr_scale),
        )
        self.loss_history_ = [float(x) for x in loss_hist]
        self.lut_ = _compute_lut(self.embedding_, self.conv_w_, self.k_sizes_)
        if self.class_weight == "balanced" and n_classes > 1 and len(x_flat) > 0:
            n_bg = min(len(offsets) - 1, 256)
            bg_step = max(1, (len(offsets) - 1) // n_bg)
            bg_offsets = np.zeros(n_bg + 1, dtype=np.int64)
            for idx_b in range(n_bg):
                s_i = idx_b * bg_step
                bg_offsets[idx_b + 1] = bg_offsets[idx_b] + (offsets[s_i + 1] - offsets[s_i])
            bg_flat = np.empty(int(bg_offsets[-1]), dtype=x_flat.dtype)
            for idx_b in range(n_bg):
                s_i = idx_b * bg_step
                seg = x_flat[offsets[s_i] : offsets[s_i + 1]].copy()
                rng.shuffle(seg)
                bg_flat[bg_offsets[idx_b] : bg_offsets[idx_b + 1]] = seg
            bg_logits = _logits_batch(
                bg_flat,
                bg_offsets,
                self.lut_,
                self.conv_b_,
                self.dense_w_,
                self.dense_b_,
                self.k_sizes_,
            )
            mean_bg = np.mean(bg_logits, axis=0)
            self.dense_b_ -= np.float32(0.55) * (mean_bg - np.mean(mean_bg))
        return self

    def transform(
        self,
        X: Genes | Sequences | Sequence[Any] | tuple[np.ndarray, np.ndarray],
        *,
        layer: str = "penultimate",
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        """Extract sequence representations from the specified network layer.

        Args:
            X: [`Genes`][serogen.genes.models.Genes], [`Sequences`][serogen.core.seq.seq.Sequences],
                sequence of 1D integer token arrays, or `(flat_tokens, offsets)` tuple.
            layer: Representation layer to extract:
                - `"penultimate"` (default): Full `(n_samples, feature_dim)` dual-pooling + global AA composition.
                - `"conv_max"`: Localized active-site motif max-pooling `(n_samples, num_filters)`.
                - `"conv_avg"`: Domain fold density average-pooling `(n_samples, num_filters)`.
                - `"task_weighted"`: Penultimate features weighted by dense head $L_2$ importance `||W_dense||_2`.
                - `"task_weighted_conv_max"`: Active-site `conv_max` features weighted by `||W_dense[:num_filters]||_2`.
                - `"embed_mean"`: Mean learned amino acid embedding `(n_samples, embed_dim)`.
                - `"logits"`: Pre-softmax class logits `(n_samples, num_classes)`.
                - `"proba"`: Softmax class probabilities `(n_samples, num_classes)`.
                - `"conv_k<N>"` (e.g. `"conv_k9"`, `"conv_k13"`): Max + avg pooled filters for kernel width `<N>`.

        Returns:
            np.ndarray: 2D `float32` feature matrix of shape `(n_samples, layer_dim)`.
        """
        self._check_is_fitted()
        if layer == "logits":
            return self.decision_function(X)
        if layer == "proba":
            return self.predict_proba(X)

        x_flat, offsets = self._pack_ragged(X)
        h = _extract_features_batch(
            x_flat,
            offsets,
            self.lut_,
            self.conv_b_,
            self.k_sizes_,
            self.feature_dim,
        )
        nf = self.num_filters
        if layer in ("penultimate", "full"):
            return h
        if layer == "conv_max":
            return np.ascontiguousarray(h[:, :nf])
        if layer == "conv_avg":
            return np.ascontiguousarray(h[:, nf : 2 * nf])
        if layer == "task_weighted":
            w_norms = np.linalg.norm(self.dense_w_, axis=1).astype(np.float32)
            w_norms = np.maximum(w_norms, np.float32(1e-6))
            return np.ascontiguousarray(h * w_norms[None, :])
        if layer == "task_weighted_conv_max":
            w_norms = np.linalg.norm(self.dense_w_[:nf], axis=1).astype(np.float32)
            w_norms = np.maximum(w_norms, np.float32(1e-6))
            return np.ascontiguousarray(h[:, :nf] * w_norms[None, :])
        if layer == "embed_mean":
            aa_start = 2 * nf + 2
            aa_freqs = h[:, aa_start : aa_start + self.vocab_size] * np.float32(4.0)
            return np.ascontiguousarray(aa_freqs @ self.embedding_)
        if layer.startswith("conv_k"):
            try:
                k_val = int(layer[6:])
            except ValueError as err:
                raise ValueError(f"Invalid kernel width in layer={layer!r}.") from err
            k_mask = np.where(self.k_sizes_ == k_val)[0]
            if len(k_mask) == 0:
                raise ValueError(f"No filters with kernel size {k_val} in model (available: {self.unique_ks_.tolist()}).")
            idx = np.concatenate([k_mask, nf + k_mask])
            return np.ascontiguousarray(h[:, idx])
        raise ValueError(
            f"Unsupported layer {layer!r}. Expected 'penultimate', 'conv_max', 'conv_avg', "
            "'task_weighted', 'task_weighted_conv_max', 'embed_mean', 'logits', 'proba', or 'conv_k<N>'."
        )

    def decision_function(
        self,
        X: Genes | Sequences | Sequence[Any] | tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        """Compute raw pre-softmax class logits `(n_samples, n_classes)` for each sequence in `X`."""
        self._check_is_fitted()
        x_flat, offsets = self._pack_ragged(X)
        return _logits_batch(
            x_flat,
            offsets,
            self.lut_,
            self.conv_b_,
            self.dense_w_,
            self.dense_b_,
            self.k_sizes_,
        )

    def predict_proba(
        self,
        X: Genes | Sequences | Sequence[Any] | tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        """Predict class probabilities `(n_samples, n_classes)` for each sequence in `X`."""
        self._check_is_fitted()
        x_flat, offsets = self._pack_ragged(X)
        return _forward_batch(
            x_flat,
            offsets,
            self.lut_,
            self.conv_b_,
            self.dense_w_,
            self.dense_b_,
            self.k_sizes_,
        )

    def predict_cluster_proba(
        self,
        X: Genes | Sequences | Sequence[Any] | tuple[np.ndarray, np.ndarray],
        cluster_ids: Sequence[int] | np.ndarray,
        *,
        pool: str = "log_proba",
        sample_weight: Sequence[float] | np.ndarray | None = None,
        ignore_non_positive: bool = True,
    ) -> tuple[np.ndarray[tuple[int], np.dtype[np.int64]], np.ndarray[tuple[int, int], np.dtype[np.float32]]]:
        """Aggregate sequence-level predicted probabilities by homology `cluster_id`.

        Args:
            X: Sequence batch input (`Genes`, `Sequences`, token arrays, or `(flat, offsets)`).
            cluster_ids: 1D array-like of integer cluster IDs aligned with `X`.
            pool: Aggregation strategy across cluster members:
                - `"log_proba"` (default): Geometric mean (product-of-experts via mean log-probabilities + softmax).
                - `"mean"`: Arithmetic mean of class probability distributions.
                - `"max"`: Elementwise maximum probability across cluster members, renormalized.
            sample_weight: Optional 1D array-like of per-sequence weights used during pooling.
            ignore_non_positive: If `True` (default), excludes `cluster_id <= 0` (unclustered genes).

        Returns:
            tuple[np.ndarray, np.ndarray]: `(unique_clusters, cluster_probs)` where `unique_clusters`
            is a 1D `int64` array of shape `(n_clusters,)` and `cluster_probs` is a 2D `float32`
            array of shape `(n_clusters, num_classes)`.
        """
        probs = self.predict_proba(X)
        cids = np.asarray(cluster_ids, dtype=np.int64)
        if len(cids) != len(probs):
            raise ValueError(
                f"Length of cluster_ids ({len(cids)}) does not match X ({len(probs)})."
            )
        if sample_weight is not None:
            weights = np.maximum(np.asarray(sample_weight, dtype=np.float32), np.float32(1e-6))
            if len(weights) != len(probs):
                raise ValueError(
                    f"Length of sample_weight ({len(weights)}) does not match X ({len(probs)})."
                )
        else:
            weights = np.ones(len(probs), dtype=np.float32)

        valid_mask = (cids > 0) if ignore_non_positive else np.ones(len(cids), dtype=bool)
        if not np.any(valid_mask):
            return np.empty(0, dtype=np.int64), np.empty((0, self.num_classes), dtype=np.float32)

        uniq_cids, inv = np.unique(cids[valid_mask], return_inverse=True)
        n_clusters = len(uniq_cids)
        n_classes = self.num_classes
        sub_probs = probs[valid_mask]
        sub_w = weights[valid_mask]

        cluster_probs = np.zeros((n_clusters, n_classes), dtype=np.float32)
        if pool == "log_proba":
            log_p = np.log(np.maximum(sub_probs, np.float32(1e-8)))
            w_sum = np.bincount(inv, weights=sub_w, minlength=n_clusters).astype(np.float32)
            w_sum = np.maximum(w_sum, np.float32(1e-8))
            for c in range(n_classes):
                cluster_probs[:, c] = (
                    np.bincount(inv, weights=log_p[:, c] * sub_w, minlength=n_clusters).astype(np.float32)
                    / w_sum
                )
            max_lp = np.max(cluster_probs, axis=1, keepdims=True)
            exp_lp = np.exp(cluster_probs - max_lp)
            cluster_probs = (exp_lp / np.maximum(np.sum(exp_lp, axis=1, keepdims=True), np.float32(1e-12))).astype(
                np.float32
            )
        elif pool == "mean":
            w_sum = np.bincount(inv, weights=sub_w, minlength=n_clusters).astype(np.float32)
            w_sum = np.maximum(w_sum, np.float32(1e-8))
            for c in range(n_classes):
                cluster_probs[:, c] = (
                    np.bincount(inv, weights=sub_probs[:, c] * sub_w, minlength=n_clusters).astype(np.float32)
                    / w_sum
                )
            row_sums = np.maximum(np.sum(cluster_probs, axis=1, keepdims=True), np.float32(1e-12))
            cluster_probs = (cluster_probs / row_sums).astype(np.float32)
        elif pool == "max":
            np.maximum.at(cluster_probs, inv, sub_probs)
            row_sums = np.maximum(np.sum(cluster_probs, axis=1, keepdims=True), np.float32(1e-12))
            cluster_probs = (cluster_probs / row_sums).astype(np.float32)
        else:
            raise ValueError(f"Unsupported pool strategy {pool!r}. Expected 'log_proba', 'mean', or 'max'.")

        return uniq_cids, cluster_probs

    def predict_proba_pooled(
        self,
        X: Genes | Sequences | Sequence[Any] | tuple[np.ndarray, np.ndarray],
        cluster_ids: Sequence[int] | np.ndarray,
        *,
        pool: str = "log_proba",
        sample_weight: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        """Predict per-sequence probabilities `(n_samples, n_classes)` with cluster-consensus pooling for `cluster_id > 0`."""
        probs = self.predict_proba(X).copy()
        cids = np.asarray(cluster_ids, dtype=np.int64)
        if len(cids) != len(probs):
            raise ValueError(
                f"Length of cluster_ids ({len(cids)}) does not match X ({len(probs)})."
            )
        pos_mask = cids > 0
        if not np.any(pos_mask):
            return probs
        uniq_cids, cluster_probs = self.predict_cluster_proba(
            X,
            cids,
            pool=pool,
            sample_weight=sample_weight,
            ignore_non_positive=True,
        )
        idx_map = np.searchsorted(uniq_cids, cids[pos_mask])
        probs[pos_mask] = cluster_probs[idx_map]
        return probs

    def predict_clusters(
        self,
        X: Genes | Sequences | Sequence[Any] | tuple[np.ndarray, np.ndarray],
        cluster_ids: Sequence[int] | np.ndarray,
        *,
        pool: str = "log_proba",
        min_confidence: float = 0.0,
        unknown_label: int | None = None,
        sample_weight: Sequence[float] | np.ndarray | None = None,
        ignore_non_positive: bool = True,
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.int64]],
        np.ndarray[tuple[int], np.dtype[Any]],
        np.ndarray[tuple[int], np.dtype[np.float32]],
    ]:
        """Predict consensus class labels and confidences for each unique `cluster_id`.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: `(unique_clusters, cluster_preds, cluster_confidences)`.
        """
        uniq_cids, cluster_probs = self.predict_cluster_proba(
            X,
            cluster_ids,
            pool=pool,
            sample_weight=sample_weight,
            ignore_non_positive=ignore_non_positive,
        )
        if len(uniq_cids) == 0:
            return (
                uniq_cids,
                np.empty(0, dtype=self.classes_.dtype),
                np.empty(0, dtype=np.float32),
            )
        top_idx = np.argmax(cluster_probs, axis=1)
        max_conf = np.max(cluster_probs, axis=1).astype(np.float32)
        top_classes = self.classes_[top_idx].astype(self.classes_.dtype, copy=True)
        if min_confidence > 0.0:
            unk = (
                (int(GeneRole.UNKNOWN) if self.classes_.dtype == np.int8 else -1)
                if unknown_label is None
                else int(unknown_label)
            )
            top_classes = np.where(max_conf >= min_confidence, top_classes, unk).astype(self.classes_.dtype)
        return uniq_cids, top_classes, max_conf

    def predict(
        self,
        X: Genes | Sequences | Sequence[Any] | tuple[np.ndarray, np.ndarray],
        min_confidence: float = 0.0,
        unknown_label: int | None = None,
    ) -> np.ndarray[tuple[int], np.dtype[Any]]:
        """Predict most likely class label (e.g. `GeneRole` integer index) for each sequence in `X`.

        Args:
            X: [`Genes`][serogen.genes.models.Genes], [`Sequences`][serogen.core.seq.seq.Sequences],
                sequence of 1D integer token arrays, or `(flat_tokens, offsets)` tuple.
            min_confidence: Minimum predicted probability required. If max predicted
                probability is below this threshold, assigns `unknown_label`.
            unknown_label: Fallback label when confidence `< min_confidence` (defaults to
                `GeneRole.UNKNOWN` when `classes_` has `int8` dtype, else `-1`).

        Returns:
            np.ndarray: 1D array of shape `(n_samples,)` of predicted class labels with `self.classes_.dtype`.
        """
        probs = self.predict_proba(X)
        out_dtype = self.classes_.dtype
        if len(probs) == 0:
            return np.empty(0, dtype=out_dtype)
        top_indices = np.argmax(probs, axis=1)
        top_classes = self.classes_[top_indices].astype(out_dtype)
        if min_confidence > 0.0:
            unk = (
                (int(GeneRole.UNKNOWN) if out_dtype == np.int8 else -1)
                if unknown_label is None
                else int(unknown_label)
            )
            max_probs = np.max(probs, axis=1)
            return np.where(max_probs >= min_confidence, top_classes, unk).astype(out_dtype)
        return top_classes

    def score(
        self,
        X: Genes | Sequences | Sequence[Any] | tuple[np.ndarray, np.ndarray],
        y: Sequence[int | GeneRole] | np.ndarray,
    ) -> float:
        """Return classification accuracy on `(X, y)`."""
        preds = self.predict(X)
        y_arr = np.asarray(y, dtype=self.classes_.dtype)
        return float(np.mean(preds == y_arr))

    def top_k_accuracy(
        self,
        X: Genes | Sequences | Sequence[Any] | tuple[np.ndarray, np.ndarray],
        y: Sequence[int | GeneRole] | np.ndarray,
        k: int = 3,
    ) -> float:
        """Compute Top-`k` classification accuracy on `(X, y)`."""
        probs = self.predict_proba(X)
        y_arr = np.asarray(y, dtype=self.classes_.dtype)
        if len(y_arr) == 0:
            return 0.0
        k_eff = min(max(1, int(k)), self.num_classes)
        top_k_idx = np.argsort(-probs, axis=1)[:, :k_eff]
        top_k_classes = self.classes_[top_k_idx]
        hits = np.any(top_k_classes == y_arr[:, None], axis=1)
        return float(np.mean(hits))

    def _extra_save_arrays(self) -> dict[str, np.ndarray]:
        """Hook for subclasses to include additional metadata arrays in `.npz` archives."""
        return {}

    def _load_extra_arrays(self, data: Any) -> None:
        """Hook for subclasses to restore additional metadata arrays from `.npz` archives."""

    def _check_is_fitted(self) -> None:
        """Verify that all required model parameters are fitted."""
        if not self.is_fitted:
            raise RuntimeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' or load pre-trained weights first."
            )
        if getattr(self, "lut_", None) is None:
            self.lut_ = _compute_lut(self.embedding_, self.conv_w_, self.k_sizes_)

    def save(self, path: Path | str) -> None:
        """Save model weights, classes, and hyperparameters to a compressed `.npz` archive."""
        self._check_is_fitted()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        cw_flag = 1 if self.class_weight == "balanced" else 0
        first_k = int(self.k_sizes_[0])
        meta = np.array(
            [
                self.vocab_size,
                self.embed_dim,
                self.num_filters,
                first_k,
                self.batch_size,
                self.epochs,
                cw_flag,
            ],
            dtype=np.int64,
        )

        float_meta = np.array(
            [
                self.learning_rate,
                self.weight_decay,
                -1.0 if self.random_state is None else float(self.random_state),
            ],
            dtype=np.float64,
        )

        save_dict: dict[str, Any] = {
            "meta": meta,
            "float_meta": float_meta,
            "classes": self.classes_,
            "embedding": self.embedding_,
            "conv_w": self.conv_w_,
            "conv_b": self.conv_b_,
            "dense_w": self.dense_w_,
            "dense_b": self.dense_b_,
            "k_sizes": self.k_sizes_,
            "loss_history": np.array(getattr(self, "loss_history_", []), dtype=np.float32),
            **self._extra_save_arrays(),
        }
        np.savez_compressed(p, **save_dict)

    @classmethod
    def load(cls, path: Path | str) -> Self:
        """Load model weights, classes, and hyperparameters from a compressed `.npz` archive."""
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".npz")

        with np.load(p) as data:
            meta = data["meta"]
            float_meta = data["float_meta"]
            rs = None if float_meta[2] < 0 else int(float_meta[2])
            cw = "balanced" if (len(meta) > 6 and meta[6] == 1) else None

            num_filters = int(meta[2])
            if "k_sizes" in data:
                k_sizes = data["k_sizes"].astype(np.int32)
                ordered_ks = tuple(dict.fromkeys(int(x) for x in k_sizes))
                ks_param: int | tuple[int, ...] = ordered_ks[0] if len(ordered_ks) == 1 else ordered_ks
            else:
                ks_param = int(meta[3])
                k_sizes = np.full(num_filters, ks_param, dtype=np.int32)

            inst = cls(
                vocab_size=int(meta[0]),
                embed_dim=int(meta[1]),
                num_filters=num_filters,
                kernel_size=ks_param,
                batch_size=int(meta[4]),
                epochs=int(meta[5]),
                learning_rate=float(float_meta[0]),
                weight_decay=float(float_meta[1]),
                random_state=rs,
                class_weight=cw,
            )
            raw_classes = data["classes"]
            if len(raw_classes) > 0 and int(np.min(raw_classes)) >= -128 and int(np.max(raw_classes)) <= 127 and raw_classes.dtype == np.int8:
                inst.classes_ = raw_classes.astype(np.int8)
            else:
                inst.classes_ = raw_classes.astype(np.int64)
            inst.embedding_ = data["embedding"].astype(np.float32)
            inst.conv_w_ = data["conv_w"].astype(np.float32)
            inst.conv_b_ = data["conv_b"].astype(np.float32)
            inst.dense_w_ = data["dense_w"].astype(np.float32)
            inst.dense_b_ = data["dense_b"].astype(np.float32)
            inst.k_sizes_ = k_sizes
            inst.unique_ks_ = np.unique(k_sizes).astype(np.int32)
            inst.k_indices_ = np.array(
                [np.where(inst.unique_ks_ == k)[0][0] for k in k_sizes],
                dtype=np.int32,
            )
            inst.loss_history_ = [float(x) for x in data["loss_history"].tolist()]
            inst.lut_ = _compute_lut(inst.embedding_, inst.conv_w_, inst.k_sizes_)
            inst._load_extra_arrays(data)
            return inst

    @classmethod
    @cache
    def pretrained(cls) -> Self:
        """Load the packaged pre-trained `GeneRoleClassifier` from `serogen.data/model.npz`."""
        resource = files("serogen.data").joinpath("model.npz")
        with as_file(resource) as model_path:
            return cls.load(model_path)


# ----------------------------------------------------------------------------------------------------------------------
# High-Performance Numba Kernels
# ----------------------------------------------------------------------------------------------------------------------
@njit(cache=True, nogil=True, fastmath=True)
def _compute_lut(embedding: np.ndarray, conv_w: np.ndarray, k_sizes: np.ndarray) -> np.ndarray:
    """Precompute token-position-channel lookup table: `(vocab_size, max_k, out_channels)`."""
    vocab_size, _ = embedding.shape
    out_channels, in_channels, max_k = conv_w.shape
    lut = np.zeros((vocab_size, max_k, out_channels), dtype=np.float32)
    for oc in range(out_channels):
        k_width = k_sizes[oc]
        for k in range(k_width):
            for t in range(vocab_size):
                s = np.float32(0.0)
                for ic in range(in_channels):
                    s += embedding[t, ic] * conv_w[oc, ic, k]
                lut[t, k, oc] = s
    return lut


@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def _extract_features_batch(
    sequences_flat: np.ndarray,
    offsets: np.ndarray,
    lut: np.ndarray,  # (vocab_size, max_k, out_channels)
    conv_b: np.ndarray,  # (out_channels,)
    k_sizes: np.ndarray,  # (out_channels,)
    feat_dim: int,
) -> np.ndarray:
    """Extract penultimate 1D-CNN dual-pooling + global AA composition features `(n_samples, feat_dim)`."""
    n_samples = len(offsets) - 1
    vocab_size, max_k, out_channels = lut.shape
    features_out = np.zeros((n_samples, feat_dim), dtype=np.float32)

    for i in prange(n_samples):
        start = int(offsets[i])
        end = int(offsets[i + 1])
        seq_len = end - start

        max_val = np.zeros(out_channels, dtype=np.float32)
        sum_val = np.zeros(out_channels, dtype=np.float32)
        val = np.empty(out_channels, dtype=np.float32)
        tok_counts = np.zeros(vocab_size, dtype=np.float32)

        for pos in range(seq_len):
            rem = seq_len - pos
            max_valid_k = max_k if rem >= max_k else rem

            t0 = int(sequences_flat[start + pos])
            if t0 < 0 or t0 >= vocab_size:
                t0 = vocab_size - 1
            tok_counts[t0] += np.float32(1.0)

            for oc in range(out_channels):
                val[oc] = conv_b[oc]

            for k in range(max_valid_k):
                tok = int(sequences_flat[start + pos + k])
                if tok < 0 or tok >= vocab_size:
                    tok = vocab_size - 1
                for oc in range(out_channels):
                    val[oc] += lut[tok, k, oc]

            for oc in range(out_channels):
                if k_sizes[oc] <= rem:
                    v = val[oc]
                    if v > np.float32(0.0):
                        sum_val[oc] += v
                        if v > max_val[oc]:
                            max_val[oc] = v

        len_safe = max(float(seq_len), 1.0)
        features_out[i, 2 * out_channels] = np.float32(np.log(len_safe) / 7.0)
        if feat_dim >= 2 * out_channels + 2 + vocab_size:
            features_out[i, 2 * out_channels + 1] = np.float32(len_safe / 1000.0)
            inv_len = np.float32(0.25 / len_safe)
            for t in range(vocab_size):
                features_out[i, 2 * out_channels + 2 + t] = tok_counts[t] * inv_len

        for oc in range(out_channels):
            k_width = int(k_sizes[oc])
            if seq_len >= k_width:
                out_len = seq_len - k_width + 1
                features_out[i, oc] = max_val[oc]
                features_out[i, out_channels + oc] = sum_val[oc] / np.float32(out_len)

    return features_out


@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def _logits_batch(
    sequences_flat: np.ndarray,
    offsets: np.ndarray,
    lut: np.ndarray,  # (vocab_size, max_k, out_channels)
    conv_b: np.ndarray,  # (out_channels,)
    dense_w: np.ndarray,  # (feat_dim, n_classes)
    dense_b: np.ndarray,  # (n_classes,)
    k_sizes: np.ndarray,  # (out_channels,)
) -> np.ndarray:
    """Compute pre-softmax logits in parallel using contiguous stride-1 SIMD LUT accumulation."""
    n_samples = len(offsets) - 1
    vocab_size, max_k, out_channels = lut.shape
    feat_dim = dense_w.shape[0]
    n_classes = len(dense_b)
    logits_out = np.empty((n_samples, n_classes), dtype=np.float32)

    for i in prange(n_samples):
        start = int(offsets[i])
        end = int(offsets[i + 1])
        seq_len = end - start

        max_val = np.zeros(out_channels, dtype=np.float32)
        sum_val = np.zeros(out_channels, dtype=np.float32)
        val = np.empty(out_channels, dtype=np.float32)
        tok_counts = np.zeros(vocab_size, dtype=np.float32)

        for pos in range(seq_len):
            rem = seq_len - pos
            max_valid_k = max_k if rem >= max_k else rem

            t0 = int(sequences_flat[start + pos])
            if t0 < 0 or t0 >= vocab_size:
                t0 = vocab_size - 1
            tok_counts[t0] += np.float32(1.0)

            for oc in range(out_channels):
                val[oc] = conv_b[oc]

            # Inner loop over `oc` is contiguous stride-1 in `lut[tok, k, oc]` -> SIMD FMA
            for k in range(max_valid_k):
                tok = int(sequences_flat[start + pos + k])
                if tok < 0 or tok >= vocab_size:
                    tok = vocab_size - 1
                for oc in range(out_channels):
                    val[oc] += lut[tok, k, oc]

            # ReLU applied before both max_pool and avg_pool
            for oc in range(out_channels):
                if k_sizes[oc] <= rem:
                    v = val[oc]
                    if v > np.float32(0.0):
                        sum_val[oc] += v
                        if v > max_val[oc]:
                            max_val[oc] = v

        h = np.zeros(feat_dim, dtype=np.float32)
        len_safe = max(float(seq_len), 1.0)
        h[2 * out_channels] = np.float32(np.log(len_safe) / 7.0)
        if feat_dim >= 2 * out_channels + 2 + vocab_size:
            h[2 * out_channels + 1] = np.float32(len_safe / 1000.0)
            inv_len = np.float32(0.25 / len_safe)
            for t in range(vocab_size):
                h[2 * out_channels + 2 + t] = tok_counts[t] * inv_len

        for oc in range(out_channels):
            k_width = int(k_sizes[oc])
            if seq_len >= k_width:
                out_len = seq_len - k_width + 1
                h[oc] = max_val[oc]
                h[out_channels + oc] = sum_val[oc] / np.float32(out_len)

        for c in range(n_classes):
            s = dense_b[c]
            for j in range(feat_dim):
                s += h[j] * dense_w[j, c]
            logits_out[i, c] = s

    return logits_out


@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def _forward_batch(
    sequences_flat: np.ndarray,
    offsets: np.ndarray,
    lut: np.ndarray,
    conv_b: np.ndarray,
    dense_w: np.ndarray,
    dense_b: np.ndarray,
    k_sizes: np.ndarray,
) -> np.ndarray:
    """Parallel forward inference kernel returning numerically stable softmax probabilities."""
    probs = _logits_batch(sequences_flat, offsets, lut, conv_b, dense_w, dense_b, k_sizes)
    n_samples, n_classes = probs.shape

    for i in prange(n_samples):
        max_logit = np.float32(-1e30)
        for c in range(n_classes):
            if probs[i, c] > max_logit:
                max_logit = probs[i, c]

        exp_sum = np.float32(0.0)
        for c in range(n_classes):
            e = np.exp(probs[i, c] - max_logit)
            probs[i, c] = e
            exp_sum += e

        inv_sum = np.float32(1.0) / exp_sum
        for c in range(n_classes):
            probs[i, c] *= inv_sum

    return probs


@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def _train_adam(
    sequences_flat: np.ndarray,
    offsets: np.ndarray,
    y_indexed: np.ndarray,
    class_weights: np.ndarray,
    sample_weights: np.ndarray,
    embedding: np.ndarray,
    conv_w: np.ndarray,
    conv_b: np.ndarray,
    dense_w: np.ndarray,
    dense_b: np.ndarray,
    k_sizes: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    backbone_lr_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parallel mini-batch AdamW training kernel with cosine annealing LR and global AA composition."""
    if seed >= 0:
        np.random.seed(seed)

    n_samples = len(offsets) - 1
    out_channels, in_channels, max_k = conv_w.shape
    feat_dim = dense_w.shape[0]
    n_classes = len(dense_b)
    vocab_size, embed_dim = embedding.shape

    # AdamW first & second moment buffers
    m_emb = np.zeros_like(embedding)
    v_emb = np.zeros_like(embedding)
    m_cw = np.zeros_like(conv_w)
    v_cw = np.zeros_like(conv_w)
    m_cb = np.zeros_like(conv_b)
    v_cb = np.zeros_like(conv_b)
    m_dw = np.zeros_like(dense_w)
    v_dw = np.zeros_like(dense_w)
    m_db = np.zeros_like(dense_b)
    v_db = np.zeros_like(dense_b)

    # Per-sample mini-batch gradient buffers for lock-free `prange` parallelism
    eff_bs = max(1, min(batch_size, n_samples))
    grad_emb_batch = np.zeros((eff_bs, vocab_size, embed_dim), dtype=np.float32)
    grad_cw_batch = np.zeros((eff_bs, out_channels, in_channels, max_k), dtype=np.float32)
    grad_cb_batch = np.zeros((eff_bs, out_channels), dtype=np.float32)
    grad_dw_batch = np.zeros((eff_bs, feat_dim, n_classes), dtype=np.float32)
    grad_db_batch = np.zeros((eff_bs, n_classes), dtype=np.float32)
    sample_losses = np.zeros(eff_bs, dtype=np.float32)

    loss_history = np.empty(epochs, dtype=np.float32)
    step = 0
    lut = _compute_lut(embedding, conv_w, k_sizes)

    indices = np.arange(n_samples, dtype=np.int32)

    for epoch in range(epochs):
        epoch_loss = 0.0
        np.random.shuffle(indices)

        # Cosine annealing learning rate schedule (decays from learning_rate down to 5% floor)
        cos_decay = 0.5 * (1.0 + np.cos(np.pi * float(epoch) / float(max(epochs, 1))))
        lr_epoch = np.float32(learning_rate * (0.05 + 0.95 * cos_decay))

        for b_start in range(0, n_samples, eff_bs):
            b_end = min(b_start + eff_bs, n_samples)
            curr_bs = b_end - b_start

            # Parallel forward + backward over all samples in the mini-batch
            for b in prange(curr_bs):
                grad_emb_batch[b].fill(0.0)
                grad_cw_batch[b].fill(0.0)
                grad_cb_batch[b].fill(0.0)
                grad_dw_batch[b].fill(0.0)
                grad_db_batch[b].fill(0.0)

                i = indices[b_start + b]
                start = int(offsets[i])
                end = int(offsets[i + 1])
                seq_len = end - start

                h = np.zeros(feat_dim, dtype=np.float32)
                len_safe = max(float(seq_len), 1.0)
                h[2 * out_channels] = np.float32(np.log(len_safe) / 7.0)

                max_val = np.zeros(out_channels, dtype=np.float32)
                sum_val = np.zeros(out_channels, dtype=np.float32)
                argmax_pos = np.full(out_channels, -1, dtype=np.int32)
                val = np.empty(out_channels, dtype=np.float32)
                active_mask = np.zeros((seq_len, out_channels), dtype=np.uint8)
                tok_counts = np.zeros(vocab_size, dtype=np.float32)

                for pos in range(seq_len):
                    rem = seq_len - pos
                    max_valid_k = max_k if rem >= max_k else rem

                    t0 = int(sequences_flat[start + pos])
                    if t0 < 0 or t0 >= vocab_size:
                        t0 = vocab_size - 1
                    tok_counts[t0] += np.float32(1.0)

                    for oc in range(out_channels):
                        val[oc] = conv_b[oc]

                    for k in range(max_valid_k):
                        tok = int(sequences_flat[start + pos + k])
                        if tok < 0 or tok >= vocab_size:
                            tok = vocab_size - 1
                        for oc in range(out_channels):
                            val[oc] += lut[tok, k, oc]

                    for oc in range(out_channels):
                        if k_sizes[oc] <= rem:
                            v = val[oc]
                            if v > np.float32(0.0):
                                active_mask[pos, oc] = 1
                                sum_val[oc] += v
                                if v > max_val[oc]:
                                    max_val[oc] = v
                                    argmax_pos[oc] = pos

                if feat_dim >= 2 * out_channels + 2 + vocab_size:
                    h[2 * out_channels + 1] = np.float32(len_safe / 1000.0)
                    inv_len = np.float32(0.25 / len_safe)
                    for t in range(vocab_size):
                        h[2 * out_channels + 2 + t] = tok_counts[t] * inv_len

                for oc in range(out_channels):
                    k_width = int(k_sizes[oc])
                    if seq_len >= k_width:
                        out_len = seq_len - k_width + 1
                        h[oc] = max_val[oc]
                        h[out_channels + oc] = sum_val[oc] / np.float32(out_len)

                logits = np.empty(n_classes, dtype=np.float32)
                max_logit = np.float32(-1e30)
                for c in range(n_classes):
                    s = dense_b[c]
                    for j in range(feat_dim):
                        s += h[j] * dense_w[j, c]
                    logits[c] = s
                    if s > max_logit:
                        max_logit = s

                probs = np.empty(n_classes, dtype=np.float32)
                exp_sum = np.float32(0.0)
                for c in range(n_classes):
                    e = np.exp(logits[c] - max_logit)
                    probs[c] = e
                    exp_sum += e

                inv_sum = np.float32(1.0) / exp_sum
                for c in range(n_classes):
                    probs[c] *= inv_sum

                target = int(y_indexed[i])
                w = class_weights[target] * sample_weights[i]
                sample_losses[b] = -w * np.log(probs[target] + np.float32(1e-10))

                # Gradient w.r.t. dense layer & feature vector h
                d_h = np.zeros(feat_dim, dtype=np.float32)
                for c in range(n_classes):
                    d_logit = w * (probs[c] - (np.float32(1.0) if c == target else np.float32(0.0)))
                    grad_db_batch[b, c] += d_logit
                    for j in range(feat_dim):
                        grad_dw_batch[b, j, c] += h[j] * d_logit
                        d_h[j] += dense_w[j, c] * d_logit

                # Backprop through ReLU avg_pool and ReLU max_pool
                for oc in range(out_channels):
                    k_width = int(k_sizes[oc])
                    if seq_len < k_width:
                        continue
                    out_len = seq_len - k_width + 1
                    d_avg_step = d_h[out_channels + oc] / np.float32(out_len)
                    d_max_oc = d_h[oc]
                    max_p = int(argmax_pos[oc])

                    for pos in range(out_len):
                        d_val = np.float32(0.0)
                        if active_mask[pos, oc] != 0:
                            d_val += d_avg_step
                        if pos == max_p and h[oc] > np.float32(0.0):
                            d_val += d_max_oc

                        if d_val != np.float32(0.0):
                            grad_cb_batch[b, oc] += d_val
                            for k in range(k_width):
                                tok = int(sequences_flat[start + pos + k])
                                if tok < 0 or tok >= vocab_size:
                                    tok = vocab_size - 1
                                for ic in range(in_channels):
                                    grad_cw_batch[b, oc, ic, k] += d_val * embedding[tok, ic]
                                    grad_emb_batch[b, tok, ic] += d_val * conv_w[oc, ic, k]

            # Accumulate mini-batch loss and apply AdamW step
            for b in range(curr_bs):
                epoch_loss += float(sample_losses[b])

            step += 1
            inv_bc = np.float32(1.0 / curr_bs)
            beta1 = np.float32(0.9)
            beta2 = np.float32(0.999)
            eps = np.float32(1e-8)
            bc1 = np.float32(1.0 - (0.9**step))
            bc2 = np.float32(1.0 - (0.999**step))
            step_size = lr_epoch * np.sqrt(bc2) / bc1
            wd_factor = np.float32(1.0) - lr_epoch * np.float32(weight_decay)
            bb_lr_scale = np.float32(backbone_lr_scale)
            bb_step_size = step_size * bb_lr_scale
            bb_wd_factor = np.float32(1.0) - lr_epoch * bb_lr_scale * np.float32(weight_decay)

            # Update dense_b
            for c in range(n_classes):
                g_sum = np.float32(0.0)
                for b in range(curr_bs):
                    g_sum += grad_db_batch[b, c]
                g = g_sum * inv_bc
                m_db[c] = beta1 * m_db[c] + (np.float32(1.0) - beta1) * g
                v_db[c] = beta2 * v_db[c] + (np.float32(1.0) - beta2) * g * g
                dense_b[c] -= step_size * m_db[c] / (np.sqrt(v_db[c]) + eps)

            # Update dense_w (decoupled AdamW: conv_max uses 1.0x LR + 1.0x WD; conv_avg uses 0.65x LR + 2.0x WD;
            # global length/AA features use 0.15x LR + 5x WD to prioritize localized active-site motifs)
            avg_wd_factor = np.float32(1.0) - lr_epoch * np.float32(weight_decay * 2.0)
            avg_step_size = step_size * np.float32(0.65)
            global_wd = weight_decay * 5.0 if weight_decay * 5.0 > 2e-3 else 2e-3
            global_wd_factor = np.float32(1.0) - lr_epoch * np.float32(global_wd)
            global_step_size = step_size * np.float32(0.15)
            for j in range(feat_dim):
                if j < out_channels:
                    eff_wd = wd_factor
                    eff_step = step_size
                elif j < 2 * out_channels:
                    eff_wd = avg_wd_factor
                    eff_step = avg_step_size
                else:
                    eff_wd = global_wd_factor
                    eff_step = global_step_size
                for c in range(n_classes):
                    g_sum = np.float32(0.0)
                    for b in range(curr_bs):
                        g_sum += grad_dw_batch[b, j, c]
                    g = g_sum * inv_bc
                    m_dw[j, c] = beta1 * m_dw[j, c] + (np.float32(1.0) - beta1) * g
                    v_dw[j, c] = beta2 * v_dw[j, c] + (np.float32(1.0) - beta2) * g * g
                    dense_w[j, c] = dense_w[j, c] * eff_wd - eff_step * m_dw[j, c] / (np.sqrt(v_dw[j, c]) + eps)

            if bb_lr_scale > np.float32(0.0):
                # Update conv_b
                for oc in range(out_channels):
                    g_sum = np.float32(0.0)
                    for b in range(curr_bs):
                        g_sum += grad_cb_batch[b, oc]
                    g = g_sum * inv_bc
                    m_cb[oc] = beta1 * m_cb[oc] + (np.float32(1.0) - beta1) * g
                    v_cb[oc] = beta2 * v_cb[oc] + (np.float32(1.0) - beta2) * g * g
                    conv_b[oc] -= bb_step_size * m_cb[oc] / (np.sqrt(v_cb[oc]) + eps)

                # Update conv_w (decoupled AdamW)
                for oc in range(out_channels):
                    k_width = int(k_sizes[oc])
                    for ic in range(in_channels):
                        for k in range(k_width):
                            g_sum = np.float32(0.0)
                            for b in range(curr_bs):
                                g_sum += grad_cw_batch[b, oc, ic, k]
                            g = g_sum * inv_bc
                            m_cw[oc, ic, k] = beta1 * m_cw[oc, ic, k] + (np.float32(1.0) - beta1) * g
                            v_cw[oc, ic, k] = beta2 * v_cw[oc, ic, k] + (np.float32(1.0) - beta2) * g * g
                            conv_w[oc, ic, k] = conv_w[oc, ic, k] * bb_wd_factor - bb_step_size * m_cw[oc, ic, k] / (
                                np.sqrt(v_cw[oc, ic, k]) + eps
                            )

                # Update embedding (sparse-aware decoupled AdamW: only decay active tokens)
                for tok in range(vocab_size):
                    tok_active = False
                    for ic in range(embed_dim):
                        g_sum = np.float32(0.0)
                        for b in range(curr_bs):
                            g_sum += grad_emb_batch[b, tok, ic]
                        if g_sum != np.float32(0.0):
                            tok_active = True
                        g = g_sum * inv_bc
                        m_emb[tok, ic] = beta1 * m_emb[tok, ic] + (np.float32(1.0) - beta1) * g
                        v_emb[tok, ic] = beta2 * v_emb[tok, ic] + (np.float32(1.0) - beta2) * g * g
                    for ic in range(embed_dim):
                        if tok_active:
                            embedding[tok, ic] *= bb_wd_factor
                        embedding[tok, ic] -= bb_step_size * m_emb[tok, ic] / (np.sqrt(v_emb[tok, ic]) + eps)

                lut = _compute_lut(embedding, conv_w, k_sizes)

        loss_history[epoch] = epoch_loss / n_samples

    return embedding, conv_w, conv_b, dense_w, dense_b, loss_history
