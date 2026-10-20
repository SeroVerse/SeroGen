r"""Numba-accelerated Relational Message-Passing Neural Network (R-MPNN) primitives.

Provides domain-agnostic CSR relational graph builders, compiled `@njit(parallel=True, fastmath=True)`
message-passing forward/backward kernels, factorized bilinear directed edge/slot scorers,
log-space Sinkhorn bipartite matching, and the [`RelationalMPNN`][serogen.core.graph.gnn.RelationalMPNN]
model class.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from numba import njit, prange


# ----------------------------------------------------------------------------------------------------------------------
# CSR Relational Graph Container & Builders
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class RelationalCSR:
    r"""Compressed Sparse Row (CSR) incoming-edge representation of a multi-relational directed graph.

    For each target vertex $v \in [0, N-1]$, the slice `k in range(indptr[v], indptr[v + 1])`
    enumerates incoming directed edges $(u_k \xrightarrow{r_k} v)$ with source vertex
    `u_k = src_indices[k]`, relation ID `r_k = rel_types[k]`, and normalized aggregation weight
    `edge_weights[k]`.

    Attributes:
        n_nodes: Total number of vertices $N$ (including any appended virtual pool vertices).
        n_relations: Total number of distinct relation types $R$.
        indptr: 1D `int32` array of shape `(N + 1,)` indexing incoming edges per target vertex.
        src_indices: 1D `int32` array of shape `(E,)` of source vertex indices.
        rel_types: 1D `int32` array of shape `(E,)` of relation type IDs in `[0, R - 1]`.
        edge_weights: 1D `float32` array of shape `(E,)` of per-edge aggregation weights.
        virtual_nodes: 1D `int32` array of virtual global pool vertex indices (empty if none).
    """

    n_nodes: int
    n_relations: int
    indptr: np.ndarray[tuple[int], np.dtype[np.int32]]
    src_indices: np.ndarray[tuple[int], np.dtype[np.int32]]
    rel_types: np.ndarray[tuple[int], np.dtype[np.int32]]
    edge_weights: np.ndarray[tuple[int], np.dtype[np.float32]]
    virtual_nodes: np.ndarray[tuple[int], np.dtype[np.int32]]


def build_relational_csr(
    n_nodes: int,
    sources: Sequence[int] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
    relations: Sequence[int] | np.ndarray | None = None,
    *,
    n_relations: int | None = None,
    add_self_loops: bool = True,
    normalize_by_degree: bool = True,
    virtual_nodes: np.ndarray | None = None,
) -> RelationalCSR:
    r"""Construct a [`RelationalCSR`][serogen.core.graph.gnn.RelationalCSR] from flat directed edge arrays.

    Args:
        n_nodes: Total number of vertices $N$.
        sources: 1D array-like of source vertex indices `u`.
        targets: 1D array-like of target vertex indices `v`.
        relations: Optional 1D array-like of integer relation IDs `r >= 0`. If `None`, all edges
            receive relation `1` (when `add_self_loops=True`, self-loops receive relation `0`).
        n_relations: Total number of relation types $R$. Inferred if `None`.
        add_self_loops: If `True` (default), prepends a self-loop `(v -> v, r=0)` for every vertex.
        normalize_by_degree: If `True` (default), normalizes incoming weights per `(target, relation)`
            group so each relation's incoming messages average cleanly.
        virtual_nodes: Optional 1D `int32` array of virtual global pool vertex indices.

    Returns:
        RelationalCSR: Incoming-edge CSR structure ready for Numba MPNN kernels.
    """
    src_arr = np.asarray(sources, dtype=np.int32)
    dst_arr = np.asarray(targets, dtype=np.int32)
    if len(src_arr) != len(dst_arr):
        raise ValueError(f"sources length ({len(src_arr)}) != targets length ({len(dst_arr)}).")

    if relations is None:
        rel_arr = np.full(len(src_arr), 1 if add_self_loops else 0, dtype=np.int32)
    else:
        rel_arr = np.asarray(relations, dtype=np.int32)
        if len(rel_arr) != len(src_arr):
            raise ValueError(f"relations length ({len(rel_arr)}) != sources length ({len(src_arr)}).")

    if add_self_loops and n_nodes > 0:
        self_idx = np.arange(n_nodes, dtype=np.int32)
        self_rel = np.zeros(n_nodes, dtype=np.int32)
        src_arr = np.concatenate([self_idx, src_arr])
        dst_arr = np.concatenate([self_idx, dst_arr])
        rel_arr = np.concatenate([self_rel, rel_arr])

    if n_relations is None:
        n_rel = int(np.max(rel_arr)) + 1 if len(rel_arr) > 0 else 1
    else:
        n_rel = max(int(n_relations), 1)

    v_nodes = (
        np.asarray(virtual_nodes, dtype=np.int32)
        if virtual_nodes is not None
        else np.empty(0, dtype=np.int32)
    )

    if n_nodes == 0 or len(src_arr) == 0:
        return RelationalCSR(
            n_nodes=n_nodes,
            n_relations=n_rel,
            indptr=np.zeros(n_nodes + 1, dtype=np.int32),
            src_indices=np.empty(0, dtype=np.int32),
            rel_types=np.empty(0, dtype=np.int32),
            edge_weights=np.empty(0, dtype=np.float32),
            virtual_nodes=v_nodes,
        )

    order = np.lexsort((src_arr, rel_arr, dst_arr))
    src_sorted = np.ascontiguousarray(src_arr[order], dtype=np.int32)
    dst_sorted = np.ascontiguousarray(dst_arr[order], dtype=np.int32)
    rel_sorted = np.ascontiguousarray(rel_arr[order], dtype=np.int32)

    counts = np.bincount(dst_sorted, minlength=n_nodes).astype(np.int32)
    indptr = np.zeros(n_nodes + 1, dtype=np.int32)
    np.cumsum(counts, out=indptr[1:])

    weights = np.ones(len(src_sorted), dtype=np.float32)
    if normalize_by_degree:
        key = dst_sorted.astype(np.int64) * n_rel + rel_sorted.astype(np.int64)
        uniq_keys, inv, key_counts = np.unique(key, return_inverse=True, return_counts=True)
        _ = uniq_keys
        weights = (1.0 / np.maximum(key_counts[inv], 1)).astype(np.float32)

    return RelationalCSR(
        n_nodes=n_nodes,
        n_relations=n_rel,
        indptr=indptr,
        src_indices=src_sorted,
        rel_types=rel_sorted,
        edge_weights=np.ascontiguousarray(weights, dtype=np.float32),
        virtual_nodes=v_nodes,
    )


def build_multihop_csr(
    node_offsets: Sequence[int] | np.ndarray,
    edge_sources: Sequence[int] | np.ndarray,
    edge_targets: Sequence[int] | np.ndarray,
    edge_orientations: Sequence[int] | np.ndarray | None = None,
    *,
    include_2hop: bool = True,
    add_virtual_pool: bool = True,
) -> RelationalCSR:
    r"""Build a multi-hop relational CSR graph for a single graph or a ragged batch of graphs.

    Constructs up to 8 directed relation types:
        - `0`: Self-loop (`v -> v`)
        - `1`: 1-hop forward (`u -> v`, codirectional)
        - `2`: 1-hop reverse (`v -> u`, codirectional)
        - `3`: 1-hop convergent/divergent (`u <-> v` when `edge_orientations != 0`)
        - `4`: 2-hop forward (`u -> w -> v`)
        - `5`: 2-hop reverse (`v -> w -> u`)
        - `6`: Real vertex to per-graph virtual global pool (`u -> pool_g`)
        - `7`: Per-graph virtual global pool to real vertex (`pool_g -> u`)

    Args:
        node_offsets: 1D array of graph node offsets `[0, n_0, n_0 + n_1, ..., N_real]`.
        edge_sources: 1D array of 0-based global source vertex indices in `[0, N_real - 1]`.
        edge_targets: 1D array of 0-based global target vertex indices in `[0, N_real - 1]`.
        edge_orientations: Optional 1D array of orientation codes (`0` = codirectional, `>0` = conv/div).
        include_2hop: If `True` (default), adds 2-hop forward (`r=4`) and reverse (`r=5`) edges.
        add_virtual_pool: If `True` (default), appends 1 virtual global pool node per graph at
            indices `[N_real, ..., N_real + G - 1]` connected via relations `6` and `7`.

    Returns:
        RelationalCSR: Multi-hop relational CSR with `n_relations = 8`.
    """
    offsets = np.asarray(node_offsets, dtype=np.int32)
    n_graphs = max(len(offsets) - 1, 0)
    n_real = int(offsets[-1]) if len(offsets) > 0 else 0
    e_src = np.asarray(edge_sources, dtype=np.int32)
    e_dst = np.asarray(edge_targets, dtype=np.int32)
    e_ori = (
        np.asarray(edge_orientations, dtype=np.int32)
        if edge_orientations is not None
        else np.zeros(len(e_src), dtype=np.int32)
    )

    src_list: list[np.ndarray] = []
    dst_list: list[np.ndarray] = []
    rel_list: list[np.ndarray] = []

    if len(e_src) > 0:
        codir = e_ori == 0
        if np.any(codir):
            u_c = e_src[codir]
            v_c = e_dst[codir]
            src_list.extend([u_c, v_c])
            dst_list.extend([v_c, u_c])
            rel_list.extend([np.full(len(u_c), 1, dtype=np.int32), np.full(len(u_c), 2, dtype=np.int32)])
        non_codir = ~codir
        if np.any(non_codir):
            u_nc = e_src[non_codir]
            v_nc = e_dst[non_codir]
            src_list.extend([u_nc, v_nc])
            dst_list.extend([v_nc, u_nc])
            rel_list.extend([np.full(len(u_nc), 3, dtype=np.int32), np.full(len(u_nc), 3, dtype=np.int32)])

        if include_2hop and n_real > 0:
            # Find 2-hop paths u -> w -> v
            adj_out: list[list[int]] = [[] for _ in range(n_real)]
            for u_i, v_i in zip(e_src, e_dst, strict=False):
                adj_out[int(u_i)].append(int(v_i))
            hop2_u: list[int] = []
            hop2_v: list[int] = []
            for u_i in range(n_real):
                for w_i in adj_out[u_i]:
                    for v_i in adj_out[w_i]:
                        if v_i != u_i:
                            hop2_u.append(u_i)
                            hop2_v.append(v_i)
            if hop2_u:
                h2u = np.array(hop2_u, dtype=np.int32)
                h2v = np.array(hop2_v, dtype=np.int32)
                src_list.extend([h2u, h2v])
                dst_list.extend([h2v, h2u])
                rel_list.extend([np.full(len(h2u), 4, dtype=np.int32), np.full(len(h2u), 5, dtype=np.int32)])

    if add_virtual_pool and n_graphs > 0:
        virtual_nodes = np.arange(n_real, n_real + n_graphs, dtype=np.int32)
        total_nodes = n_real + n_graphs
        vp_src: list[np.ndarray] = []
        vp_dst: list[np.ndarray] = []
        vp_rel: list[np.ndarray] = []
        for g in range(n_graphs):
            s = int(offsets[g])
            e = int(offsets[g + 1])
            if e > s:
                nodes_g = np.arange(s, e, dtype=np.int32)
                v_node = np.full(e - s, n_real + g, dtype=np.int32)
                vp_src.extend([nodes_g, v_node])
                vp_dst.extend([v_node, nodes_g])
                vp_rel.extend([np.full(e - s, 6, dtype=np.int32), np.full(e - s, 7, dtype=np.int32)])
        if vp_src:
            src_list.extend(vp_src)
            dst_list.extend(vp_dst)
            rel_list.extend(vp_rel)
    else:
        virtual_nodes = np.empty(0, dtype=np.int32)
        total_nodes = n_real

    all_src = np.concatenate(src_list) if src_list else np.empty(0, dtype=np.int32)
    all_dst = np.concatenate(dst_list) if dst_list else np.empty(0, dtype=np.int32)
    all_rel = np.concatenate(rel_list) if rel_list else np.empty(0, dtype=np.int32)

    return build_relational_csr(
        total_nodes,
        all_src,
        all_dst,
        all_rel,
        n_relations=8,
        add_self_loops=True,
        normalize_by_degree=True,
        virtual_nodes=virtual_nodes,
    )


# ----------------------------------------------------------------------------------------------------------------------
# Compiled Numba Kernels: R-MPNN Forward, Backward, Bilinear Edge Scorer, Sinkhorn, AdamW
# ----------------------------------------------------------------------------------------------------------------------
@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def mpnn_layer_forward(
    h_in: np.ndarray,  # (N, D)
    indptr: np.ndarray,  # (N + 1,)
    src_indices: np.ndarray,  # (E,)
    rel_types: np.ndarray,  # (E,)
    edge_weights: np.ndarray,  # (E,)
    w_rel: np.ndarray,  # (R, D, D)
    b_rel: np.ndarray,  # (D,)
    clamped_mask: np.ndarray,  # (N,) uint8
    clamped_anchor: np.ndarray,  # (N, D)
    norm_eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Parallel relational message-passing forward kernel with residual connection, ReLU, and RMSNorm.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: `(h_out, pre_norm, rms_inv)` for backward pass.
    """
    n_nodes, dim = h_in.shape
    h_out = np.empty((n_nodes, dim), dtype=np.float32)
    pre_norm = np.empty((n_nodes, dim), dtype=np.float32)
    rms_inv = np.empty(n_nodes, dtype=np.float32)
    eps = np.float32(norm_eps)
    inv_d = np.float32(1.0 / max(dim, 1))

    for v in prange(n_nodes):
        msg = np.empty(dim, dtype=np.float32)
        for d in range(dim):
            msg[d] = b_rel[d]

        start = int(indptr[v])
        end = int(indptr[v + 1])
        for k in range(start, end):
            u = int(src_indices[k])
            r = int(rel_types[k])
            w_e = edge_weights[k]
            for d_in in range(dim):
                hu_val = w_e * h_in[u, d_in]
                for d_out in range(dim):
                    msg[d_out] += hu_val * w_rel[r, d_in, d_out]

        # ReLU activation on aggregated relational message + residual connection
        sq_sum = np.float32(0.0)
        is_clamped = clamped_mask[v] != 0
        for d in range(dim):
            act = msg[d] if msg[d] > np.float32(0.0) else np.float32(0.0)
            res = h_in[v, d] + act
            if is_clamped:
                res = np.float32(0.5) * res + np.float32(0.5) * clamped_anchor[v, d]
            pre_norm[v, d] = res
            sq_sum += res * res

        inv_rms = np.float32(1.0) / np.sqrt(sq_sum * inv_d + eps)
        rms_inv[v] = inv_rms
        for d in range(dim):
            h_out[v, d] = pre_norm[v, d] * inv_rms

    return h_out, pre_norm, rms_inv


@njit(cache=True, nogil=True, fastmath=True)
def mpnn_layer_backward(
    d_h_out: np.ndarray,  # (N, D)
    h_in: np.ndarray,  # (N, D)
    pre_norm: np.ndarray,  # (N, D)
    rms_inv: np.ndarray,  # (N,)
    indptr: np.ndarray,  # (N + 1,)
    src_indices: np.ndarray,  # (E,)
    rel_types: np.ndarray,  # (E,)
    edge_weights: np.ndarray,  # (E,)
    w_rel: np.ndarray,  # (R, D, D)
    clamped_mask: np.ndarray,  # (N,) uint8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Backward pass through RMSNorm, clamped residual anchor, ReLU, and CSR relational aggregation."""
    n_nodes, dim = h_in.shape
    n_rel = w_rel.shape[0]
    d_h_in = np.zeros((n_nodes, dim), dtype=np.float32)
    d_w_rel = np.zeros((n_rel, dim, dim), dtype=np.float32)
    d_b_rel = np.zeros(dim, dtype=np.float32)
    inv_d = np.float32(1.0 / max(dim, 1))

    for v in range(n_nodes):
        inv_rms = rms_inv[v]
        inv_rms3 = inv_rms * inv_rms * inv_rms * inv_d

        # RMSNorm backward: y = x * inv_rms => dx = dy * inv_rms - x * inv_rms^3 * (sum_j dy_j * x_j)
        dot_dy_x = np.float32(0.0)
        for d in range(dim):
            dot_dy_x += d_h_out[v, d] * pre_norm[v, d]

        d_res = np.empty(dim, dtype=np.float32)
        is_clamped = clamped_mask[v] != 0
        scale_c = np.float32(0.5) if is_clamped else np.float32(1.0)
        for d in range(dim):
            dr = (d_h_out[v, d] * inv_rms - pre_norm[v, d] * inv_rms3 * dot_dy_x) * scale_c
            d_res[d] = dr
            d_h_in[v, d] += dr

        # Reconstruct pre-ReLU check: act = (pre_norm / scale_c) - h_in > 0
        d_msg = np.empty(dim, dtype=np.float32)
        any_active = False
        for d in range(dim):
            raw_res = (
                (np.float32(2.0) * pre_norm[v, d]) if is_clamped else pre_norm[v, d]
            )
            act = raw_res - h_in[v, d]
            if act > np.float32(1e-7):
                d_msg[d] = d_res[d]
                d_b_rel[d] += d_res[d]
                any_active = True
            else:
                d_msg[d] = np.float32(0.0)

        if not any_active:
            continue

        start = int(indptr[v])
        end = int(indptr[v + 1])
        for k in range(start, end):
            u = int(src_indices[k])
            r = int(rel_types[k])
            w_e = edge_weights[k]
            for d_in in range(dim):
                hu_w = w_e * h_in[u, d_in]
                grad_u = np.float32(0.0)
                for d_out in range(dim):
                    dm = d_msg[d_out]
                    if dm != np.float32(0.0):
                        d_w_rel[r, d_in, d_out] += hu_w * dm
                        grad_u += w_rel[r, d_in, d_out] * dm
                d_h_in[u, d_in] += w_e * grad_u

    return d_h_in, d_w_rel, d_b_rel


@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def bilinear_edge_scores_forward(
    h_nodes: np.ndarray,  # (M, D)
    w_src: np.ndarray,  # (D, K)
    w_dst: np.ndarray,  # (D, K)
    w_chan: np.ndarray,  # (C, K)
    b_chan: np.ndarray,  # (C,)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute factorized bilinear multi-channel edge logits `S[u, v, c] = sum_k (h_u W_src)_k (h_v W_dst)_k W_chan[c, k] + b_c`."""
    m_nodes, dim = h_nodes.shape
    n_chan, rank_k = w_chan.shape
    u_proj = np.zeros((m_nodes, rank_k), dtype=np.float32)
    v_proj = np.zeros((m_nodes, rank_k), dtype=np.float32)

    for i in prange(m_nodes):
        for d in range(dim):
            hi = h_nodes[i, d]
            for k in range(rank_k):
                u_proj[i, k] += hi * w_src[d, k]
                v_proj[i, k] += hi * w_dst[d, k]

    scores = np.empty((m_nodes, m_nodes, n_chan), dtype=np.float32)
    inv_sqrt_k = np.float32(1.0 / np.sqrt(float(max(rank_k, 1))))

    for u in prange(m_nodes):
        for v in range(m_nodes):
            if u == v:
                for c in range(n_chan):
                    scores[u, v, c] = np.float32(-1e9)
                continue
            for c in range(n_chan):
                s = b_chan[c]
                for k in range(rank_k):
                    s += u_proj[u, k] * v_proj[v, k] * w_chan[c, k] * inv_sqrt_k
                scores[u, v, c] = s

    return scores, u_proj, v_proj


@njit(cache=True, nogil=True, fastmath=True)
def bilinear_edge_scores_backward(
    d_scores: np.ndarray,  # (M, M, C)
    h_nodes: np.ndarray,  # (M, D)
    u_proj: np.ndarray,  # (M, K)
    v_proj: np.ndarray,  # (M, K)
    w_src: np.ndarray,  # (D, K)
    w_dst: np.ndarray,  # (D, K)
    w_chan: np.ndarray,  # (C, K)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Backward pass for factorized bilinear multi-channel edge scorer."""
    m_nodes, dim = h_nodes.shape
    n_chan, rank_k = w_chan.shape
    inv_sqrt_k = np.float32(1.0 / np.sqrt(float(max(rank_k, 1))))

    d_u_proj = np.zeros((m_nodes, rank_k), dtype=np.float32)
    d_v_proj = np.zeros((m_nodes, rank_k), dtype=np.float32)
    d_w_chan = np.zeros((n_chan, rank_k), dtype=np.float32)
    d_b_chan = np.zeros(n_chan, dtype=np.float32)

    for u in range(m_nodes):
        for v in range(m_nodes):
            if u == v:
                continue
            for c in range(n_chan):
                ds = d_scores[u, v, c]
                if ds == np.float32(0.0):
                    continue
                d_b_chan[c] += ds
                ds_scaled = ds * inv_sqrt_k
                for k in range(rank_k):
                    uv_k = u_proj[u, k] * v_proj[v, k]
                    d_w_chan[c, k] += ds_scaled * uv_k
                    wc = w_chan[c, k] * ds_scaled
                    d_u_proj[u, k] += wc * v_proj[v, k]
                    d_v_proj[v, k] += wc * u_proj[u, k]

    d_h_nodes = np.zeros((m_nodes, dim), dtype=np.float32)
    d_w_src = np.zeros((dim, rank_k), dtype=np.float32)
    d_w_dst = np.zeros((dim, rank_k), dtype=np.float32)

    for i in range(m_nodes):
        for d in range(dim):
            hi = h_nodes[i, d]
            grad_hi = np.float32(0.0)
            for k in range(rank_k):
                du = d_u_proj[i, k]
                dv = d_v_proj[i, k]
                d_w_src[d, k] += hi * du
                d_w_dst[d, k] += hi * dv
                grad_hi += w_src[d, k] * du + w_dst[d, k] * dv
            d_h_nodes[i, d] = grad_hi

    return d_h_nodes, d_w_src, d_w_dst, d_w_chan, d_b_chan


@njit(cache=True, nogil=True, fastmath=True)
def sinkhorn_bipartite_matching(
    log_alpha: np.ndarray,
    n_iters: int = 20,
    tau: float = 1.0,
) -> np.ndarray:
    r"""Differentiable log-space Sinkhorn-Knopp operator producing a doubly stochastic soft assignment matrix.

    Args:
        log_alpha: 2D array `(M, K)` of unnormalized affinity logits between $M$ sources and $K$ targets.
        n_iters: Number of alternating log-row and log-column normalization iterations. Defaults to `20`.
        tau: Entropic temperature $\tau > 0$. Smaller $\tau$ approaches a hard permutation matrix.

    Returns:
        np.ndarray: 2D `float32` soft assignment matrix `P` of shape `(M, K)`.
    """
    m, k = log_alpha.shape
    if m == 0 or k == 0:
        return np.empty((m, k), dtype=np.float32)

    inv_tau = np.float32(1.0 / max(float(tau), 1e-4))
    log_p = np.empty((m, k), dtype=np.float32)
    for i in range(m):
        for j in range(k):
            log_p[i, j] = np.float32(log_alpha[i, j]) * inv_tau

    for _ in range(n_iters):
        # Row log-sum-exp normalization
        for i in range(m):
            max_r = np.float32(-1e30)
            for j in range(k):
                if log_p[i, j] > max_r:
                    max_r = log_p[i, j]
            s = np.float32(0.0)
            for j in range(k):
                s += np.exp(log_p[i, j] - max_r)
            lse = max_r + np.log(s + np.float32(1e-12))
            for j in range(k):
                log_p[i, j] -= lse

        # Column log-sum-exp normalization
        for j in range(k):
            max_c = np.float32(-1e30)
            for i in range(m):
                if log_p[i, j] > max_c:
                    max_c = log_p[i, j]
            s = np.float32(0.0)
            for i in range(m):
                s += np.exp(log_p[i, j] - max_c)
            lse = max_c + np.log(s + np.float32(1e-12))
            for i in range(m):
                log_p[i, j] -= lse

    out = np.empty((m, k), dtype=np.float32)
    for i in range(m):
        for j in range(k):
            out[i, j] = np.exp(log_p[i, j])
    return out


@njit(cache=True, nogil=True, fastmath=True)
def _adamw_step_1d(
    param: np.ndarray,
    grad: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    step_size: float,
    wd_factor: float,
    apply_wd: bool,
) -> None:
    beta1 = np.float32(0.9)
    beta2 = np.float32(0.999)
    eps = np.float32(1e-8)
    ss = np.float32(step_size)
    wf = np.float32(wd_factor)
    for i in range(len(param)):
        g = grad[i]
        m[i] = beta1 * m[i] + (np.float32(1.0) - beta1) * g
        v[i] = beta2 * v[i] + (np.float32(1.0) - beta2) * g * g
        if apply_wd:
            param[i] *= wf
        param[i] -= ss * m[i] / (np.sqrt(v[i]) + eps)


@njit(cache=True, nogil=True, fastmath=True)
def _adamw_step_2d(
    param: np.ndarray,
    grad: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    step_size: float,
    wd_factor: float,
) -> None:
    beta1 = np.float32(0.9)
    beta2 = np.float32(0.999)
    eps = np.float32(1e-8)
    ss = np.float32(step_size)
    wf = np.float32(wd_factor)
    r, c = param.shape
    for i in range(r):
        for j in range(c):
            g = grad[i, j]
            m[i, j] = beta1 * m[i, j] + (np.float32(1.0) - beta1) * g
            v[i, j] = beta2 * v[i, j] + (np.float32(1.0) - beta2) * g * g
            param[i, j] = param[i, j] * wf - ss * m[i, j] / (np.sqrt(v[i, j]) + eps)


@njit(cache=True, nogil=True, fastmath=True)
def _adamw_step_3d(
    param: np.ndarray,
    grad: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    step_size: float,
    wd_factor: float,
) -> None:
    beta1 = np.float32(0.9)
    beta2 = np.float32(0.999)
    eps = np.float32(1e-8)
    ss = np.float32(step_size)
    wf = np.float32(wd_factor)
    d0, d1, d2 = param.shape
    for i in range(d0):
        for j in range(d1):
            for k in range(d2):
                g = grad[i, j, k]
                m[i, j, k] = beta1 * m[i, j, k] + (np.float32(1.0) - beta1) * g
                v[i, j, k] = beta2 * v[i, j, k] + (np.float32(1.0) - beta2) * g * g
                param[i, j, k] = param[i, j, k] * wf - ss * m[i, j, k] / (np.sqrt(v[i, j, k]) + eps)


# ----------------------------------------------------------------------------------------------------------------------
# High-Level RelationalMPNN Model
# ----------------------------------------------------------------------------------------------------------------------
class RelationalMPNN:
    r"""Numba-powered Relational Message-Passing Neural Network (R-MPNN) for node & directed tree inference.

    Architecture:
        1. Input projection: $h^{(0)} = \text{RMSNorm}(X W_{\text{in}} + b_{\text{in}})$.
        2. $L$ Relational Message-Passing layers with residual connections, ReLU, RMSNorm, and
           optional clamped-node anchor injection (`clamped_mask`) so pre-existing annotations
           anchor contextual propagation without drifting.
        3. Node readout head: $Z_{\text{node}} = h^{(L)} W_{\text{node}} + b_{\text{node}}$ (`(N, num_node_classes)`).
        4. Factorized bilinear multi-slot directed edge readout head:
           $S[u, v, c] = \frac{1}{\sqrt{K}} \sum_{k=1}^K (h_u^{(L)} W_{\text{src}})_k (h_v^{(L)} W_{\text{dst}})_k W_{\text{chan}}[c, k] + b_{\text{chan}}[c]$
           producing `(M, M, num_edge_channels)` logits suitable for decoding via
           [`maximum_spanning_arborescence`][serogen.core.graph.arborescence.maximum_spanning_arborescence].
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_node_classes: int = 32,
        num_edge_channels: int = 8,
        num_relations: int = 8,
        num_layers: int = 3,
        bilinear_rank: int = 32,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-4,
        random_state: int | None = 42,
    ) -> None:
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_node_classes = int(num_node_classes)
        self.num_edge_channels = int(num_edge_channels)
        self.num_relations = int(num_relations)
        self.num_layers = int(num_layers)
        self.bilinear_rank = int(bilinear_rank)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.random_state = random_state

        self._step = 0
        self.loss_history_: list[float] = []
        self._init_weights()

    def _init_weights(self) -> None:
        rng = np.random.default_rng(self.random_state)
        d_in = self.in_dim
        d_hid = self.hidden_dim
        n_rel = self.num_relations
        n_lay = self.num_layers
        n_cls = self.num_node_classes
        n_chan = self.num_edge_channels
        rank_k = self.bilinear_rank

        self.w_in_ = rng.normal(0.0, np.sqrt(2.0 / max(d_in, 1)), size=(d_in, d_hid)).astype(np.float32)
        self.b_in_ = np.zeros(d_hid, dtype=np.float32)

        # Stack of L relational MPNN layers: (L, R, D, D) and (L, D)
        self.w_rel_ = rng.normal(
            0.0,
            np.sqrt(1.0 / max(d_hid * n_rel, 1)),
            size=(n_lay, n_rel, d_hid, d_hid),
        ).astype(np.float32)
        self.b_rel_ = np.zeros((n_lay, d_hid), dtype=np.float32)

        # Node classification head
        self.w_node_ = rng.normal(0.0, np.sqrt(1.0 / max(d_hid, 1)), size=(d_hid, n_cls)).astype(np.float32)
        self.b_node_ = np.zeros(n_cls, dtype=np.float32)

        # Factorized bilinear edge scoring head
        self.w_edge_src_ = rng.normal(0.0, np.sqrt(1.0 / max(d_hid, 1)), size=(d_hid, rank_k)).astype(np.float32)
        self.w_edge_dst_ = rng.normal(0.0, np.sqrt(1.0 / max(d_hid, 1)), size=(d_hid, rank_k)).astype(np.float32)
        self.w_edge_chan_ = rng.normal(0.0, np.sqrt(1.0 / max(rank_k, 1)), size=(n_chan, rank_k)).astype(np.float32)
        self.b_edge_chan_ = np.zeros(n_chan, dtype=np.float32)

        # AdamW moment buffers
        self._m_w_in = np.zeros_like(self.w_in_)
        self._v_w_in = np.zeros_like(self.w_in_)
        self._m_b_in = np.zeros_like(self.b_in_)
        self._v_b_in = np.zeros_like(self.b_in_)

        self._m_w_rel = np.zeros_like(self.w_rel_)
        self._v_w_rel = np.zeros_like(self.w_rel_)
        self._m_b_rel = np.zeros_like(self.b_rel_)
        self._v_b_rel = np.zeros_like(self.b_rel_)

        self._m_w_node = np.zeros_like(self.w_node_)
        self._v_w_node = np.zeros_like(self.w_node_)
        self._m_b_node = np.zeros_like(self.b_node_)
        self._v_b_node = np.zeros_like(self.b_node_)

        self._m_w_esrc = np.zeros_like(self.w_edge_src_)
        self._v_w_esrc = np.zeros_like(self.w_edge_src_)
        self._m_w_edst = np.zeros_like(self.w_edge_dst_)
        self._v_w_edst = np.zeros_like(self.w_edge_dst_)
        self._m_w_echan = np.zeros_like(self.w_edge_chan_)
        self._v_w_echan = np.zeros_like(self.w_edge_chan_)
        self._m_b_echan = np.zeros_like(self.b_edge_chan_)
        self._v_b_echan = np.zeros_like(self.b_edge_chan_)

    def _prepare_features(
        self,
        x: np.ndarray,
        csr: RelationalCSR,
        clamped_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pad virtual pool rows if `x` only has real vertices, and normalize `clamped_mask`."""
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim != 2 or x_arr.shape[1] != self.in_dim:
            raise ValueError(f"Expected 2D input features of width {self.in_dim}; got {x_arr.shape}.")
        n_total = csr.n_nodes
        if x_arr.shape[0] < n_total:
            pad = np.zeros((n_total - x_arr.shape[0], self.in_dim), dtype=np.float32)
            x_full = np.ascontiguousarray(np.vstack([x_arr, pad]))
        elif x_arr.shape[0] == n_total:
            x_full = np.ascontiguousarray(x_arr)
        else:
            raise ValueError(f"x has {x_arr.shape[0]} rows, exceeding csr.n_nodes ({n_total}).")

        if clamped_mask is None:
            c_mask = np.zeros(n_total, dtype=np.uint8)
        else:
            cm = np.asarray(clamped_mask, dtype=np.uint8)
            if len(cm) < n_total:
                c_mask = np.zeros(n_total, dtype=np.uint8)
                c_mask[: len(cm)] = cm
            else:
                c_mask = np.ascontiguousarray(cm[:n_total])
        return x_full, c_mask

    def encode(
        self,
        x: np.ndarray,
        csr: RelationalCSR,
        *,
        clamped_mask: np.ndarray | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        r"""Run the $L$-layer R-MPNN encoder and return final node embeddings `(N, hidden_dim)`."""
        x_full, c_mask = self._prepare_features(x, csr, clamped_mask)
        h0 = x_full @ self.w_in_ + self.b_in_[None, :]
        h = np.ascontiguousarray(h0, dtype=np.float32)
        for lay in range(self.num_layers):
            h, _, _ = mpnn_layer_forward(
                h,
                csr.indptr,
                csr.src_indices,
                csr.rel_types,
                csr.edge_weights,
                self.w_rel_[lay],
                self.b_rel_[lay],
                c_mask,
                h0,
            )
        return h

    def predict_node_logits(
        self,
        x: np.ndarray,
        csr: RelationalCSR,
        *,
        clamped_mask: np.ndarray | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        r"""Compute pre-softmax node class logits `(N, num_node_classes)`."""
        h = self.encode(x, csr, clamped_mask=clamped_mask)
        return np.ascontiguousarray(h @ self.w_node_ + self.b_node_[None, :], dtype=np.float32)

    def predict_node_proba(
        self,
        x: np.ndarray,
        csr: RelationalCSR,
        *,
        clamped_mask: np.ndarray | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        r"""Compute softmax node class probabilities `(N, num_node_classes)`."""
        logits = self.predict_node_logits(x, csr, clamped_mask=clamped_mask)
        max_l = np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(logits - max_l)
        return (exp_l / np.maximum(np.sum(exp_l, axis=1, keepdims=True), np.float32(1e-12))).astype(np.float32)

    def score_edges(
        self,
        h_nodes: np.ndarray,
        node_indices: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float32]]:
        r"""Score all directed candidate edges `(u -> v, channel)` among `h_nodes` (or `h_nodes[node_indices]`).

        Returns:
            np.ndarray: 3D `float32` tensor of shape `(M, M, num_edge_channels)`.
        """
        h_sub = (
            np.ascontiguousarray(h_nodes[np.asarray(node_indices, dtype=np.int32)], dtype=np.float32)
            if node_indices is not None
            else np.ascontiguousarray(h_nodes, dtype=np.float32)
        )
        scores, _, _ = bilinear_edge_scores_forward(
            h_sub,
            self.w_edge_src_,
            self.w_edge_dst_,
            self.w_edge_chan_,
            self.b_edge_chan_,
        )
        return scores

    def fit_step(
        self,
        x: np.ndarray,
        csr: RelationalCSR,
        *,
        node_targets: np.ndarray | None = None,
        node_weights: np.ndarray | None = None,
        edge_node_indices: np.ndarray | None = None,
        edge_parent_targets: np.ndarray | None = None,
        edge_slot_targets: np.ndarray | None = None,
        clamped_mask: np.ndarray | None = None,
        lr_scale: float = 1.0,
    ) -> float:
        r"""Execute one complete forward + backward + AdamW optimization step on a graph or batch.

        Args:
            x: Input node feature matrix `(N_real, in_dim)` or `(N_total, in_dim)`.
            csr: [`RelationalCSR`][serogen.core.graph.gnn.RelationalCSR] connectivity structure.
            node_targets: Optional 1D `int32` array of ground-truth node class indices (`-1` = ignore).
            node_weights: Optional 1D `float32` array of per-node loss weights.
            edge_node_indices: Optional 1D `int32` array of active tree vertices `[v_0, ..., v_{M-1}]`.
            edge_parent_targets: Optional 1D `int32` array of length `M` giving target parent index in `[0, M-1]`
                for each active tree vertex (`-1` for root or unsupervised vertices).
            edge_slot_targets: Optional 1D `int32` array of length `M` giving target slot/channel in
                `[0, num_edge_channels - 1]` for each active tree vertex (`-1` = marginalize over slots).
            clamped_mask: Optional 1D `uint8` / boolean mask of clamped vertices.
            lr_scale: Cosine/schedule multiplier on `self.learning_rate`.

        Returns:
            float: Total scalar loss for this step.
        """
        x_full, c_mask = self._prepare_features(x, csr, clamped_mask)
        n_total = csr.n_nodes

        # 1. Forward pass with layer caches
        h0 = np.ascontiguousarray(x_full @ self.w_in_ + self.b_in_[None, :], dtype=np.float32)
        h_inputs: list[np.ndarray] = []
        pre_norms: list[np.ndarray] = []
        rms_invs: list[np.ndarray] = []

        h_curr = h0
        for lay in range(self.num_layers):
            h_inputs.append(h_curr)
            h_curr, pn, ri = mpnn_layer_forward(
                h_curr,
                csr.indptr,
                csr.src_indices,
                csr.rel_types,
                csr.edge_weights,
                self.w_rel_[lay],
                self.b_rel_[lay],
                c_mask,
                h0,
            )
            pre_norms.append(pn)
            rms_invs.append(ri)

        total_loss = 0.0
        d_h_final = np.zeros_like(h_curr)

        # 2a. Node classification cross-entropy loss & gradient
        d_w_node = np.zeros_like(self.w_node_)
        d_b_node = np.zeros_like(self.b_node_)
        if node_targets is not None:
            y_node = np.asarray(node_targets, dtype=np.int32)
            n_sup = min(len(y_node), n_total)
            w_n = (
                np.asarray(node_weights, dtype=np.float32)
                if node_weights is not None
                else np.ones(n_sup, dtype=np.float32)
            )
            valid_n = np.where((y_node[:n_sup] >= 0) & (y_node[:n_sup] < self.num_node_classes))[0]
            if len(valid_n) > 0:
                logits_n = h_curr[valid_n] @ self.w_node_ + self.b_node_[None, :]
                max_l = np.max(logits_n, axis=1, keepdims=True)
                exp_l = np.exp(logits_n - max_l)
                probs_n = exp_l / np.maximum(np.sum(exp_l, axis=1, keepdims=True), np.float32(1e-12))

                targets_v = y_node[valid_n]
                weights_v = w_n[valid_n] / np.float32(max(len(valid_n), 1))
                picked_p = probs_n[np.arange(len(valid_n)), targets_v]
                total_loss += float(np.sum(-weights_v * np.log(picked_p + np.float32(1e-10))))

                d_logits_n = probs_n.copy()
                d_logits_n[np.arange(len(valid_n)), targets_v] -= np.float32(1.0)
                d_logits_n *= weights_v[:, None]

                d_w_node = h_curr[valid_n].T @ d_logits_n
                d_b_node = np.sum(d_logits_n, axis=0)
                d_h_final[valid_n] += d_logits_n @ self.w_node_.T

        # 2b. Directed arborescence parent+slot cross-entropy loss & gradient
        d_w_esrc = np.zeros_like(self.w_edge_src_)
        d_w_edst = np.zeros_like(self.w_edge_dst_)
        d_w_echan = np.zeros_like(self.w_edge_chan_)
        d_b_echan = np.zeros_like(self.b_edge_chan_)

        if edge_parent_targets is not None:
            e_idx = (
                np.asarray(edge_node_indices, dtype=np.int32)
                if edge_node_indices is not None
                else np.arange(len(edge_parent_targets), dtype=np.int32)
            )
            p_targets = np.asarray(edge_parent_targets, dtype=np.int32)
            s_targets = (
                np.asarray(edge_slot_targets, dtype=np.int32)
                if edge_slot_targets is not None
                else np.zeros(len(p_targets), dtype=np.int32)
            )
            m_sub = len(e_idx)
            if m_sub >= 2:
                h_sub = np.ascontiguousarray(h_curr[e_idx], dtype=np.float32)
                scores, u_proj, v_proj = bilinear_edge_scores_forward(
                    h_sub,
                    self.w_edge_src_,
                    self.w_edge_dst_,
                    self.w_edge_chan_,
                    self.b_edge_chan_,
                )
                d_scores = np.zeros_like(scores)
                sup_children = np.where((p_targets >= 0) & (p_targets < m_sub))[0]
                if len(sup_children) > 0:
                    inv_m = np.float32(1.0 / len(sup_children))
                    for v_loc in sup_children:
                        u_true = int(p_targets[v_loc])
                        if u_true == v_loc:
                            continue
                        s_true = int(s_targets[v_loc])
                        if s_true < 0 or s_true >= self.num_edge_channels:
                            s_true = 0
                        # Softmax over all candidate (parent u != v_loc, channel c) pairs for child v_loc
                        col_logits = scores[:, v_loc, :].reshape(-1)
                        max_c = float(np.max(col_logits))
                        exp_c = np.exp(col_logits - max_c)
                        sum_c = float(np.sum(exp_c)) + 1e-12
                        probs_c = (exp_c / sum_c).reshape(m_sub, self.num_edge_channels)

                        p_gold = float(probs_c[u_true, s_true])
                        total_loss += float(-inv_m * np.log(max(p_gold, 1e-10)))

                        grad_slice = probs_c * inv_m
                        grad_slice[u_true, s_true] -= inv_m
                        grad_slice[v_loc, :] = 0.0
                        d_scores[:, v_loc, :] += grad_slice

                    d_h_sub, d_w_esrc, d_w_edst, d_w_echan, d_b_echan = bilinear_edge_scores_backward(
                        d_scores,
                        h_sub,
                        u_proj,
                        v_proj,
                        self.w_edge_src_,
                        self.w_edge_dst_,
                        self.w_edge_chan_,
                    )
                    d_h_final[e_idx] += d_h_sub

        # 3. Backward pass through L relational MPNN layers
        d_w_rel_all = np.zeros_like(self.w_rel_)
        d_b_rel_all = np.zeros_like(self.b_rel_)
        d_h_curr = d_h_final
        d_h0_accum = np.zeros_like(h0)

        for lay in range(self.num_layers - 1, -1, -1):
            # Account for clamped anchor gradient flowing directly to h0
            if np.any(c_mask != 0):
                inv_rms = rms_invs[lay]
                pn = pre_norms[lay]
                inv_d = np.float32(1.0 / max(self.hidden_dim, 1))
                for v in range(n_total):
                    if c_mask[v] != 0:
                        ir = inv_rms[v]
                        ir3 = ir * ir * ir * inv_d
                        dot_val = np.float32(np.sum(d_h_curr[v] * pn[v]))
                        d_h0_accum[v] += np.float32(0.5) * (d_h_curr[v] * ir - pn[v] * ir3 * dot_val)

            d_h_curr, dw_l, db_l = mpnn_layer_backward(
                d_h_curr,
                h_inputs[lay],
                pre_norms[lay],
                rms_invs[lay],
                csr.indptr,
                csr.src_indices,
                csr.rel_types,
                csr.edge_weights,
                self.w_rel_[lay],
                c_mask,
            )
            d_w_rel_all[lay] = dw_l
            d_b_rel_all[lay] = db_l

        d_h0_total = d_h_curr + d_h0_accum
        d_w_in = x_full.T @ d_h0_total
        d_b_in = np.sum(d_h0_total, axis=0)

        # 4. AdamW parameter update
        self._step += 1
        bc1 = 1.0 - (0.9**self._step)
        bc2 = 1.0 - (0.999**self._step)
        lr_t = float(self.learning_rate * lr_scale)
        step_size = float(lr_t * np.sqrt(bc2) / bc1)
        wd_factor = float(1.0 - lr_t * self.weight_decay)

        _adamw_step_2d(self.w_in_, d_w_in, self._m_w_in, self._v_w_in, step_size, wd_factor)
        _adamw_step_1d(self.b_in_, d_b_in, self._m_b_in, self._v_b_in, step_size, wd_factor, False)

        for lay in range(self.num_layers):
            _adamw_step_3d(
                self.w_rel_[lay],
                d_w_rel_all[lay],
                self._m_w_rel[lay],
                self._v_w_rel[lay],
                step_size,
                wd_factor,
            )
            _adamw_step_1d(
                self.b_rel_[lay],
                d_b_rel_all[lay],
                self._m_b_rel[lay],
                self._v_b_rel[lay],
                step_size,
                wd_factor,
                False,
            )

        if node_targets is not None:
            _adamw_step_2d(self.w_node_, d_w_node, self._m_w_node, self._v_w_node, step_size, wd_factor)
            _adamw_step_1d(self.b_node_, d_b_node, self._m_b_node, self._v_b_node, step_size, wd_factor, False)

        if edge_parent_targets is not None:
            _adamw_step_2d(self.w_edge_src_, d_w_esrc, self._m_w_esrc, self._v_w_esrc, step_size, wd_factor)
            _adamw_step_2d(self.w_edge_dst_, d_w_edst, self._m_w_edst, self._v_w_edst, step_size, wd_factor)
            _adamw_step_2d(self.w_edge_chan_, d_w_echan, self._m_w_echan, self._v_w_echan, step_size, wd_factor)
            _adamw_step_1d(
                self.b_edge_chan_,
                d_b_echan,
                self._m_b_echan,
                self._v_b_echan,
                step_size,
                wd_factor,
                False,
            )

        self.loss_history_.append(total_loss)
        return total_loss

    def save(self, path: Path | str) -> Path:
        r"""Save model weights and hyperparameters to a compressed `.npz` archive (`allow_pickle=False`)."""
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".npz")
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = np.array(
            [
                self.in_dim,
                self.hidden_dim,
                self.num_node_classes,
                self.num_edge_channels,
                self.num_relations,
                self.num_layers,
                self.bilinear_rank,
                -1 if self.random_state is None else int(self.random_state),
            ],
            dtype=np.int64,
        )
        float_meta = np.array([self.learning_rate, self.weight_decay], dtype=np.float64)
        np.savez_compressed(
            p,
            meta=meta,
            float_meta=float_meta,
            w_in=self.w_in_,
            b_in=self.b_in_,
            w_rel=self.w_rel_,
            b_rel=self.b_rel_,
            w_node=self.w_node_,
            b_node=self.b_node_,
            w_edge_src=self.w_edge_src_,
            w_edge_dst=self.w_edge_dst_,
            w_edge_chan=self.w_edge_chan_,
            b_edge_chan=self.b_edge_chan_,
            loss_history=np.array(self.loss_history_, dtype=np.float32),
        )
        return p

    @classmethod
    def load(cls, path: Path | str) -> Self:
        r"""Load a [`RelationalMPNN`][serogen.core.graph.gnn.RelationalMPNN] from a compressed `.npz` archive."""
        p = Path(path)
        if not p.suffix and not p.exists():
            p = p.with_suffix(".npz")
        with np.load(p, allow_pickle=False) as data:
            meta = data["meta"]
            float_meta = data["float_meta"]
            rs = None if int(meta[7]) < 0 else int(meta[7])
            inst = cls(
                in_dim=int(meta[0]),
                hidden_dim=int(meta[1]),
                num_node_classes=int(meta[2]),
                num_edge_channels=int(meta[3]),
                num_relations=int(meta[4]),
                num_layers=int(meta[5]),
                bilinear_rank=int(meta[6]),
                learning_rate=float(float_meta[0]),
                weight_decay=float(float_meta[1]),
                random_state=rs,
            )
            inst.w_in_ = data["w_in"].astype(np.float32)
            inst.b_in_ = data["b_in"].astype(np.float32)
            inst.w_rel_ = data["w_rel"].astype(np.float32)
            inst.b_rel_ = data["b_rel"].astype(np.float32)
            inst.w_node_ = data["w_node"].astype(np.float32)
            inst.b_node_ = data["b_node"].astype(np.float32)
            inst.w_edge_src_ = data["w_edge_src"].astype(np.float32)
            inst.w_edge_dst_ = data["w_edge_dst"].astype(np.float32)
            inst.w_edge_chan_ = data["w_edge_chan"].astype(np.float32)
            inst.b_edge_chan_ = data["b_edge_chan"].astype(np.float32)
            inst.loss_history_ = [float(x) for x in data["loss_history"].tolist()]
            return inst
