r"""Directed spanning arborescence solvers (Chu-Liu/Edmonds & Constrained $K$-Best Branch-and-Bound).

Provides compiled Numba algorithms for finding maximum/minimum weight directed spanning trees
(arborescences) rooted at a target node, with optional slot/valence exclusivity constraints
and $k$-best arborescence enumeration:

- [`maximum_spanning_arborescence`][serogen.core.graph.arborescence.maximum_spanning_arborescence]
- [`minimum_spanning_arborescence`][serogen.core.graph.arborescence.minimum_spanning_arborescence]
- [`k_best_arborescences`][serogen.core.graph.arborescence.k_best_arborescences]
- [`ArborescenceResult`][serogen.core.graph.arborescence.ArborescenceResult]
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit

_NEG_INF = -1e30


@dataclass(slots=True, frozen=True)
class ArborescenceResult:
    r"""Result of a directed spanning arborescence optimization.

    Supports both standard 2-tuple unpacking (`parents, score = maximum_spanning_arborescence(...)`)
    and rich attribute access (`.parents`, `.score`, `.slots`, `.edges`).

    Attributes:
        parents: 1D `int32` array of shape `(N,)` where `parents[v]` is the parent node $u$ of $v$
            (`-1` for `root` or excluded nodes).
        score: Total weight (or cost) of the selected arborescence edges.
        slots: 1D `int32` array of shape `(N,)` giving the selected channel/slot index $s$ on the
            incoming edge $(u \to v, s)$ (`0` for 2D inputs, `-1` for `root`).
        edges: 2D `int32` array of shape `(M, 3)` containing `(parent_u, child_v, slot_s)` for each
            directed edge in the arborescence.
    """

    parents: np.ndarray[tuple[int], np.dtype[np.int32]]
    score: float
    slots: np.ndarray[tuple[int], np.dtype[np.int32]]
    edges: np.ndarray[tuple[int, int], np.dtype[np.int32]]

    def __iter__(self) -> Iterator[Any]:
        r"""Unpack as `(parents, score)` for tuple assignment compatibility."""
        yield self.parents
        yield self.score

    def __len__(self) -> int:
        return 2

    def __getitem__(self, item: int) -> Any:
        if item == 0:
            return self.parents
        if item == 1:
            return self.score
        if item == 2:
            return self.slots
        raise IndexError(f"ArborescenceResult index {item} out of range [0, 2].")


# ----------------------------------------------------------------------------------------------------------------------
# Iterative Chu-Liu / Edmonds Unconstrained 2D MSA Kernel (Zero Recursion)
# ----------------------------------------------------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _edmonds_greedy_and_cycle(
    weights: np.ndarray,
    root: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Select best incoming edges and detect if a directed cycle exists."""
    n = weights.shape[0]
    parents = np.full(n, -1, dtype=np.int32)
    best_w = np.full(n, _NEG_INF, dtype=np.float64)

    for v in range(n):
        if v == root:
            continue
        max_w = _NEG_INF
        best_u = -1
        for u in range(n):
            if u == v:
                continue
            w = float(weights[u, v])
            if w > max_w:
                max_w = w
                best_u = u
        parents[v] = best_u
        best_w[v] = max_w

    visited = np.full(n, -1, dtype=np.int32)
    cycle_nodes = np.full(n, -1, dtype=np.int32)
    cycle_len = 0
    in_cycle = np.zeros(n, dtype=np.uint8)

    for start_v in range(n):
        if start_v == root or visited[start_v] != -1:
            continue
        curr = start_v
        steps = 0
        while curr != -1 and curr != root and visited[curr] == -1 and steps <= n:
            visited[curr] = start_v
            curr = int(parents[curr])
            steps += 1
        if curr != -1 and curr != root and visited[curr] == start_v:
            c_start = curr
            while True:
                cycle_nodes[cycle_len] = curr
                in_cycle[curr] = 1
                cycle_len += 1
                curr = int(parents[curr])
                if curr == c_start or cycle_len >= n:
                    break
            break

    return parents, best_w, cycle_nodes, in_cycle, cycle_len


def _edmonds_msa_2d(weights: np.ndarray, root: int) -> tuple[np.ndarray, float]:
    """Iterative stack-based Chu-Liu/Edmonds optimum branching solver for 2D `(N, N)` weights."""
    n0 = weights.shape[0]
    if n0 <= 1:
        return np.full(n0, -1, dtype=np.int32), 0.0

    # Stack of contraction frames for iterative expansion
    stack: list[
        tuple[
            int,
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ] = []

    curr_w = weights
    curr_root = int(root)
    curr_parents = np.full(n0, -1, dtype=np.int32)

    while True:
        n = curr_w.shape[0]
        parents, best_w, cycle_nodes, in_cycle, cycle_len = _edmonds_greedy_and_cycle(curr_w, curr_root)
        if cycle_len == 0:
            curr_parents = parents
            break

        new_n = n - cycle_len + 1
        c_id = new_n - 1
        old_to_new = np.empty(n, dtype=np.int32)
        new_to_old = np.empty(new_n, dtype=np.int32)
        idx = 0
        for v in range(n):
            if in_cycle[v] != 0:
                old_to_new[v] = c_id
            else:
                old_to_new[v] = idx
                new_to_old[idx] = v
                idx += 1
        new_to_old[c_id] = cycle_nodes[0]

        contracted_w = np.full((new_n, new_n), _NEG_INF, dtype=np.float64)
        enter_u = np.full(new_n, -1, dtype=np.int32)
        enter_v = np.full(new_n, -1, dtype=np.int32)
        exit_u = np.full(new_n, -1, dtype=np.int32)

        for u in range(n):
            nu = int(old_to_new[u])
            for v in range(n):
                if u == v:
                    continue
                nv = int(old_to_new[v])
                if nu == nv:
                    continue
                w = float(curr_w[u, v])
                if w <= _NEG_INF * 0.1:
                    continue
                if in_cycle[u] == 0 and in_cycle[v] == 0:
                    if w > contracted_w[nu, nv]:
                        contracted_w[nu, nv] = w
                elif in_cycle[u] == 0 and in_cycle[v] != 0:
                    adj_w = w - best_w[v]
                    if adj_w > contracted_w[nu, c_id]:
                        contracted_w[nu, c_id] = adj_w
                        enter_u[nu] = u
                        enter_v[nu] = v
                elif in_cycle[u] != 0 and in_cycle[v] == 0:
                    if w > contracted_w[c_id, nv]:
                        contracted_w[c_id, nv] = w
                        exit_u[nv] = u

        new_root = int(old_to_new[curr_root])
        stack.append(
            (
                n,
                curr_root,
                parents,
                cycle_nodes[:cycle_len].copy(),
                old_to_new,
                new_to_old,
                enter_u,
                enter_v,
                exit_u,
                np.array([c_id, new_root], dtype=np.int32),
            )
        )
        curr_w = contracted_w
        curr_root = new_root

    # Unwind stack to expand contracted cycles
    while stack:
        (
            n,
            frame_root,
            parents,
            cyc_nodes,
            _old_to_new,
            new_to_old,
            enter_u,
            enter_v,
            exit_u,
            meta_ids,
        ) = stack.pop()
        c_id = int(meta_ids[0])
        new_root = int(meta_ids[1])
        new_n = len(new_to_old)

        expanded = np.full(n, -1, dtype=np.int32)
        for cv in cyc_nodes:
            expanded[int(cv)] = parents[int(cv)]

        for nv in range(new_n):
            if nv == new_root:
                continue
            np_u = int(curr_parents[nv])
            if np_u < 0:
                continue
            if nv == c_id:
                orig_u = int(enter_u[np_u])
                orig_v = int(enter_v[np_u])
                expanded[orig_v] = orig_u
            else:
                orig_v = int(new_to_old[nv])
                if np_u == c_id:
                    expanded[orig_v] = exit_u[nv]
                else:
                    expanded[orig_v] = new_to_old[np_u]
        expanded[frame_root] = -1
        curr_parents = expanded

    total_score = 0.0
    for v in range(n0):
        if v != root and curr_parents[v] >= 0:
            total_score += float(weights[curr_parents[v], v])
    return curr_parents, total_score


# ----------------------------------------------------------------------------------------------------------------------
# Numba Branch-and-Bound Constrained & K-Best Arborescence Kernel
# ----------------------------------------------------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _prepare_sorted_incoming_candidates(
    weights_3d: np.ndarray,  # (N, N, S)
    groups_3d: np.ndarray,  # (N, N, S)
    active_mask: np.ndarray,  # (N,) uint8
    root: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-sort valid incoming candidate edges `(u, slot, weight, group)` per target vertex `v`."""
    n, _, s_dim = weights_3d.shape
    max_cands = n * s_dim
    cand_u = np.full((n, max_cands), -1, dtype=np.int32)
    cand_s = np.full((n, max_cands), -1, dtype=np.int32)
    cand_g = np.full((n, max_cands), -1, dtype=np.int32)
    cand_w = np.full((n, max_cands), _NEG_INF, dtype=np.float64)
    cand_counts = np.zeros(n, dtype=np.int32)

    for v in range(n):
        if v == root or active_mask[v] == 0:
            continue
        cnt = 0
        for u in range(n):
            if u == v or active_mask[u] == 0:
                continue
            for s in range(s_dim):
                w = float(weights_3d[u, v, s])
                if w > _NEG_INF * 0.1:
                    cand_u[v, cnt] = u
                    cand_s[v, cnt] = s
                    cand_g[v, cnt] = int(groups_3d[u, v, s])
                    cand_w[v, cnt] = w
                    cnt += 1
        # Insertion sort descending by weight
        for i in range(1, cnt):
            ku = cand_u[v, i]
            ks = cand_s[v, i]
            kg = cand_g[v, i]
            kw = cand_w[v, i]
            j = i - 1
            while j >= 0 and cand_w[v, j] < kw:
                cand_u[v, j + 1] = cand_u[v, j]
                cand_s[v, j + 1] = cand_s[v, j]
                cand_g[v, j + 1] = cand_g[v, j]
                cand_w[v, j + 1] = cand_w[v, j]
                j -= 1
            cand_u[v, j + 1] = ku
            cand_s[v, j + 1] = ks
            cand_g[v, j + 1] = kg
            cand_w[v, j + 1] = kw
        cand_counts[v] = cnt

    # Order non-root active vertices by descending regret (w_0 - w_1) so tightest bottlenecks branch first
    active_non_root = np.empty(n, dtype=np.int32)
    regrets = np.empty(n, dtype=np.float64)
    m_active = 0
    for v in range(n):
        if v == root or active_mask[v] == 0:
            continue
        active_non_root[m_active] = v
        if cand_counts[v] >= 2:
            regrets[m_active] = cand_w[v, 0] - cand_w[v, 1]
        elif cand_counts[v] == 1:
            regrets[m_active] = 1e12
        else:
            regrets[m_active] = 1e15
        m_active += 1

    active_non_root = active_non_root[:m_active]
    regrets = regrets[:m_active]
    for i in range(1, m_active):
        kv = active_non_root[i]
        kr = regrets[i]
        j = i - 1
        while j >= 0 and regrets[j] < kr:
            active_non_root[j + 1] = active_non_root[j]
            regrets[j + 1] = regrets[j]
            j -= 1
        active_non_root[j + 1] = kv
        regrets[j + 1] = kr

    return cand_u, cand_s, cand_g, cand_w, cand_counts, active_non_root


@njit(cache=True, nogil=True)
def _creates_cycle(parents: np.ndarray, u: int, v: int, n: int) -> bool:
    """Return True if adding directed edge `u -> v` creates a cycle in `parents`."""
    curr = u
    steps = 0
    while curr >= 0 and steps <= n:
        if curr == v:
            return True
        curr = int(parents[curr])
        steps += 1
    return False


@njit(cache=True, nogil=True)
def _branch_and_bound_k_best(
    weights_3d: np.ndarray,
    groups_3d: np.ndarray,
    active_mask: np.ndarray,
    root: int,
    k_best: int,
    max_backtracks: int = 250000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Iterative stack-based Numba branch-and-bound for exact constrained top-`k` arborescences."""
    n = weights_3d.shape[0]
    cand_u, cand_s, cand_g, cand_w, cand_counts, order_v = _prepare_sorted_incoming_candidates(
        weights_3d, groups_3d, active_mask, root
    )
    m = len(order_v)
    out_parents = np.full((k_best, n), -1, dtype=np.int32)
    out_slots = np.full((k_best, n), -1, dtype=np.int32)
    out_scores = np.full(k_best, _NEG_INF, dtype=np.float64)

    if m == 0:
        out_scores[0] = 0.0
        return out_parents, out_slots, out_scores, 1

    for idx in range(m):
        if cand_counts[order_v[idx]] == 0:
            return out_parents, out_slots, out_scores, 0

    # Suffix optimistic upper bound: sum of best incoming weights from `depth..m-1`
    suffix_bound = np.zeros(m + 1, dtype=np.float64)
    for idx in range(m - 1, -1, -1):
        suffix_bound[idx] = suffix_bound[idx + 1] + cand_w[order_v[idx], 0]

    max_group = int(np.max(groups_3d))
    used_groups = np.zeros(max(max_group + 1, 1), dtype=np.uint8)

    curr_parents = np.full(n, -1, dtype=np.int32)
    curr_slots = np.full(n, -1, dtype=np.int32)
    curr_cand_idx = np.zeros(m, dtype=np.int32)
    curr_score_prefix = np.zeros(m + 1, dtype=np.float64)

    depth = 0
    found = 0
    backtracks = 0

    while depth >= 0:
        if depth == m:
            # Complete valid arborescence found! Insert into sorted top-k buffer
            total_sc = curr_score_prefix[m]
            if found < k_best or total_sc > out_scores[k_best - 1]:
                ins = found if found < k_best else k_best - 1
                while ins > 0 and out_scores[ins - 1] < total_sc:
                    if ins < k_best:
                        out_scores[ins] = out_scores[ins - 1]
                        out_parents[ins] = out_parents[ins - 1]
                        out_slots[ins] = out_slots[ins - 1]
                    ins -= 1
                out_scores[ins] = total_sc
                out_parents[ins] = curr_parents.copy()
                out_slots[ins] = curr_slots.copy()
                if found < k_best:
                    found += 1

            depth -= 1
            if depth >= 0:
                v_prev = int(order_v[depth])
                c_prev = int(curr_cand_idx[depth])
                g_prev = int(cand_g[v_prev, c_prev])
                if g_prev >= 0:
                    used_groups[g_prev] = 0
                curr_parents[v_prev] = -1
                curr_slots[v_prev] = -1
                curr_cand_idx[depth] += 1
            continue

        v = int(order_v[depth])
        c_idx = int(curr_cand_idx[depth])
        n_cands = int(cand_counts[v])
        threshold = out_scores[k_best - 1] if found == k_best else _NEG_INF * 0.5

        advanced = False
        while c_idx < n_cands:
            w = cand_w[v, c_idx]
            # Since candidates for `v` are sorted descending by weight, if optimistic bound fails we can prune!
            if curr_score_prefix[depth] + w + suffix_bound[depth + 1] <= threshold + 1e-12:
                c_idx = n_cands
                break

            g = int(cand_g[v, c_idx])
            if g >= 0 and used_groups[g] != 0:
                c_idx += 1
                continue

            u = int(cand_u[v, c_idx])
            if _creates_cycle(curr_parents, u, v, n):
                c_idx += 1
                continue

            # Valid edge choice at `depth`
            curr_cand_idx[depth] = c_idx
            curr_parents[v] = u
            curr_slots[v] = cand_s[v, c_idx]
            if g >= 0:
                used_groups[g] = 1
            curr_score_prefix[depth + 1] = curr_score_prefix[depth] + w
            depth += 1
            if depth < m:
                curr_cand_idx[depth] = 0
            advanced = True
            break

        if not advanced:
            backtracks += 1
            if backtracks >= max_backtracks and found > 0:
                break
            depth -= 1
            if depth >= 0:
                v_prev = int(order_v[depth])
                c_prev = int(curr_cand_idx[depth])
                g_prev = int(cand_g[v_prev, c_prev])
                if g_prev >= 0:
                    used_groups[g_prev] = 0
                curr_parents[v_prev] = -1
                curr_slots[v_prev] = -1
                curr_cand_idx[depth] += 1

    return out_parents[:found], out_slots[:found], out_scores[:found], found


# ----------------------------------------------------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------------------------------------------------
def _normalize_arborescence_inputs(
    weights: np.ndarray,
    root: int,
    edge_groups: np.ndarray | None,
    required_nodes: np.ndarray | None,
    slot_exclusivity: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Normalize 2D `(N, N)` or 3D `(N, N, S)` weights and exclusivity groups."""
    w_arr = np.asarray(weights, dtype=np.float64)
    if w_arr.ndim == 2:
        n = w_arr.shape[0]
        if w_arr.shape[1] != n:
            raise ValueError(f"2D weights matrix must be square (N, N); got {w_arr.shape}.")
        w_3d = np.ascontiguousarray(w_arr[:, :, None])
        s_dim = 1
    elif w_arr.ndim == 3:
        n, n2, s_dim = w_arr.shape
        if n != n2:
            raise ValueError(f"3D weights tensor must have shape (N, N, S); got {w_arr.shape}.")
        w_3d = np.ascontiguousarray(w_arr)
    else:
        raise ValueError(f"weights must be 2D (N, N) or 3D (N, N, S); got {w_arr.ndim}D.")

    if n > 0 and (root < 0 or root >= n):
        raise ValueError(f"root index {root} out of range [0, {n - 1}].")

    active_mask = np.ones(n, dtype=np.uint8)
    if required_nodes is not None:
        req = np.asarray(required_nodes)
        if req.dtype == np.bool_:
            if len(req) != n:
                raise ValueError(f"Boolean required_nodes mask length {len(req)} != {n}.")
            active_mask = req.astype(np.uint8)
        else:
            active_mask.fill(0)
            active_mask[req.astype(np.int32)] = 1
        if n > 0:
            active_mask[root] = 1

    has_constraints = False
    if edge_groups is not None:
        g_arr = np.asarray(edge_groups, dtype=np.int32)
        if g_arr.ndim == 2 and w_arr.ndim == 2:
            g_3d = np.ascontiguousarray(g_arr[:, :, None])
        elif g_arr.shape == w_3d.shape:
            g_3d = np.ascontiguousarray(g_arr)
        else:
            raise ValueError(f"edge_groups shape {g_arr.shape} incompatible with weights shape {w_arr.shape}.")
        has_constraints = bool(np.any(g_3d >= 0))
    elif w_arr.ndim == 3 and slot_exclusivity and s_dim > 1:
        # Default 3D slot exclusivity: at most one child of parent `u` may occupy `(u, s)`
        g_3d = np.empty((n, n, s_dim), dtype=np.int32)
        for u in range(n):
            for s in range(s_dim):
                g_3d[u, :, s] = u * s_dim + s
        has_constraints = True
    else:
        g_3d = np.full((n, n, s_dim), -1, dtype=np.int32)

    return w_3d, g_3d, active_mask, has_constraints


def _build_result(
    parents: np.ndarray,
    slots: np.ndarray,
    score: float,
    root: int,
) -> ArborescenceResult:
    """Package parent and slot arrays into an `ArborescenceResult`."""
    n = len(parents)
    edge_list: list[tuple[int, int, int]] = []
    for v in range(n):
        u = int(parents[v])
        if v != root and u >= 0:
            s = int(slots[v]) if slots[v] >= 0 else 0
            edge_list.append((u, v, s))
    edges = np.array(edge_list, dtype=np.int32).reshape(-1, 3) if edge_list else np.empty((0, 3), dtype=np.int32)
    return ArborescenceResult(
        parents=parents.astype(np.int32),
        score=float(score),
        slots=slots.astype(np.int32),
        edges=edges,
    )


def maximum_spanning_arborescence(
    weights: np.ndarray,
    root: int = 0,
    *,
    edge_groups: np.ndarray | None = None,
    required_nodes: np.ndarray | None = None,
    slot_exclusivity: bool = True,
) -> ArborescenceResult:
    r"""Find the maximum-weight directed spanning arborescence rooted at `root`.

    Uses Chu-Liu / Edmonds' optimum branching algorithm in $O(N^2)$ time when unconstrained,
    and automatically switches to compiled Numba branch-and-bound when slot/valence
    exclusivity constraints (`edge_groups` or 3D `(N, N, S)` tensors with `slot_exclusivity=True`)
    are active.

    Args:
        weights: 2D array `(N, N)` where `weights[u, v]` is the score of directed edge `u -> v`,
            or 3D array `(N, N, S)` where `weights[u, v, s]` is the score of `u -> v` via slot `s`.
        root: Index of the root vertex (`0 <= root < N`). Defaults to `0`.
        edge_groups: Optional integer array of shape matching `weights` assigning exclusivity group IDs
            (`>= 0`). At most one edge from each non-negative group ID may be selected. (`-1` = unconstrained).
        required_nodes: Optional boolean mask `(N,)` or 1D index array specifying which vertices
            must be spanned by the arborescence. Defaults to all `N` vertices.
        slot_exclusivity: If `True` (default) and `weights` is 3D `(N, N, S)` with `edge_groups=None`,
            enforces that at most one outgoing edge from parent `u` may use slot `s`.

    Returns:
        ArborescenceResult: Result object supporting both `parents, score = maximum_spanning_arborescence(...)`
        unpacking and `.parents`, `.slots`, `.edges`, `.score` attribute access.
    """
    w_3d, g_3d, active_mask, has_constraints = _normalize_arborescence_inputs(
        weights, root, edge_groups, required_nodes, slot_exclusivity
    )
    n, _, _ = w_3d.shape
    if n == 0:
        return _build_result(np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), 0.0, 0)
    if n == 1:
        return _build_result(np.array([-1], dtype=np.int32), np.array([-1], dtype=np.int32), 0.0, 0)

    if not has_constraints and bool(np.all(active_mask != 0)):
        best_slots = np.argmax(w_3d, axis=2).astype(np.int32)
        w_2d = np.max(w_3d, axis=2)
        parents, score = _edmonds_msa_2d(w_2d, int(root))
        slots = np.full(n, -1, dtype=np.int32)
        for v in range(n):
            if v != root and parents[v] >= 0:
                slots[v] = best_slots[parents[v], v]
        return _build_result(parents, slots, score, root)

    out_p, out_s, out_sc, found = _branch_and_bound_k_best(w_3d, g_3d, active_mask, int(root), 1)
    if found == 0:
        return _build_result(np.full(n, -1, dtype=np.int32), np.full(n, -1, dtype=np.int32), float(_NEG_INF), root)
    return _build_result(out_p[0], out_s[0], float(out_sc[0]), root)


def minimum_spanning_arborescence(
    costs: np.ndarray,
    root: int = 0,
    *,
    edge_groups: np.ndarray | None = None,
    required_nodes: np.ndarray | None = None,
    slot_exclusivity: bool = True,
) -> ArborescenceResult:
    r"""Find the minimum-cost directed spanning arborescence rooted at `root`.

    Args:
        costs: 2D `(N, N)` or 3D `(N, N, S)` array of non-negative/real edge costs (use `np.inf` for
            forbidden edges).
        root: Index of the root vertex (`0 <= root < N`). Defaults to `0`.
        edge_groups: Optional exclusivity group IDs (`-1` = unconstrained, `>= 0` = at most one edge per group).
        required_nodes: Optional boolean mask or index array of vertices to span.
        slot_exclusivity: Enforce `(parent, slot)` exclusivity on 3D inputs when `True`.

    Returns:
        ArborescenceResult: Minimum-cost arborescence (`parents, min_cost = minimum_spanning_arborescence(...)`).
    """
    c_arr = np.asarray(costs, dtype=np.float64)
    neg_weights = np.where(np.isinf(c_arr) & (c_arr > 0), _NEG_INF, -c_arr)
    res = maximum_spanning_arborescence(
        neg_weights,
        root=root,
        edge_groups=edge_groups,
        required_nodes=required_nodes,
        slot_exclusivity=slot_exclusivity,
    )
    min_cost = -res.score if res.score > _NEG_INF * 0.1 else float("inf")
    return ArborescenceResult(
        parents=res.parents,
        score=min_cost,
        slots=res.slots,
        edges=res.edges,
    )


def k_best_arborescences(
    weights: np.ndarray,
    root: int = 0,
    k: int = 5,
    *,
    edge_groups: np.ndarray | None = None,
    required_nodes: np.ndarray | None = None,
    slot_exclusivity: bool = True,
) -> list[ArborescenceResult]:
    r"""Enumerate the top-`k` highest-scoring valid directed spanning arborescences in descending order.

    Args:
        weights: 2D `(N, N)` or 3D `(N, N, S)` array of directed edge scores `u -> v`.
        root: Index of the root vertex (`0 <= root < N`). Defaults to `0`.
        k: Number of top arborescences to return (`k >= 1`). Defaults to `5`.
        edge_groups: Optional integer array of exclusivity group IDs (`>= 0`).
        required_nodes: Optional boolean mask or 1D index array of vertices that must be spanned.
        slot_exclusivity: If `True` (default) and `weights` is 3D, enforces `(parent, slot)` exclusivity.

    Returns:
        list[ArborescenceResult]: Up to `k` valid arborescences ordered from highest to lowest score.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}.")
    w_3d, g_3d, active_mask, _ = _normalize_arborescence_inputs(
        weights, root, edge_groups, required_nodes, slot_exclusivity
    )
    n = w_3d.shape[0]
    if n == 0:
        return [_build_result(np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), 0.0, 0)]
    if n == 1:
        return [_build_result(np.array([-1], dtype=np.int32), np.array([-1], dtype=np.int32), 0.0, 0)]

    out_p, out_s, out_sc, found = _branch_and_bound_k_best(w_3d, g_3d, active_mask, int(root), int(k))
    return [_build_result(out_p[i], out_s[i], float(out_sc[i]), root) for i in range(found)]
