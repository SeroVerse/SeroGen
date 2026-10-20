r"""Force-directed and spectral graph layout algorithms with Numba-accelerated kernels.

Provides [`LayoutEngine`][serogen.core.graph.layout.LayoutEngine] for computing 2D and $D$-dimensional
node coordinates for [`GraphModel`][serogen.core.graph.models.GraphModel] instances,
[`GraphBatch`][serogen.core.graph.models.GraphBatch] pan-graphs, and sparse adjacency matrices
using the Fruchterman-Reingold spring-electrical algorithm.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import scipy.linalg as linalg
from numba import njit
from scipy.sparse import csgraph, csr_array, issparse

from serogen.core.graph.models import GraphBatch, GraphModel

type LayoutMode = Literal["fruchterman_reingold", "fr"]
type InitMode = Literal["random", "circular", "spectral"]


# Numba Kernels --------------------------------------------------------------------------------------------------------
@njit(nogil=True, cache=True)
def _fruchterman_reingold_kernel(
    pos: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_w: np.ndarray,
    fixed_mask: np.ndarray,
    k: float,
    iterations: int,
    t0: float,
    gravity: float,
    tol: float,
) -> np.ndarray:
    r"""Numba-compiled Fruchterman-Reingold force-directed layout solver.

    Args:
        pos: Initial node coordinates of shape `(N, dim)` (`float64`).
        edge_u: 1D array of undirected edge source indices (`int32`).
        edge_v: 1D array of undirected edge target indices (`int32`).
        edge_w: 1D array of positive edge weights (`float64`).
        fixed_mask: 1D boolean mask where `True` pins a node at its initial position.
        k: Optimal spring distance constant $k > 0$.
        iterations: Maximum number of simulated annealing iterations.
        t0: Initial temperature limiting maximum per-step displacement.
        gravity: Central gravity attraction coefficient ($g \ge 0$).
        tol: Early convergence threshold on maximum node step displacement.

    Returns:
        Updated `(N, dim)` coordinate array (`float64`).
    """
    n_nodes, dim = pos.shape
    n_edges = len(edge_u)
    if n_nodes <= 1 or iterations <= 0:
        return pos

    coords = pos.copy()
    disp = np.empty((n_nodes, dim), dtype=np.float64)
    delta = np.empty(dim, dtype=np.float64)
    center = np.zeros(dim, dtype=np.float64)

    k_sq = k * k
    inv_k = 1.0 / max(k, 1e-12)
    dt = t0 / float(iterations + 1)
    t = t0

    for _ in range(iterations):
        disp.fill(0.0)

        # Compute current centroid for central gravity pull
        if gravity > 0.0:
            center.fill(0.0)
            for i in range(n_nodes):
                for d in range(dim):
                    center[d] += coords[i, d]
            for d in range(dim):
                center[d] /= float(n_nodes)

        # 1. All-pairs repulsive forces: f_r(dist) = k^2 / dist  =>  vector = delta * (k^2 / dist^2)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                dist_sq = 0.0
                for d in range(dim):
                    diff = coords[i, d] - coords[j, d]
                    delta[d] = diff
                    dist_sq += diff * diff

                if dist_sq < 1e-4:
                    dist_sq = 1e-4

                rep = k_sq / dist_sq
                for d in range(dim):
                    f_d = delta[d] * rep
                    disp[i, d] += f_d
                    disp[j, d] -= f_d

        # 2. Edge attractive forces: f_a(dist) = w * dist^2 / k  =>  vector = delta * (w * dist / k)
        for e in range(n_edges):
            u = edge_u[e]
            v = edge_v[e]
            if u == v:
                continue
            w = edge_w[e]
            dist_sq = 0.0
            for d in range(dim):
                diff = coords[u, d] - coords[v, d]
                delta[d] = diff
                dist_sq += diff * diff

            dist = np.sqrt(dist_sq)
            att = w * dist * inv_k
            for d in range(dim):
                f_d = delta[d] * att
                disp[u, d] -= f_d
                disp[v, d] += f_d

        # 3. Gentle central gravity to keep disconnected components cohesive
        if gravity > 0.0:
            grav_scale = gravity * k
            for i in range(n_nodes):
                for d in range(dim):
                    disp[i, d] -= (coords[i, d] - center[d]) * grav_scale

        # 4. Temperature-clamped position update
        max_step_sq = 0.0
        for i in range(n_nodes):
            if fixed_mask[i]:
                continue
            norm_sq = 0.0
            for d in range(dim):
                norm_sq += disp[i, d] * disp[i, d]

            if norm_sq > 0.0:
                norm = np.sqrt(norm_sq)
                step_scale = (t / norm) if norm > t else 1.0
                step_sq = 0.0
                for d in range(dim):
                    step_d = disp[i, d] * step_scale
                    coords[i, d] += step_d
                    step_sq += step_d * step_d
                if step_sq > max_step_sq:
                    max_step_sq = step_sq

        t -= dt
        if t <= 0.0 or np.sqrt(max_step_sq) < tol:
            break

    return coords


# Helpers --------------------------------------------------------------------------------------------------------------
def rescale_layout(
    pos: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    scale: float = 1.0,
    center: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    r"""Rescale node coordinates so the largest centered coordinate magnitude equals `scale`.

    Args:
        pos: `(N, dim)` coordinate array.
        scale: Target maximum radius around `center`.
        center: Optional `(dim,)` center point. Defaults to origin.

    Returns:
        Rescaled `(N, dim)` coordinate array (`float64`).
    """
    n_nodes, dim = pos.shape
    out = pos.astype(np.float64, copy=True)
    if n_nodes == 0:
        return out

    target_center = np.zeros(dim, dtype=np.float64) if center is None else np.asarray(center, dtype=np.float64)
    if len(target_center) != dim:
        raise ValueError(f"center dimension ({len(target_center)}) must match layout dim ({dim}).")

    if n_nodes == 1:
        out[0] = target_center
        return out

    out -= np.mean(out, axis=0, keepdims=True)
    max_lim = float(np.max(np.abs(out)))
    if max_lim > 0.0:
        out *= float(scale) / max_lim
    out += target_center[None, :]
    return out


# Layout Engine --------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LayoutEngine:
    r"""Configurable graph layout engine with Numba-accelerated Fruchterman-Reingold physics.

    Supports laying out single [`GraphModel`][serogen.core.graph.models.GraphModel] instances,
    [`GraphBatch`][serogen.core.graph.models.GraphBatch] collections (via their unified
    [`.as_pan_graph()`][serogen.core.graph.models.GraphBatch.as_pan_graph] representation),
    or raw SciPy sparse / NumPy adjacency matrices.

    Attributes:
        mode: Layout algorithm (`"fruchterman_reingold"` or shorthand `"fr"`).
        dim: Coordinate dimensionality ($D \ge 1$). Defaults to `2`.
        iterations: Number of simulated annealing iterations. Defaults to `50`.
        k: Optimal inter-node spring distance. If `None`, defaults to $1 / \sqrt[\text{dim}]{N}$.
        scale: Bounding radius for output rescaling (`[-scale, +scale]`), or `None` to disable rescaling.
        center: Optional $D$-dimensional center offset. Defaults to origin.
        gravity: Central gravity coefficient keeping disconnected components compact. Defaults to `0.05`.
        tol: Early stopping threshold on maximum node displacement. Defaults to `1e-4`.
        init: Coordinate initialization strategy (`"random"`, `"circular"`, or `"spectral"`).
        use_weights: Whether edge weights scale attractive forces. Defaults to `True`.
        as_bipartite: Whether to expand edges as bipartite nodes when laying out a `GraphModel`.
        node_mask: Optional bitmask passed to `GraphBatch.as_pan_graph(node_mask=...)`.
        seed: Deterministic random seed for coordinate initialization. Defaults to `42`.
    """

    mode: LayoutMode = "fruchterman_reingold"
    dim: int = 2
    iterations: int = 50
    k: float | None = None
    scale: float | None = 1.0
    center: Sequence[float] | np.ndarray | None = None
    gravity: float = 0.05
    tol: float = 1e-4
    init: InitMode = "random"
    use_weights: bool = True
    as_bipartite: bool = False
    node_mask: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        r"""Validate layout hyperparameters."""
        if self.mode not in ("fruchterman_reingold", "fr"):
            raise ValueError(f"Invalid mode {self.mode!r}. Expected 'fruchterman_reingold' or 'fr'.")
        if self.dim < 1:
            raise ValueError(f"dim must be >= 1, got {self.dim}.")
        if self.iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {self.iterations}.")
        if self.k is not None and self.k <= 0.0:
            raise ValueError(f"k must be > 0 when specified, got {self.k}.")
        if self.gravity < 0.0:
            raise ValueError(f"gravity must be >= 0, got {self.gravity}.")
        if self.tol < 0.0:
            raise ValueError(f"tol must be >= 0, got {self.tol}.")
        if self.init not in ("random", "circular", "spectral"):
            raise ValueError(f"Invalid init strategy {self.init!r}. Expected 'random', 'circular', or 'spectral'.")

    def __call__(
        self,
        graph: GraphModel | GraphBatch[Any, Any] | csr_array | np.ndarray,
        *,
        pos: np.ndarray | None = None,
        fixed: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        r"""Compute layout coordinates for `graph` (alias for [`layout`][serogen.core.graph.layout.LayoutEngine.layout])."""
        return self.layout(graph, pos=pos, fixed=fixed)

    def layout(
        self,
        graph: GraphModel | GraphBatch[Any, Any] | csr_array | np.ndarray,
        *,
        pos: np.ndarray | None = None,
        fixed: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        r"""Compute `(N, dim)` layout coordinates for a graph, batch pan-graph, or adjacency matrix.

        Args:
            graph: Input [`GraphModel`][serogen.core.graph.models.GraphModel],
                [`GraphBatch`][serogen.core.graph.models.GraphBatch] (laid out via `.as_pan_graph()`),
                sparse `csr_array`, or dense 2D adjacency array.
            pos: Optional initial `(N, dim)` node coordinates for warm-starting or pinning.
            fixed: Optional sequence/array of node indices (or boolean mask of length `N`)
                whose positions should remain fixed during optimization.

        Returns:
            2D NumPy array of shape `(N, dim)` with `float64` coordinates.
        """
        adj = self._extract_adjacency(graph)
        n_nodes = int(adj.shape[0])

        center_arr = (
            np.zeros(self.dim, dtype=np.float64) if self.center is None else np.asarray(self.center, dtype=np.float64)
        )
        if len(center_arr) != self.dim:
            raise ValueError(f"center length ({len(center_arr)}) must match dim ({self.dim}).")

        if n_nodes == 0:
            return np.empty((0, self.dim), dtype=np.float64)

        if n_nodes == 1:
            out = np.empty((1, self.dim), dtype=np.float64)
            out[0] = pos[0] if pos is not None else center_arr
            return out

        init_pos = self._init_positions(adj, n_nodes, pos)
        fixed_mask = self._resolve_fixed_mask(n_nodes, fixed, has_pos=pos is not None)
        edge_u, edge_v, edge_w = self._extract_undirected_edges(adj)

        k_eff = float(self.k) if self.k is not None else float(1.0 / (n_nodes ** (1.0 / self.dim)))
        span = np.max(init_pos, axis=0) - np.min(init_pos, axis=0)
        t0 = max(float(np.max(span)) * 0.1, 0.1)
        tol_eff = float(self.tol) * max(float(np.max(span)), 1.0)

        coords = _fruchterman_reingold_kernel(
            init_pos,
            edge_u,
            edge_v,
            edge_w,
            fixed_mask,
            k=k_eff,
            iterations=self.iterations,
            t0=t0,
            gravity=float(self.gravity),
            tol=tol_eff,
        )

        if self.scale is not None and not np.any(fixed_mask):
            return rescale_layout(coords, scale=float(self.scale), center=center_arr)
        if self.center is not None and not np.any(fixed_mask):
            coords = coords - np.mean(coords, axis=0, keepdims=True) + center_arr[None, :]
        return coords

    def _extract_adjacency(
        self,
        graph: GraphModel | GraphBatch[Any, Any] | csr_array | np.ndarray,
    ) -> csr_array:
        r"""Normalize input `GraphModel`, `GraphBatch`, or matrix into a square `csr_array`."""
        if isinstance(graph, GraphBatch):
            adj, _, _ = graph.as_pan_graph(node_mask=self.node_mask)
            return adj
        if hasattr(graph, "as_graph"):
            adj, _ = cast(GraphModel, graph).as_graph(as_bipartite=self.as_bipartite, symmetrize=False)
            return adj
        if issparse(graph):
            csr = csr_array(graph)
            if csr.ndim != 2 or csr.shape[0] != csr.shape[1]:
                raise ValueError(f"Adjacency matrix must be square 2D, got shape {csr.shape}.")
            return csr
        if isinstance(graph, np.ndarray):
            if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
                raise ValueError(f"Adjacency array must be square 2D, got shape {graph.shape}.")
            return csr_array(graph)
        raise TypeError(
            f"Unsupported graph type {type(graph).__name__!r}. "
            "Expected GraphModel, GraphBatch, scipy.sparse.csr_array, or 2D numpy.ndarray."
        )

    def _extract_undirected_edges(
        self,
        adj: csr_array,
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.int32]],
        np.ndarray[tuple[int], np.dtype[np.int32]],
        np.ndarray[tuple[int], np.dtype[np.float64]],
    ]:
        r"""Convert a directed/undirected CSR matrix into unique upper-triangular edges `(u < v, w)`."""
        if adj.nnz == 0:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64),
            )

        adj_f = adj.astype(np.float64)
        if not self.use_weights:
            adj_f.data.fill(1.0)

        sym = csr_array(adj_f + adj_f.T)
        if not self.use_weights and sym.nnz > 0:
            sym.data.fill(1.0)

        coo = sym.tocoo()
        row = np.asarray(coo.row, dtype=np.int32)
        col = np.asarray(coo.col, dtype=np.int32)
        data = np.asarray(coo.data, dtype=np.float64)

        upper = (row < col) & (data > 0.0)
        return (
            np.ascontiguousarray(row[upper], dtype=np.int32),
            np.ascontiguousarray(col[upper], dtype=np.int32),
            np.ascontiguousarray(data[upper], dtype=np.float64),
        )

    def _init_positions(
        self,
        adj: csr_array,
        n_nodes: int,
        pos: np.ndarray | None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        r"""Initialize `(N, dim)` node coordinates from `pos` or `self.init` strategy."""
        rng = np.random.default_rng(self.seed)
        if pos is not None:
            pos_arr = np.asarray(pos, dtype=np.float64)
            if pos_arr.shape != (n_nodes, self.dim):
                raise ValueError(f"pos shape {pos_arr.shape} must match (n_nodes, dim) = ({n_nodes}, {self.dim}).")
            return np.ascontiguousarray(pos_arr, dtype=np.float64)

        if self.init == "circular":
            out = np.zeros((n_nodes, self.dim), dtype=np.float64)
            angles = np.linspace(0.0, 2.0 * np.pi, n_nodes, endpoint=False)
            out[:, 0] = np.cos(angles)
            if self.dim >= 2:
                out[:, 1] = np.sin(angles)
            if self.dim > 2:
                out[:, 2:] = rng.uniform(-0.1, 0.1, size=(n_nodes, self.dim - 2))
            return out

        if self.init == "spectral" and n_nodes > self.dim and adj.nnz > 0:
            adj_sym = adj.astype(np.float64)
            if not self.use_weights:
                adj_sym.data.fill(1.0)
            adj_sym = csr_array(adj_sym + adj_sym.T)
            if not self.use_weights and adj_sym.nnz > 0:
                adj_sym.data.fill(1.0)
            lap = csgraph.laplacian(adj_sym, normed=True)
            lap_dense = np.asarray(lap.toarray() if hasattr(lap, "toarray") else lap, dtype=np.float64)
            _, vecs = linalg.eigh(lap_dense)
            # Skip the trivial constant eigenvector at index 0
            spec = np.ascontiguousarray(vecs[:, 1 : 1 + self.dim], dtype=np.float64)
            # Add tiny deterministic jitter to break exact spectral degeneracies
            spec += rng.normal(0.0, 1e-3, size=spec.shape)
            return rescale_layout(spec, scale=1.0)

        return np.ascontiguousarray(rng.uniform(-1.0, 1.0, size=(n_nodes, self.dim)), dtype=np.float64)

    @staticmethod
    def _resolve_fixed_mask(
        n_nodes: int,
        fixed: Sequence[int] | np.ndarray | None,
        *,
        has_pos: bool,
    ) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        r"""Convert `fixed` indices or boolean array into a 1D boolean mask of length `n_nodes`."""
        mask = np.zeros(n_nodes, dtype=np.bool_)
        if fixed is None:
            return mask
        if not has_pos:
            raise ValueError("fixed nodes require initial coordinates `pos` to be provided.")

        fixed_arr = np.asarray(fixed)
        if fixed_arr.dtype == np.bool_:
            if len(fixed_arr) != n_nodes:
                raise ValueError(f"Boolean fixed mask length ({len(fixed_arr)}) must match n_nodes ({n_nodes}).")
            return fixed_arr.copy()

        indices = fixed_arr.astype(np.int64, copy=False).ravel()
        if len(indices) > 0:
            if np.any((indices < 0) | (indices >= n_nodes)):
                raise IndexError(f"fixed node indices out of bounds for graph with {n_nodes} nodes.")
            mask[indices] = True
        return mask
