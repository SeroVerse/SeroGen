"""Core graph representations, bitfield packing, and vectorized SoA containers."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, Protocol, Self

import numpy as np
import scipy.linalg as linalg
from numba import njit, prange
from scipy.sparse import csgraph, csr_array
from scipy.stats import wasserstein_distance

from serogen.core.collections import BatchedContainer, ScalarView

# Constants ------------------------------------------------------------------------------------------------------------
TAG_EDGE: int = 1 << 31  # 0x80000000 (Bit 31 discriminator for injected edge nodes)


@njit(cache=True, nogil=True, parallel=True)
def _slice_graph_batch_kernel(
    kept_indices: np.ndarray,
    node_types: np.ndarray,
    node_offsets: np.ndarray,
    edge_sources: np.ndarray,
    edge_targets: np.ndarray,
    edge_types: np.ndarray,
    edge_offsets: np.ndarray,
    new_node_offsets: np.ndarray,
    new_edge_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gather nodes and rebase edges in parallel across selected graphs in a `GraphBatch`."""
    num_kept = len(kept_indices)
    total_nodes = int(new_node_offsets[num_kept])
    total_edges = int(new_edge_offsets[num_kept])

    new_nodes = np.empty(total_nodes, dtype=np.uint32)
    new_sources = np.empty(total_edges, dtype=np.uint32)
    new_targets = np.empty(total_edges, dtype=np.uint32)
    new_types = np.empty(total_edges, dtype=np.uint32)

    for out_i in prange(num_kept):
        orig_i = kept_indices[out_i]
        m_start = int(node_offsets[orig_i])
        m_end = int(node_offsets[orig_i + 1])
        n_nodes = m_end - m_start
        dst_m_start = int(new_node_offsets[out_i])

        for j in range(n_nodes):
            new_nodes[dst_m_start + j] = node_types[m_start + j]

        e_start = int(edge_offsets[orig_i])
        e_end = int(edge_offsets[orig_i + 1])
        n_edges = e_end - e_start
        dst_e_start = int(new_edge_offsets[out_i])
        local_shift = dst_m_start - m_start

        for j in range(n_edges):
            new_sources[dst_e_start + j] = np.uint32(int(edge_sources[e_start + j]) + local_shift)
            new_targets[dst_e_start + j] = np.uint32(int(edge_targets[e_start + j]) + local_shift)
            new_types[dst_e_start + j] = edge_types[e_start + j]

    return new_nodes, new_sources, new_targets, new_types


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class BitField:
    """
    Descriptor defining a bitfield slice (shift, width, mask).
    Provides safe packing, extraction, and bitmask computation.
    """

    shift: int
    width: int

    @property
    def mask(self) -> int:
        """Absolute bitmask positioned at field shift."""
        return ((1 << self.width) - 1) << self.shift

    @property
    def max_val(self) -> int:
        """Maximum unsigned value that fits within this bitfield width."""
        return (1 << self.width) - 1

    def extract(self, val: int) -> int:
        """Extracts and right-shifts this bitfield's value from an integer."""
        return (val >> self.shift) & self.max_val

    def extract_array(self, arr: np.ndarray) -> np.ndarray:
        """Extracts and right-shifts this bitfield's value across a numpy array."""
        return (arr >> self.shift) & self.max_val

    def pack(self, field_val: int) -> int:
        """Validates and shifts field_val into its bit position."""
        if not (0 <= field_val <= self.max_val):
            raise ValueError(f"Value {field_val} exceeds {self.width} bits (max {self.max_val})")
        return (field_val & self.max_val) << self.shift

    def __get__(self, instance: Any, owner: Any = None) -> Any:
        """Descriptor access: returns self on class access, extracted field integer on instance access."""
        if instance is None:
            return self
        return (int(instance) >> self.shift) & self.max_val


class GraphNode(ABC):
    """Abstract base protocol for all core nodes."""

    @property
    @abstractmethod
    def type_id(self) -> int:
        """Semantic integer label used in core adjacency matrices and embeddings."""
        ...

    @property
    def is_edge_node(self) -> bool:
        """Returns True if this node represents an injected edge in a bipartite core."""
        return bool(self.type_id & TAG_EDGE)


class IntNode(int, GraphNode):
    """Base model for core nodes represented as bit-packed primitive integers.

    Inherits from Python's built-in `int` and `GraphNode`, enabling zero-allocation
    integer node semantics across NumPy arrays, SciPy CSR matrices, and core algorithms.
    Hosts BitField member descriptors that automatically extract bit values when accessed
    on instances, while providing bitmasking, range validation, and packing at the class level.
    """

    _fields: dict[str, BitField]
    _total_bits: int = 32

    def __init_subclass__(cls, total_bits: int = 32, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._total_bits = total_bits
        fields: dict[str, BitField] = {}
        occupied_mask = 0

        for attr_name, attr_val in cls.__dict__.items():
            if isinstance(attr_val, BitField):
                name_upper = attr_name.upper()
                if attr_val.shift + attr_val.width > total_bits:
                    raise ValueError(
                        f"{cls.__name__}.{attr_name} (shift={attr_val.shift}, width={attr_val.width}) "
                        f"exceeds total {total_bits} bits."
                    )
                field_mask = attr_val.mask
                if (occupied_mask & field_mask) and not any(
                    f.shift == attr_val.shift and f.width == attr_val.width for f in fields.values()
                ):
                    raise ValueError(
                        f"BitField overlap detected in {cls.__name__} for field '{attr_name}' "
                        f"(shift={attr_val.shift}, width={attr_val.width})."
                    )
                occupied_mask |= field_mask
                fields[name_upper] = attr_val

        cls._fields = fields

    @property
    def type_id(self) -> int:
        return int(self)

    @property
    def is_edge_node(self) -> bool:
        return bool(int(self) & TAG_EDGE)

    @classmethod
    def get_field(cls, field_name: str) -> BitField:
        name = field_name.upper()
        if name not in cls._fields:
            raise AttributeError(f"{cls.__name__} has no BitField named '{field_name}'.")
        return cls._fields[name]

    @classmethod
    def mask(cls, *field_names: str) -> int:
        """Constructs an absolute bitmask covering the specified bitfield names."""
        m = 0
        for name in field_names:
            m |= cls.get_field(name).mask
        return m

    @classmethod
    def mask_excluding(cls, *field_names: str) -> int:
        """Constructs a bitmask excluding the specified bitfield names."""
        full_mask = (1 << cls._total_bits) - 1
        return full_mask & ~cls.mask(*field_names)

    @classmethod
    def pack(cls, **kwargs: int) -> int:
        """Packs keyword field values into a composite integer."""
        val = 0
        for name, field_val in kwargs.items():
            val |= cls.get_field(name).pack(field_val)
        return val

    @classmethod
    def extract(cls, val: int, field_name: str) -> int:
        return cls.get_field(field_name).extract(val)

    @classmethod
    def extract_array(cls, arr: np.ndarray, field_name: str) -> np.ndarray:
        return cls.get_field(field_name).extract_array(arr)


@dataclass(slots=True, frozen=True)
class DirectedEdge(ABC):
    """Abstract base class for all directed core edges."""

    source: int
    target: int
    edge_type: int = 1

    def __post_init__(self) -> None:
        if self.source < 0 or self.target < 0:
            raise ValueError(
                f"DirectedEdge source and target must be non-negative node indices, got source={self.source}, target={self.target}"
            )


class GraphModel(ScalarView, Protocol):
    """
    Protocol and mixin providing core topological algorithms to scalar views representing graphs.
    Requires concrete `@dataclass(slots=True, frozen=True)` classes to provide `nodes` and `edges`.
    """

    @property
    def nodes(self) -> Sequence[Any]:
        """Ordered sequence of graph nodes."""
        ...

    @property
    def edges(self) -> Sequence[DirectedEdge]:
        """Ordered sequence of directed edges connecting `nodes`."""
        ...

    @cache  # noqa: B019
    def as_graph(
        self,
        as_bipartite: bool = False,
        symmetrize: bool = False,
    ) -> tuple[csr_array, np.ndarray[tuple[int], np.dtype[np.uint32]]]:
        """
        Builds a sparse CSR adjacency matrix and node label array.

        All biological and genomic models in KBank are strictly directed graphs.
        By default, returns the canonical directed representation. If a mathematical
        algorithm requires a symmetric input matrix (e.g. Laplacian spectrum calculation
        or bidirectional message passing), specify `symmetrize=True`.

        Args:
            as_bipartite: If True, treats each directed edge as an intermediate node,
                forming a bipartite incidence core where edge weights are 1.
                If False, edges directly connect primary nodes with edge weights equal
                to the integer edge type.
            symmetrize: If True, produces a symmetrized (bidirectional) adjacency matrix projection.

        Returns:
            A tuple of:
                - adj: Sparse CSR adjacency matrix of shape (total_nodes, total_nodes).
                - labels: 1D NumPy array of node labels (`np.ndarray[tuple[int], np.dtype[np.uint32]]`).
        """
        is_directed = not symmetrize

        nodes = self.nodes
        edges = self.edges
        num_primary = len(nodes)
        num_edges = len(edges)
        total_nodes = num_primary + num_edges if as_bipartite else num_primary

        if total_nodes == 0:
            return csr_array((0, 0), dtype=np.uint32), np.empty(0, dtype=np.uint32)

        labels = np.empty(total_nodes, dtype=np.uint32)
        for i, node in enumerate(nodes):
            labels[i] = node.type_id if hasattr(node, "type_id") else int(node)

        if num_edges == 0:
            return csr_array((total_nodes, total_nodes), dtype=np.uint32), labels

        sources = np.fromiter((e.source for e in edges), dtype=np.uint32, count=num_edges)
        targets = np.fromiter((e.target for e in edges), dtype=np.uint32, count=num_edges)
        edge_types = np.fromiter((int(e.edge_type) for e in edges), dtype=np.uint32, count=num_edges)

        if as_bipartite:
            edge_node_ids = np.arange(num_primary, total_nodes, dtype=np.uint32)
            labels[num_primary:total_nodes] = edge_types

            if is_directed:
                rows = np.concatenate([sources, edge_node_ids])
                cols = np.concatenate([edge_node_ids, targets])
                data = np.ones(2 * num_edges, dtype=np.uint32)
            else:
                rows = np.concatenate([sources, edge_node_ids, targets, edge_node_ids])
                cols = np.concatenate([edge_node_ids, sources, edge_node_ids, targets])
                data = np.ones(4 * num_edges, dtype=np.uint32)
        else:
            if is_directed:
                rows = sources
                cols = targets
                data = edge_types
            else:
                rows = np.concatenate([sources, targets])
                cols = np.concatenate([targets, sources])
                data = np.concatenate([edge_types, edge_types])

        adj = csr_array((data, (rows, cols)), shape=(total_nodes, total_nodes), dtype=np.uint32)
        return adj, labels

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def __len__(self) -> int:
        return self.num_nodes

    @cache  # noqa: B019
    def laplacian_spectrum(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """
        Computes the continuous topological spectrum of the core.

        Constructs the unweighted, symmetrized adjacency matrix, computes the
        normalized core Laplacian using :func:`scipy.sparse.csgraph.laplacian`,
        and solves for the sorted eigenvalues using :func:`scipy.linalg.eigvalsh`.

        Returns:
            A 1D sorted NumPy array of eigenvalues (`np.ndarray[tuple[int], np.dtype[np.float64]]`).
        """
        adj, _ = self.as_graph(as_bipartite=False)
        if adj.shape[0] == 0:
            return np.array([], dtype=np.float64)

        adj_unweighted = adj.astype(np.float64)
        adj_unweighted.data.fill(1.0)
        adj_sym = adj_unweighted + adj_unweighted.T
        adj_sym.data.fill(1.0)

        L = csgraph.laplacian(adj_sym, normed=True)
        L_dense = np.asarray(L.toarray() if hasattr(L, "toarray") else L, dtype=np.float64)
        eigenvalues = linalg.eigvalsh(L_dense)
        return np.sort(eigenvalues)

    def spectral_distance(self, other: GraphModel) -> float:
        """
        Computes the topological Wasserstein Distance (Earth Mover's Distance)
        between this core model and another, comparing skeletal shapes.

        Extracts the Laplacian spectra from both graphs and computes their
        Wasserstein-1 distance using :func:`scipy.stats.wasserstein_distance`.

        Args:
            other: Another GraphModel instance to compare against.

        Returns:
            Wasserstein distance between the two normalized Laplacian spectra as a float.
        """
        spec1 = self.laplacian_spectrum()
        spec2 = other.laplacian_spectrum()
        if len(spec1) == 0 or len(spec2) == 0:
            return 0.0 if len(spec1) == len(spec2) else 1.0
        return float(wasserstein_distance(spec1, spec2))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self.nodes)} nodes, {len(self.edges)} edges)"


@dataclass(slots=True, frozen=True)
class GraphBatch[T: GraphModel, S: GraphBatch[Any, Any]](BatchedContainer[T, S]):
    """
    Base Structure-of-Arrays (SoA) container for batched graph datasets.
    Provides block-diagonal CSR adjacency, Wasserstein spectral distance matrices,
    pan-graph compression, and vectorized SoA array slicing and concatenation.
    """

    node_types: np.ndarray[tuple[int], np.dtype[np.uint32]]
    node_offsets: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_sources: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_targets: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_types: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_offsets: np.ndarray[tuple[int], np.dtype[np.uint32]]

    def __len__(self) -> int:
        return len(self.node_offsets) - 1

    @property
    def num_graphs(self) -> int:
        return len(self)

    @property
    def total_nodes(self) -> int:
        return len(self.node_types)

    @property
    def total_edges(self) -> int:
        return len(self.edge_sources)

    @property
    def node_lengths(self) -> np.ndarray[tuple[int], np.dtype[np.uint32]]:
        """Number of nodes in each graph in the batch."""
        return np.diff(self.node_offsets)

    @property
    def edge_lengths(self) -> np.ndarray[tuple[int], np.dtype[np.uint32]]:
        """Number of edges in each graph in the batch."""
        return np.diff(self.edge_offsets)

    @classmethod
    def empty(cls) -> Self:
        """Create an empty 0-length graph batch."""
        return cls(
            node_types=np.empty(0, dtype=np.uint32),
            node_offsets=np.zeros(1, dtype=np.uint32),
            edge_sources=np.empty(0, dtype=np.uint32),
            edge_targets=np.empty(0, dtype=np.uint32),
            edge_types=np.empty(0, dtype=np.uint32),
            edge_offsets=np.zeros(1, dtype=np.uint32),
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        """Concatenate multiple graph batches into a single contiguous batch."""
        batch_list: list[Self] = [b for b in batches if len(b) > 0]
        if not batch_list:
            return cls.empty()
        if len(batch_list) == 1:
            return batch_list[0]

        nodes, n_offs, sources, targets, types, e_offs = cls._concat_soa_base(batch_list)
        return cls(
            node_types=nodes,
            node_offsets=n_offs,
            edge_sources=sources,
            edge_targets=targets,
            edge_types=types,
            edge_offsets=e_offs,
        )

    def get_block_diagonal_graph(self, unweighted: bool = False) -> csr_array:
        """
        Returns a single disconnected CSR core representing the entire batch.

        Args:
            unweighted: If True, sets all non-zero edge weights to 1;
                otherwise retains the integer edge types.

        Returns:
            Block-diagonal sparse CSR adjacency matrix of shape (total_nodes, total_nodes).
        """
        N = len(self.node_types)
        if N == 0 or len(self.edge_sources) == 0:
            return csr_array((N, N), dtype=np.uint32)

        weights = np.ones(len(self.edge_types), dtype=np.uint32) if unweighted else self.edge_types
        return csr_array((weights, (self.edge_sources, self.edge_targets)), shape=(N, N), dtype=np.uint32)

    def spectral_distance_matrix(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """
        Computes the all-vs-all continuous topological distance matrix across the batch.

        Extracts per-core Laplacian spectra from the block-diagonal CSR core via
        :func:`scipy.sparse.csgraph.laplacian` and :func:`scipy.linalg.eigvalsh`, then
        computes pairwise Wasserstein distances using :func:`scipy.stats.wasserstein_distance`.

        Returns:
            A dense symmetric matrix (N x N) of Wasserstein distances (`np.ndarray[tuple[int, int], np.dtype[np.float64]]`).
        """
        N = len(self)
        dist_matrix = np.zeros((N, N), dtype=np.float64)
        if N == 0:
            return dist_matrix

        adj_full = self.get_block_diagonal_graph(unweighted=True)
        spectra = []
        for i in range(N):
            m_start, m_end = self.node_offsets[i], self.node_offsets[i + 1]
            if m_start == m_end:
                spectra.append(np.array([], dtype=np.float64))
                continue

            adj_sub = adj_full[m_start:m_end, m_start:m_end]
            adj_sub = adj_sub + adj_sub.T
            adj_sub.data = np.ones_like(adj_sub.data)

            L = csgraph.laplacian(adj_sub.astype(np.float64), normed=True)
            L_dense = np.asarray(L.toarray() if hasattr(L, "toarray") else L, dtype=np.float64)
            eigenvalues = linalg.eigvalsh(L_dense)
            spectra.append(np.sort(eigenvalues))

        for i in range(N):
            for j in range(i + 1, N):
                if len(spectra[i]) == 0 or len(spectra[j]) == 0:
                    d = 0.0 if len(spectra[i]) == len(spectra[j]) else 1.0
                else:
                    d = float(wasserstein_distance(spectra[i], spectra[j]))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        return dist_matrix

    def as_pan_graph(
        self,
        node_mask: int | None = None,
    ) -> tuple[
        csr_array,
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
    ]:
        """
        Compresses the SoA batch into a unified directed pan-core network.
        Merges identical nodes, records node occurrence frequencies,
        and computes the exact occurrence frequency (count) of each directed transition.

        Pan-graphs in KBank are strictly directed consensus networks representing
        synteny flow and biosynthetic transitions.

        Args:
            node_mask: Optional bitmask to apply to node types before merging.

        Returns:
            A tuple of:
                - adj: Sparse CSR pan-core adjacency matrix of shape (num_pan_nodes, num_pan_nodes)
                  where data values represent transition occurrence counts.
                - pan_nodes: 1D array of pan-node integer labels (`np.ndarray[tuple[int], np.dtype[np.uint32]]`).
                - node_counts: 1D array of frequencies indicating how many original nodes merged into
                  each pan-node (`np.ndarray[tuple[int], np.dtype[np.uint32]]`).
        """
        N = len(self.node_types)
        if N == 0:
            return csr_array((0, 0), dtype=np.uint32), np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint32)

        masked_types = (self.node_types & node_mask) if node_mask is not None else self.node_types
        pan_nodes, inverse_indices, node_counts = np.unique(masked_types, return_inverse=True, return_counts=True)
        num_pan_nodes = len(pan_nodes)

        if len(self.edge_sources) == 0:
            return csr_array((num_pan_nodes, num_pan_nodes), dtype=np.uint32), pan_nodes, node_counts.astype(np.uint32)

        p_src = inverse_indices[self.edge_sources]
        p_tgt = inverse_indices[self.edge_targets]

        edge_pairs = np.column_stack([p_src, p_tgt])
        unique_edges, edge_counts = np.unique(edge_pairs, axis=0, return_counts=True)

        rows = unique_edges[:, 0]
        cols = unique_edges[:, 1]
        data = edge_counts.astype(np.uint32)

        adj = csr_array((data, (rows, cols)), shape=(num_pan_nodes, num_pan_nodes), dtype=np.uint32)
        return adj, pan_nodes, node_counts.astype(np.uint32)

    def _slice_soa_arrays(
        self, kept_indices: np.ndarray[tuple[int], np.dtype[np.integer[Any]]]
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
    ]:
        """
        Vectorized helper to slice the core SoA arrays for a subset of graphs.
        """
        num_kept = len(kept_indices)
        if num_kept == 0:
            return (
                np.array([], dtype=np.uint32),
                np.zeros(1, dtype=np.uint32),
                np.array([], dtype=np.uint32),
                np.array([], dtype=np.uint32),
                np.array([], dtype=np.uint32),
                np.zeros(1, dtype=np.uint32),
            )

        idx = np.ascontiguousarray(kept_indices, dtype=np.int64)
        node_lens = self.node_offsets[idx + 1] - self.node_offsets[idx]
        edge_lens = self.edge_offsets[idx + 1] - self.edge_offsets[idx]

        new_node_offsets = np.empty(num_kept + 1, dtype=np.uint32)
        new_node_offsets[0] = 0
        np.cumsum(node_lens, dtype=np.uint32, out=new_node_offsets[1:])

        new_edge_offsets = np.empty(num_kept + 1, dtype=np.uint32)
        new_edge_offsets[0] = 0
        np.cumsum(edge_lens, dtype=np.uint32, out=new_edge_offsets[1:])

        new_nodes, new_sources, new_targets, new_types = _slice_graph_batch_kernel(
            idx,
            self.node_types,
            self.node_offsets,
            self.edge_sources,
            self.edge_targets,
            self.edge_types,
            self.edge_offsets,
            new_node_offsets,
            new_edge_offsets,
        )
        return (
            new_nodes,
            new_node_offsets,
            new_sources,
            new_targets,
            new_types,
            new_edge_offsets,
        )

    @classmethod
    def _concat_soa_base(
        cls, batches: Sequence[Any]
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
    ]:
        """
        Fast vectorized concatenation of core SoA arrays across multiple batches.
        Shifts node_offsets, edge_offsets, and rebases edge_sources and edge_targets.
        """
        if not batches:
            return (
                np.empty(0, dtype=np.uint32),
                np.zeros(1, dtype=np.uint32),
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.uint32),
                np.zeros(1, dtype=np.uint32),
            )

        new_node_types = np.concatenate([b.node_types for b in batches])
        new_edge_types = np.concatenate([b.edge_types for b in batches])

        total_graphs = sum(len(b) for b in batches)
        new_node_offsets = np.empty(total_graphs + 1, dtype=np.uint32)
        new_node_offsets[0] = 0
        new_edge_offsets = np.empty(total_graphs + 1, dtype=np.uint32)
        new_edge_offsets[0] = 0

        sources_list: list[np.ndarray] = []
        targets_list: list[np.ndarray] = []
        node_shift = 0
        edge_shift = 0
        curr_g = 0

        for b in batches:
            n_g = len(b)
            if n_g > 0:
                new_node_offsets[curr_g + 1 : curr_g + 1 + n_g] = b.node_offsets[1:] + node_shift
                new_edge_offsets[curr_g + 1 : curr_g + 1 + n_g] = b.edge_offsets[1:] + edge_shift
                curr_g += n_g

            if len(b.edge_sources) > 0:
                sources_list.append(b.edge_sources + node_shift)
                targets_list.append(b.edge_targets + node_shift)

            node_shift += len(b.node_types)
            edge_shift += len(b.edge_types)

        new_sources = np.concatenate(sources_list) if sources_list else np.empty(0, dtype=np.uint32)
        new_targets = np.concatenate(targets_list) if targets_list else np.empty(0, dtype=np.uint32)

        return (
            new_node_types,
            new_node_offsets,
            new_sources,
            new_targets,
            new_edge_types,
            new_edge_offsets,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({len(self):,} items | {self.total_nodes:,} nodes, {self.total_edges:,} edges)"
        )
