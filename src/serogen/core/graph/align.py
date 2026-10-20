"""Generic bipartite graph alignment framework with pluggable cost strategies and topological embeddings."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_array, diags

from serogen.core.graph.models import GraphModel


# Configuration Dataclass ----------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AlignmentConfig:
    """Immutable configuration for bipartite graph alignment solvers."""

    iterations: int = 2
    topo_weight: float = 1.0
    penalty_unmatched: float = 1e6
    as_bipartite: bool = True
    symmetrize_embedding: bool = False

    @classmethod
    def default(cls) -> Self:
        """Return default `AlignmentConfig`."""
        return cls()

    @classmethod
    def rigid(cls) -> Self:
        """Strict alignment prioritizing rigid semantic constraints over topology."""
        return cls(iterations=1, topo_weight=0.2, penalty_unmatched=1e8)

    @classmethod
    def topological(cls) -> Self:
        """Relaxed alignment prioritizing topological graph flow smoothing."""
        return cls(iterations=4, topo_weight=2.5, penalty_unmatched=1e5)


# Embedding Strategies -------------------------------------------------------------------------------------------------
class EmbeddingStrategy(ABC):
    """Abstract base strategy for computing continuous topological node embeddings."""

    @abstractmethod
    def compute(
        self,
        adj: csr_array,
        features: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Compute node embeddings from sparse adjacency and initial node features."""
        ...


@dataclass(frozen=True, slots=True)
class WeisfeilerLehmanEmbedding(EmbeddingStrategy):
    """Computes generalized continuous Weisfeiler-Lehman topological smoothing."""

    iterations: int = 2
    smoothing_factor: float = 0.5

    def compute(
        self,
        adj: csr_array,
        features: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Smooth continuous features across sparse graph adjacency."""
        n_nodes = adj.shape[0]
        if n_nodes == 0:
            return features

        deg = np.asarray(adj.sum(axis=1), dtype=np.float64).ravel()
        deg[deg == 0] = 1.0
        deg_inv = diags(1.0 / deg)
        adj_norm = deg_inv @ adj

        embeddings = [features]
        current_h = features
        alpha = self.smoothing_factor
        for _ in range(self.iterations):
            current_h = (1.0 - alpha) * current_h + alpha * (adj_norm @ current_h)
            embeddings.append(current_h)

        return np.concatenate(embeddings, axis=1)


@dataclass(frozen=True, slots=True)
class DegreeProfileEmbedding(EmbeddingStrategy):
    """Computes in-degree, out-degree, and total degree profiles for each node."""

    def compute(
        self,
        adj: csr_array,
        features: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Augment node features with in-degree and out-degree computed from sparse adjacency."""
        n_nodes = adj.shape[0]
        if n_nodes == 0:
            return features
        out_deg = np.asarray(adj.sum(axis=1), dtype=np.float64).ravel()[:, None]
        in_deg = np.asarray(adj.sum(axis=0), dtype=np.float64).ravel()[:, None]
        return np.column_stack([features, out_deg, in_deg])


# Semantic Cost Strategies ---------------------------------------------------------------------------------------------
class CostStrategy(ABC):
    """Abstract base strategy for computing semantic node compatibility cost matrices."""

    @abstractmethod
    def compute_cost(
        self,
        labels_a: np.ndarray[tuple[int], np.dtype[Any]],
        labels_b: np.ndarray[tuple[int], np.dtype[Any]],
        penalty: float = 1e6,
        graph_a: GraphModel | None = None,
        graph_b: GraphModel | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Compute `(N, M)` pairwise node compatibility cost matrix."""
        ...


@dataclass(frozen=True, slots=True)
class ExactMatchCostStrategy(CostStrategy):
    """Isomorphic cost model: 0.0 for identical node types, infinite penalty for mismatches."""

    def compute_cost(
        self,
        labels_a: np.ndarray[tuple[int], np.dtype[Any]],
        labels_b: np.ndarray[tuple[int], np.dtype[Any]],
        penalty: float = 1e6,
        graph_a: GraphModel | None = None,
        graph_b: GraphModel | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Compute exact equality cost matrix using vectorized broadcasting."""
        if len(labels_a) == 0 or len(labels_b) == 0:
            return np.empty((len(labels_a), len(labels_b)), dtype=np.float64)
        return np.where(labels_a[:, None] == labels_b[None, :], 0.0, np.inf)


@dataclass(frozen=True, slots=True)
class CallableCostStrategy(CostStrategy):
    """Adapter wrapping custom user-defined cost functions."""

    fn: Callable[..., np.ndarray[tuple[int, int], np.dtype[np.float64]]]

    def compute_cost(
        self,
        labels_a: np.ndarray[tuple[int], np.dtype[Any]],
        labels_b: np.ndarray[tuple[int], np.dtype[Any]],
        penalty: float = 1e6,
        graph_a: GraphModel | None = None,
        graph_b: GraphModel | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Delegate cost computation to wrapped callable."""
        return np.asarray(self.fn(labels_a, labels_b), dtype=np.float64)


# Rich Alignment Result Object -----------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Alignment:
    """Outcome of bipartite graph alignment containing matched node pairs, costs, and unmatched nodes."""

    matches: dict[int, int]
    costs: dict[int, float]
    total_cost: float
    graph_a: GraphModel
    graph_b: GraphModel
    unmatched_a: tuple[int, ...]
    unmatched_b: tuple[int, ...]


# Alignment Solver Classes ---------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class GraphAligner:
    """Configurable bipartite graph alignment solver using Hungarian assignment."""

    cost_strategy: CostStrategy = ExactMatchCostStrategy()
    embedding_strategy: EmbeddingStrategy = WeisfeilerLehmanEmbedding()
    config: AlignmentConfig = AlignmentConfig.default()

    def align(self, graph_a: GraphModel, graph_b: GraphModel) -> Alignment:
        """Execute bipartite graph alignment between two `GraphModel` instances."""
        is_sym = self.config.symmetrize_embedding
        adj_a, labels_a = graph_a.as_graph(as_bipartite=self.config.as_bipartite, symmetrize=is_sym)
        adj_b, labels_b = graph_b.as_graph(as_bipartite=self.config.as_bipartite, symmetrize=is_sym)

        n_a, n_b = len(labels_a), len(labels_b)
        if n_a == 0 or n_b == 0:
            return Alignment(
                matches={},
                costs={},
                total_cost=0.0,
                graph_a=graph_a,
                graph_b=graph_b,
                unmatched_a=tuple(range(n_a)),
                unmatched_b=tuple(range(n_b)),
            )

        c_base = self.cost_strategy.compute_cost(
            labels_a,
            labels_b,
            penalty=self.config.penalty_unmatched,
            graph_a=graph_a,
            graph_b=graph_b,
        )

        deg_a = np.asarray(adj_a.sum(axis=1), dtype=np.float64).ravel()[:, None]
        deg_b = np.asarray(adj_b.sum(axis=1), dtype=np.float64).ravel()[:, None]

        emb_a = self.embedding_strategy.compute(adj_a, deg_a)
        emb_b = self.embedding_strategy.compute(adj_b, deg_b)

        c_topo = np.linalg.norm(emb_a[:, None, :] - emb_b[None, :, :], axis=2) * self.config.topo_weight

        c_total = c_base + c_topo
        c_safe = np.where(np.isinf(c_total), self.config.penalty_unmatched, c_total)

        row_ind, col_ind = linear_sum_assignment(c_safe)

        matches: dict[int, int] = {}
        costs: dict[int, float] = {}
        total_cost = 0.0

        for r, c in zip(row_ind, col_ind, strict=False):
            cost_val = float(c_total[r, c])
            if cost_val < self.config.penalty_unmatched and not np.isinf(cost_val):
                matches[int(r)] = int(c)
                costs[int(r)] = cost_val
                total_cost += cost_val

        unmatched_a = tuple(i for i in range(n_a) if i not in matches)
        unmatched_b = tuple(j for j in range(n_b) if j not in matches.values())

        return Alignment(
            matches=matches,
            costs=costs,
            total_cost=total_cost,
            graph_a=graph_a,
            graph_b=graph_b,
            unmatched_a=unmatched_a,
            unmatched_b=unmatched_b,
        )


class IsomorphicGraphAligner(GraphAligner):
    """Specialized aligner pre-configured for same-domain structural isomorphism matching."""

    def __init__(self, config: AlignmentConfig | None = None) -> None:
        """Initialize `IsomorphicGraphAligner`."""
        super().__init__(
            cost_strategy=ExactMatchCostStrategy(),
            embedding_strategy=WeisfeilerLehmanEmbedding(iterations=3),
            config=config or AlignmentConfig(as_bipartite=False, symmetrize_embedding=False),
        )
