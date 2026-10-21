"""Unit tests for `LayoutEngine` and Fruchterman-Reingold force-directed layout in `serogen.core.graph.layout`."""

from dataclasses import dataclass
from typing import Any, Self, overload

import numpy as np
import pytest
from scipy.sparse import csr_array

from serogen.core.graph.layout import LayoutEngine, rescale_layout
from serogen.core.graph.models import DirectedEdge, GraphBatch, GraphModel, IntNode


class _MockNode(IntNode):
    pass


@dataclass(slots=True, frozen=True)
class _MockEdge(DirectedEdge):
    pass


@dataclass(slots=True, frozen=True)
class _MockGraph(GraphModel):
    nodes: tuple[_MockNode, ...]
    edges: tuple[_MockEdge, ...]


@dataclass(slots=True, frozen=True)
class _MockGraphBatch(GraphBatch[_MockGraph, "_MockGraphBatch"]):
    def _get_single_item(self, idx: int) -> _MockGraph:
        n_s, n_e = int(self.node_offsets[idx]), int(self.node_offsets[idx + 1])
        e_s, e_e = int(self.edge_offsets[idx]), int(self.edge_offsets[idx + 1])
        nodes = tuple(_MockNode(int(x)) for x in self.node_types[n_s:n_e])
        edges = tuple(
            _MockEdge(
                source=int(self.edge_sources[k]) - n_s,
                target=int(self.edge_targets[k]) - n_s,
                edge_type=int(self.edge_types[k]),
            )
            for k in range(e_s, e_e)
        )
        return _MockGraph(nodes=nodes, edges=edges)

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> _MockGraph: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> Self: ...

    @overload
    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> _MockGraph | Self: ...

    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> _MockGraph | Self:
        if isinstance(item, int | np.integer):
            return self._get_single_item(int(item))
        return type(self)(*self._slice_soa_arrays(np.arange(len(self))[item]))


def test_layout_engine_graph_model_and_init_modes() -> None:
    """Verify `LayoutEngine` on a `GraphModel` across random, circular, and spectral initializations."""
    nodes = tuple(_MockNode(i) for i in range(6))
    edges = (
        _MockEdge(0, 1, 1),
        _MockEdge(1, 2, 1),
        _MockEdge(2, 0, 1),
        _MockEdge(2, 3, 1),
        _MockEdge(3, 4, 1),
        _MockEdge(4, 5, 1),
        _MockEdge(5, 3, 1),
    )
    graph = _MockGraph(nodes=nodes, edges=edges)

    for init_mode in ("random", "circular", "spectral"):
        engine = LayoutEngine(mode="fr", dim=2, iterations=80, init=init_mode, scale=2.0, seed=123)
        coords = engine(graph)
        assert coords.shape == (6, 2)
        assert coords.dtype == np.float64
        assert np.all(np.isfinite(coords))
        assert np.isclose(np.max(np.abs(coords)), 2.0, atol=1e-6)

        # Intra-triangle distances (0,1,2) should on average be shorter than cross-triangle distances (0 vs 5)
        d_intra = float(np.linalg.norm(coords[0] - coords[1]))
        d_cross = float(np.linalg.norm(coords[0] - coords[5]))
        assert d_intra < d_cross


def test_layout_engine_graph_batch_pan_graph() -> None:
    """Verify `LayoutEngine` lays out `GraphBatch` instances via `.as_pan_graph()` by default."""
    batch = _MockGraphBatch(
        node_types=np.array([10, 20, 30, 10, 20, 40], dtype=np.uint32),
        node_offsets=np.array([0, 3, 6], dtype=np.uint32),
        edge_sources=np.array([0, 1, 3, 4], dtype=np.uint32),
        edge_targets=np.array([1, 2, 4, 5], dtype=np.uint32),
        edge_types=np.array([1, 1, 1, 1], dtype=np.uint32),
        edge_offsets=np.array([0, 2, 4], dtype=np.uint32),
    )
    # Unique pan-nodes are {10, 20, 30, 40} -> 4 nodes
    engine = LayoutEngine(dim=2, iterations=60, seed=42)
    coords = engine.layout(batch)
    assert coords.shape == (4, 2)
    assert np.all(np.isfinite(coords))


def test_layout_engine_weighted_attraction_and_fixed_nodes() -> None:
    """Verify edge weights pull strongly-coupled nodes closer and `fixed` pins specified nodes."""
    # Star-like 3-node graph: 0--1 has weight 50, 1--2 has weight 1
    adj = csr_array(
        np.array(
            [
                [0.0, 50.0, 0.0],
                [50.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    engine_w = LayoutEngine(iterations=100, use_weights=True, scale=None, seed=7)
    coords_w = engine_w(adj)
    d_strong = float(np.linalg.norm(coords_w[0] - coords_w[1]))
    d_weak = float(np.linalg.norm(coords_w[1] - coords_w[2]))
    assert d_strong < d_weak

    # Fixed node test
    init_pos = np.array([[-1.0, 0.0], [0.2, 0.5], [1.0, 0.0]], dtype=np.float64)
    pinned = engine_w.layout(adj, pos=init_pos, fixed=[0, 2])
    np.testing.assert_allclose(pinned[0], init_pos[0])
    np.testing.assert_allclose(pinned[2], init_pos[2])
    assert not np.allclose(pinned[1], init_pos[1])


def test_layout_engine_edge_cases_and_validation() -> None:
    """Verify 0-node, 1-node, 3D layouts, `rescale_layout`, and parameter validation."""
    engine_3d = LayoutEngine(dim=3, center=(1.0, 2.0, 3.0), scale=1.5)
    empty_coords = engine_3d(csr_array((0, 0), dtype=np.float64))
    assert empty_coords.shape == (0, 3)

    single_coords = engine_3d(csr_array((1, 1), dtype=np.float64))
    np.testing.assert_allclose(single_coords, np.array([[1.0, 2.0, 3.0]]))

    rescaled = rescale_layout(np.array([[-2.0, 0.0], [2.0, 0.0]]), scale=5.0, center=np.array([1.0, -1.0]))
    np.testing.assert_allclose(rescaled, np.array([[-4.0, -1.0], [6.0, -1.0]]))

    invalid_mode: Any = "invalid"
    with pytest.raises(ValueError, match="Invalid mode"):
        LayoutEngine(mode=invalid_mode)

    with pytest.raises(ValueError, match="fixed nodes require initial coordinates"):
        engine_3d.layout(np.zeros((2, 2)), fixed=[0])
