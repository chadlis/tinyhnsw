import random

import pytest

from tinyhnsw.bruteforce import BruteForceIndex
from tinyhnsw.distance import cosine_distance, euclidean_distance
from tinyhnsw.layer import Layer
from tinyhnsw.types import Node


def _build_random_layer(
    n: int,
    dim: int,
    m: int,
    ef_construction: int,
    seed: int = 42,
) -> tuple[Layer, dict[int, Node], dict[int, list[float]]]:
    rng = random.Random(seed)
    vectors = {i: [rng.uniform(-1.0, 1.0) for _ in range(dim)] for i in range(n)}
    layer = Layer(distance_fn=euclidean_distance, m=m, ef_construction=ef_construction)
    nodes: dict[int, Node] = {}
    for i, v in vectors.items():
        node = Node(id=i, vector=v, level=0)
        layer.insert(node)
        nodes[i] = node
    return layer, nodes, vectors


class TestLayerConstruction:
    def test_invalid_m(self) -> None:
        with pytest.raises(ValueError):
            Layer(distance_fn=euclidean_distance, m=0, ef_construction=10)

    def test_invalid_ef_construction(self) -> None:
        with pytest.raises(ValueError):
            Layer(distance_fn=euclidean_distance, m=5, ef_construction=0)


class TestLayerSearchValidations:
    def test_empty_layer_raises(self) -> None:
        layer = Layer(distance_fn=euclidean_distance, m=5, ef_construction=10)
        with pytest.raises(ValueError, match="empty"):
            layer.search([1.0, 0.0], k=1, ef=10)

    def test_k_less_than_one_raises(self) -> None:
        layer = Layer(distance_fn=euclidean_distance, m=5, ef_construction=10)
        layer.insert(Node(id=0, vector=[1.0, 0.0], level=0))
        with pytest.raises(ValueError, match="k"):
            layer.search([1.0, 0.0], k=0, ef=10)

    def test_ef_less_than_k_raises(self) -> None:
        layer = Layer(distance_fn=euclidean_distance, m=5, ef_construction=10)
        layer.insert(Node(id=0, vector=[1.0, 0.0], level=0))
        with pytest.raises(ValueError, match="ef"):
            layer.search([1.0, 0.0], k=5, ef=2)


class TestLayerInsert:
    def test_duplicate_id_raises(self) -> None:
        layer = Layer(distance_fn=euclidean_distance, m=5, ef_construction=10)
        layer.insert(Node(id=0, vector=[1.0, 0.0], level=0))
        with pytest.raises(ValueError):
            layer.insert(Node(id=0, vector=[0.0, 1.0], level=0))


class TestLayerSearch:
    def test_single_node_search_returns_that_node(self) -> None:
        layer = Layer(distance_fn=euclidean_distance, m=5, ef_construction=10)
        layer.insert(Node(id=42, vector=[1.0, 2.0], level=0))

        results = layer.search([1.0, 2.0], k=1, ef=10)

        assert len(results) == 1
        assert results[0][0] == 42
        assert results[0][1] == pytest.approx(0.0)

    def test_results_sorted_ascending(self) -> None:
        layer, _, _ = _build_random_layer(n=20, dim=4, m=6, ef_construction=16)

        results = layer.search([0.1, 0.2, 0.3, 0.4], k=10, ef=16)

        distances = [d for _, d in results]
        assert distances == sorted(distances)

    def test_works_with_cosine_distance(self) -> None:
        layer = Layer(distance_fn=cosine_distance, m=3, ef_construction=10)
        layer.insert(Node(id=0, vector=[1.0, 0.0], level=0))
        layer.insert(Node(id=1, vector=[0.0, 1.0], level=0))
        layer.insert(Node(id=2, vector=[0.9, 0.1], level=0))

        top = layer.search([1.0, 0.0], k=1, ef=10)[0]
        assert top[0] == 0


class TestLayerInvariants:
    def test_edges_are_bidirectional(self) -> None:
        _, nodes, _ = _build_random_layer(n=15, dim=4, m=5, ef_construction=12)

        for node in nodes.values():
            for neighbor_id in node.neighbors[0]:
                assert node.id in nodes[neighbor_id].neighbors[0], (
                    f"edge ({node.id} -> {neighbor_id}) is not symmetric"
                )

    def test_m_cap_is_respected_after_many_inserts(self) -> None:
        m = 4
        _, nodes, _ = _build_random_layer(n=30, dim=4, m=m, ef_construction=16)

        for node in nodes.values():
            assert len(node.neighbors[0]) <= m, (
                f"node {node.id} has {len(node.neighbors[0])} neighbors, exceeds m={m}"
            )

    def test_recall_matches_brute_force_on_small_graph(self) -> None:
        n, dim = 40, 4
        layer, _, vectors = _build_random_layer(n=n, dim=dim, m=8, ef_construction=20)

        brute = BruteForceIndex(distance_fn=euclidean_distance)
        for i, v in vectors.items():
            brute.add(i, v)

        rng = random.Random(123)
        queries = [[rng.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(20)]
        matches = sum(
            1 for q in queries if layer.search(q, k=1, ef=20)[0][0] == brute.query(q, k=1)[0][0]
        )
        assert matches >= 18  # >= 90% recall on small graph with ef >= m
