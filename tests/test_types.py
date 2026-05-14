import pytest

from tinyhnsw.types import Node


class TestNode:
    def test_neighbors_initialized_per_level(self) -> None:
        node = Node(id=0, vector=[1.0, 0.0], level=2)
        assert set(node.neighbors.keys()) == {0, 1, 2}
        assert all(s == set() for s in node.neighbors.values())

    def test_level_zero_has_one_layer(self) -> None:
        node = Node(id=0, vector=[1.0], level=0)
        assert set(node.neighbors.keys()) == {0}

    def test_negative_level(self) -> None:
        with pytest.raises(ValueError, match="non negative"):
            Node(id=0, vector=[1.0], level=-1)

    def test_add_neighbor(self) -> None:
        node = Node(id=0, vector=[1.0], level=1)
        node.neighbors[0].add(3)
        node.neighbors[1].add(5)
        assert node.neighbors[0] == {3}
        assert node.neighbors[1] == {5}
