import pytest

from tinyhnsw.bruteforce import BruteForceIndex
from tinyhnsw.distance import cosine_distance, euclidean_distance


class TestBruteForceIndex:
    def test_query_returns_sorted_results(self) -> None:
        index = BruteForceIndex(distance_fn=euclidean_distance)
        index.add(0, [1.0, 0.0])
        index.add(1, [0.0, 1.0])
        index.add(2, [0.9, 0.1])

        results = index.query([1.0, 0.0], k=2)

        assert len(results) == 2
        assert results[0][0] == 0  # exact match first
        assert results[1][0] == 2  # closest second

    def test_k_greater_than_n(self) -> None:
        index = BruteForceIndex(distance_fn=euclidean_distance)
        index.add(0, [1.0])
        index.add(1, [2.0])

        results = index.query([1.0], k=10)
        assert len(results) == 2

    def test_works_with_cosine_distance(self) -> None:
        index = BruteForceIndex(distance_fn=cosine_distance)
        index.add(0, [1.0, 0.0])
        index.add(1, [1.0, 1.0])

        results = index.query([1.0, 0.0], k=1)
        assert results[0][0] == 0

    def test_len(self) -> None:
        index = BruteForceIndex(distance_fn=euclidean_distance)
        assert len(index) == 0
        index.add(0, [1.0])
        assert len(index) == 1

    def test_duplicate_id(self) -> None:
        index = BruteForceIndex(distance_fn=euclidean_distance)
        index.add(0, [1.0])
        with pytest.raises(ValueError, match="already exist"):
            index.add(0, [2.0])

    def test_dimension_mismatch_on_add(self) -> None:
        index = BruteForceIndex(distance_fn=euclidean_distance)
        index.add(0, [1.0, 2.0])
        with pytest.raises(ValueError, match="dimension"):
            index.add(1, [1.0, 2.0, 3.0])

    def test_dimension_mismatch_on_query(self) -> None:
        index = BruteForceIndex(distance_fn=euclidean_distance)
        index.add(0, [1.0, 2.0])
        with pytest.raises(ValueError, match="dimension"):
            index.query([1.0], k=1)

    def test_query_empty_index(self) -> None:
        index = BruteForceIndex(distance_fn=euclidean_distance)
        with pytest.raises(ValueError, match="empty"):
            index.query([1.0], k=1)

    def test_k_less_than_one(self) -> None:
        index = BruteForceIndex(distance_fn=euclidean_distance)
        index.add(0, [1.0])
        with pytest.raises(ValueError, match="greater than zero"):
            index.query([1.0], k=0)
