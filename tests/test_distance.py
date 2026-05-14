import math

import pytest

from tinyhnsw.distance import cosine_distance, euclidean_distance


class TestEuclideanDistance:
    def test_identical_vectors(self) -> None:
        assert euclidean_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_known_value(self) -> None:
        assert euclidean_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(math.sqrt(2))

    def test_different_lengths(self) -> None:
        with pytest.raises(ValueError, match="different lengths"):
            euclidean_distance([1.0], [1.0, 2.0])

    def test_empty_vectors(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            euclidean_distance([], [])


class TestCosineDistance:
    def test_identical_vectors(self) -> None:
        assert cosine_distance([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)

    def test_orthogonal_vectors(self) -> None:
        assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)

    def test_opposite_vectors(self) -> None:
        assert cosine_distance([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0)

    def test_collinear_vectors(self) -> None:
        assert cosine_distance([1.0, 2.0], [2.0, 4.0]) == pytest.approx(0.0)

    def test_different_lengths(self) -> None:
        with pytest.raises(ValueError, match="different lengths"):
            cosine_distance([1.0], [1.0, 2.0])

    def test_empty_vectors(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            cosine_distance([], [])

    def test_first_null_vector(self) -> None:
        with pytest.raises(ValueError, match="null vector"):
            cosine_distance([0.0, 0.0], [1.0, 2.0])

    def test_second_null_vector(self) -> None:
        with pytest.raises(ValueError, match="null vector"):
            cosine_distance([1.0, 2.0], [0.0, 0.0])
