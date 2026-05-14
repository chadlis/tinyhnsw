import math

from tinyhnsw.types import Vector


def euclidean_distance(vec1: Vector, vec2: Vector) -> float:
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors have different lengths : {len(vec1)} != {len(vec2)}!")
    if len(vec1) == 0:
        raise ValueError("Vectors are empty!")

    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2, strict=True)))


def cosine_distance(vec1: Vector, vec2: Vector) -> float:
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors have different lengths : {len(vec1)} != {len(vec2)}!")
    if len(vec1) == 0:
        raise ValueError("Vectors are empty!")
    if all(v == 0 for v in vec1):
        raise ValueError("The first vector is the null vector!")
    if all(v == 0 for v in vec2):
        raise ValueError("The second vector is the null vector!")

    return 1 - (
        _dot_product(vec1, vec2)
        / (math.sqrt(_dot_product(vec1, vec1)) * math.sqrt(_dot_product(vec2, vec2)))
    )


def _dot_product(vec1: Vector, vec2: Vector) -> float:
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors have different lengths : {len(vec1)} != {len(vec2)}!")
    if len(vec1) == 0:
        raise ValueError("Vectors are empty!")

    return sum((a * b for a, b in zip(vec1, vec2, strict=True)))
