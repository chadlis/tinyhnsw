from tinyhnsw.types import DistanceFn, Vector


class BruteForceIndex:
    def __init__(self, distance_fn: DistanceFn):
        self._distance_fn = distance_fn
        self._vectors: dict[int, Vector] = {}
        self._dimension: int | None = None

    def add(self, id: int, vector: Vector) -> None:
        if id in self._vectors:
            raise ValueError(f"A vector with id '{id}' already exist!")
        if not self._vectors:
            self._dimension = len(vector)
        else:
            if len(vector) != self._dimension:
                raise ValueError(
                    f"Vector dimension {len(vector)} should match existing "
                    f"vectors dimension {self._dimension}"
                )
        self._vectors[id] = vector

    def query(self, vector: Vector, k: int) -> list[tuple[int, float]]:
        if not self._vectors:
            raise ValueError("Index is empty!")
        if len(vector) != self._dimension:
            raise ValueError(
                f"Query vector dimension {len(vector)} "
                f"should match index vectors dimension {self._dimension}"
            )
        if k < 1:
            raise ValueError(f"k='{k}' should be greater than zero")
        distances: dict[int, float] = {
            id: self._distance_fn(vector, v) for id, v in self._vectors.items()
        }
        return sorted(distances.items(), key=lambda item: item[1])[:k]

    def __len__(self) -> int:
        return len(self._vectors)
