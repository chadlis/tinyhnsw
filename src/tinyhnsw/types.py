from collections.abc import Callable
from dataclasses import dataclass

Vector = list[float]
DistanceFn = Callable[[Vector, Vector], float]


@dataclass
class Node:
    id: int
    vector: Vector
    level: int

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError(f"level {self.level} should be non negative")
        self.neighbors: dict[int, set[int]] = {lev: set() for lev in range(self.level + 1)}
