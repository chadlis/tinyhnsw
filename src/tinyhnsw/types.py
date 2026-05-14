from collections.abc import Callable

Vector = list[float]
DistanceFn = Callable[[Vector, Vector], float]
