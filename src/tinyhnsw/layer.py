import heapq

from tinyhnsw.types import DistanceFn, Node, Vector


class Layer:
    def __init__(self, distance_fn: DistanceFn, m: int, ef_construction: int):
        if m < 1:
            raise ValueError(f"m={m} should be greater than zero")
        if ef_construction < 1:
            raise ValueError(f"ef_construction={ef_construction} should be greater than zero")
        self.distance_fn = distance_fn
        self.m = m
        self.ef_construction = ef_construction
        self.nodes: dict[int, Node] = {}
        self.entry_point: Node | None = None

    def insert(self, node: Node) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Node with id {node.id} already exists!")

        self.nodes[node.id] = node

        if self.entry_point is None:
            self.entry_point = node
            return

        neighbors = self.search(node.vector, k=self.m, ef=self.ef_construction)

        for neighbor_id, _ in neighbors:
            node.neighbors[0].add(neighbor_id)
            self.nodes[neighbor_id].neighbors[0].add(node.id)

            if len(self.nodes[neighbor_id].neighbors[0]) > self.m:
                neighbor_node = self.nodes[neighbor_id]
                scored = [
                    (self.distance_fn(neighbor_node.vector, self.nodes[nid].vector), nid)
                    for nid in neighbor_node.neighbors[0]
                ]
                scored.sort()
                kept = {nid for _, nid in scored[: self.m]}
                pruned = neighbor_node.neighbors[0] - kept
                for pruned_id in pruned:
                    self.nodes[pruned_id].neighbors[0].discard(neighbor_node.id)
                neighbor_node.neighbors[0] = kept

    def search(self, query: Vector, k: int, ef: int) -> list[tuple[int, float]]:
        if not self.nodes:
            raise ValueError("Layer is empty!")
        assert self.entry_point is not None
        if k < 1:
            raise ValueError(f"k={k} should be greater than zero")
        if ef < k:
            raise ValueError(f"ef={ef} should be >= k={k}")

        candidates: list[tuple[float, int]] = []
        results: list[tuple[float, int]] = []
        visited: set[int] = set()

        d = self.distance_fn(self.entry_point.vector, query)
        heapq.heappush(candidates, (d, self.entry_point.id))
        heapq.heappush(results, (-d, self.entry_point.id))
        visited.add(self.entry_point.id)

        while candidates:
            closest_dist, closest_id = heapq.heappop(candidates)
            farthest_dist = -results[0][0]
            if closest_dist > farthest_dist:
                break
            for neighbor in self.nodes[closest_id].neighbors[0]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                d = self.distance_fn(self.nodes[neighbor].vector, query)
                farthest_dist = -results[0][0]
                if len(results) < ef or d < farthest_dist:
                    heapq.heappush(candidates, (d, neighbor))
                    heapq.heappush(results, (-d, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)

        return sorted([(nid, -neg_d) for neg_d, nid in results], key=lambda x: x[1])[:k]
