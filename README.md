# tinyhnsw

A minimal HNSW (Hierarchical Navigable Small World) implementation built from scratch in Python.


## What is HNSW?

Given millions of high-dimensional vectors (embeddings), finding the closest one to a query with brute force is O(N·d) — too slow. HNSW solves this with a layered graph structure inspired by skip lists:

- **Layer 0** contains all vectors, connected to their nearest neighbors
- **Higher layers** contain progressively fewer vectors with long-range connections
- **Search** starts at the top layer (fast, coarse jumps) and descends to layer 0 (slow, precise refinement)

This gives O(log N) approximate search with high recall.

## Roadmap

Roadmap
- [x] Distance functions (euclidean, cosine)
- [ ] Brute-force KNN baseline
- [ ] Shared types & data structures
- [ ] Single-layer NSW graph
- [ ] Hierarchical multi-layer structure
- [ ] HNSW search (multi-layer greedy + beam)
- [ ] HNSW insert with neighbor selection
- [ ] Recall benchmark vs brute-force
- [ ] Serialization
