# Metrics Reference

This page provides a quick reference for all available metrics in pyERGM.

## Edge-Based Metrics

| Metric | Directed | Undirected | Description |
|--------|----------|------------|-------------|
| `NumberOfEdgesDirected` | ✓ | | Total count of edges in a directed network |
| `NumberOfEdgesUndirected` | | ✓ | Total count of edges in an undirected network |
| `NumberOfTriangles` | | ✓ | Count of triangles in the network |

## Degree Metrics

| Metric | Directed | Undirected | Description |
|--------|----------|------------|-------------|
| `InDegree` | ✓ | | Vector of in-degree for each node |
| `OutDegree` | ✓ | | Vector of out-degree for each node |
| `UndirectedDegree` | | ✓ | Vector of degree for each node |

## Reciprocity Metrics

| Metric | Directed | Undirected | Description |
|--------|----------|------------|-------------|
| `Reciprocity` | ✓ | | Vector of reciprocated edges per node |
| `TotalReciprocity` | ✓ | | Total count of reciprocated edges |

## Type-Based Metrics

| Metric | Directed | Undirected | Description |
|--------|----------|------------|-------------|
| `NumberOfEdgesTypesDirected` | ✓ | | Edge counts between node types (directed) |
| `NumberOfEdgesTypesUndirected` | | ✓ | Edge counts between node types (undirected) |
| `NumberOfNodesPerType` | ✓ | ✓ | Count of nodes of each type |

## Node Attribute Metrics

| Metric | Directed | Undirected | Description |
|--------|----------|------------|-------------|
| `NodeAttrSum` | ✓ | ✓ | Sum of node attributes for connected pairs |
| `NodeAttrSumOut` | ✓ | | Out-edge weighted by source node attribute |
| `NodeAttrSumIn` | ✓ | | In-edge weighted by target node attribute |
| `SumDistancesConnectedNeurons` | ✓ | ✓ | Sum of Euclidean distances between connected nodes |

---

For detailed API documentation, see the [API Reference](api/metrics.md).
