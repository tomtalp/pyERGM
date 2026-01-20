# Metrics Module

This module contains all the network statistics (metrics) that can be used with ERGM models.

## MetricsCollection

A collection of metrics that handles the calculation of network statistics.

::: pyERGM.metrics.MetricsCollection
    options:
      members:
        - __init__
        - calculate_statistics
        - calculate_sample_statistics
        - calc_num_of_features
        - get_metric
        - get_parameter_names
        - get_ignored_features
        - choose_optimization_scheme

## Base Metric Class

::: pyERGM.metrics.Metric
    options:
      members:
        - __init__
        - calculate
        - calculate_for_sample
        - calc_change_score

---

## Edge Count Metrics

### NumberOfEdgesUndirected

::: pyERGM.metrics.NumberOfEdgesUndirected

### NumberOfEdgesDirected

::: pyERGM.metrics.NumberOfEdgesDirected

### NumberOfTriangles

::: pyERGM.metrics.NumberOfTriangles

---

## Degree Metrics

### InDegree

::: pyERGM.metrics.InDegree

### OutDegree

::: pyERGM.metrics.OutDegree

### UndirectedDegree

::: pyERGM.metrics.UndirectedDegree

---

## Reciprocity Metrics

### Reciprocity

::: pyERGM.metrics.Reciprocity

### TotalReciprocity

::: pyERGM.metrics.TotalReciprocity

---

## Type-Based Metrics

### NumberOfEdgesTypesUndirected

::: pyERGM.metrics.NumberOfEdgesTypesUndirected

### NumberOfEdgesTypesDirected

::: pyERGM.metrics.NumberOfEdgesTypesDirected

---

## Node Attribute Metrics

### NodeAttrSum

::: pyERGM.metrics.NodeAttrSum

### NodeAttrSumOut

::: pyERGM.metrics.NodeAttrSumOut

### NodeAttrSumIn

::: pyERGM.metrics.NodeAttrSumIn

### SumDistancesConnectedNeurons

::: pyERGM.metrics.SumDistancesConnectedNeurons

### NumberOfNodesPerType

::: pyERGM.metrics.NumberOfNodesPerType
