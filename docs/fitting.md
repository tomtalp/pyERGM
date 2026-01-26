Fitting an ERGM model requires a graph and a set of statistics that describe the graph. The model is then fit by maximizing the likelihood of the observed graph under the model. 

The following example demonstrates how to fit a simple ERGM model from [Sampson's monastery data](https://networkdata.ics.uci.edu/netdata/html/sampson.html).

```python
from pyERGM.ergm import ERGM
from pyERGM.metrics import *
from pyERGM.datasets import load_sampson

sampson_matrix = load_sampson()

num_nodes = sampson_matrix.shape[0]
is_directed = True
metrics = [NumberOfEdgesDirected(), TotalReciprocity()]

model = ERGM(num_nodes, metrics, is_directed=is_directed)
model.fit(sampson_matrix)
```

The above example fits a model from the Sampson's monastery data using the number of edges and total reciprocity as statistics. The graph is represented as an adjacency matrix.

## ERGM
```python
class pyERGM.ERGM(n_nodes, metrics_collection, is_directed, **kwargs)
```

**Parameters**:

* **n_nodes** (*int*) - Number of nodes in the graph.
* **metrics_collection** (*Collection[Metric]*) - A list of Metric objects for calculating statistics of a graph.
* **is_directed** (*bool*) - Whether the graph is directed or not.
* **initial_thetas** (*np.ndarray*) - Optional. The initial values of the coefficients of the ERGM. If not provided, they are randomly initialized.
* **initial_normalization_factor** (*float*) - Optional. The initial value of the normalization factor. If not provided, it is randomly initialized.
* **seed_MCMC_proba** (*float*) - Optional. The probability of a connection in the seed graph for MCMC sampling, in case no seed graph is provided. *Defaults to 0.25*
* **sample_size** (*int*) - Optional. The number of graphs to sample via MCMC. If number of samples is odd, it is increased by 1. This is because downstream algorithms assume the sample size is even (e.g. the Covariance matrix estimation). *Defaults to 1000*
* **fix_collinearity** (*bool*) - Optional. Whether to fix collinearity in the metrics. *Defaults to True*
* **collinearity_fixer_sample_size** (*int*) - Optional. The number of graphs to sample for fixing collinearity. *Defaults to 1000*

### fit
```python
pyERGM.ERGM.fit(observed_graph, **kwargs)
```
Fit an ERGM model to a given graph with one of the two fitting methods - MPLE or MCMLE.

With the exception of dyadic dependent models, all models are fit using the MCMLE method. Dyadic dependent models are fit using the MPLE method, which simply amounts to running a logistic regression.


**Parameters**:

* **observed_graph** (*np.ndarray*) - The adjacency matrix of the observed graph.
* **lr** (*float*) - Optional. The learning rate for the optimization. *Defaults to 0.1*
* **opt_steps** (*int*) - Optional. The number of optimization steps to run. *Defaults to 1000*
* **steps_for_decay** (*int*) - Optional. The number of steps after which to decay the optimization params. *Defaults to 100* # TODO - redundant parameter?
* **lr_decay_pct** (*float*) - Optional. The decay factor for the learning rate. *Defaults to 0.01*
* **l2_grad_thresh** (*float*) - Optional. The threshold for the L2 norm of the gradient to stop the optimization. Relevant only for convergence criterion `zero_grad_norm`. *Defaults to 0.001*
* **sliding_grad_window_k** (*int*) - Optional. The size of the sliding window for the gradient, for which we use to calculate the mean gradient norm. This value is then tested against l2_grad_thresh to decide whether optimization halts.Relevant only for convergence criterion `zero_grad_norm`. *Defaults to 10*
* **max_sliding_window_size** (*int*) - Optional. The maximum size of the sliding window for the gradient. Relevant only for convergence criterion `zero_grad_norm`. *Defaults to 100*
* **max_nets_for_sample** (*int*) - Optional. The maximum number of graphs to sample with MCMC. *Defaults to 1000* #TODO - Do we still need this? Seems like increasing the sample size isn't necessary (we'll gonna pick large sample sizes anyway)    
* **sample_pct_growth** (*float*) - Optional. The percentage growth of the number of graphs to sample, which we want to increase over time. *Defaults to 0.02*. #TODO - Same as `max_nets_for_sample`. Do we still need this?
* **optimization_method** (*str*) - Optional. The optimization method to use. Can be either "newton_raphson" or "gradient_descent". *Defaults to "newton_raphson"*.
* **convergence_criterion** (*str*) - Optional. The criterion for convergence. Can be either "hotelling" or "zero_grad_norm". *Defaults to "zero_grad_norm"*. # TODO - Revisit this when we fix convergence criterion.
* **cov_matrix_estimation_method** (*str*) - Optional. The method to estimate the covariance matrix. Supported methods - `naive`, `batch`, `multivariate_initial_sequence`. *Defaults to "batch"*.
* **cov_matrix_num_batches** (*int*) - Optional. The number of batches to use for estimating the covariance matrix.Relevant only for `cov_matrix_estimation_method="batch"`. *Defaults to 25*.
* **hotelling_confidence** (*float*) - Optional. The confidence level for the Hotelling's T-squared test. *Defaults to 0.99*.
* **theta_init_method** (*str*) - Optional. The method to initialize the theta values. Can be either "uniform" or "mple". The MPLE method can be used even for dyadic dependent models, since it serves as a good starting point for the MCMLE. *Defaults to "mple"*.
* **no_mple** (*bool*) - Optional. Whether to skip the MPLE step and go directly to MCMLE. *Defaults to False*.
* **mcmc_burn_in** (*int*) - Optional. The number of burn-in steps for the MCMC sampler. *Defaults to 1000*.
* **mcmc_steps_per_sample** (*int*) - Optional. The number of steps to run the MCMC sampler for each sample. *Defaults to 10*.
* **mcmc_seed_network** (*np.ndarray*) - Optional. The seed network for the MCMC sampler. If not provided, the thetas that are currently set are used to calculate the MPLE prediction, from which the sample is drawn. *Defaults to None*.
* **mcmc_sample_size** (*int*) - Optional. The number of networks to sample with the MCMC sampler. *Defaults to 100*.
* **mple_lr** (*float*) - Optional. The learning rate for the logistic regression model in the MPLE step. *Defaults to 0.001*.
* **mple_stopping_thr** (*float*) - Optional. The stopping threshold for the logistic regression model in the MPLE step. *Defaults to 1e-6*.
* **mple_max_iter** (*int*) - Optional. The maximum number of iterations for the logistic regression model in the MPLE step. *Defaults to 1000*.


**Returns**:

* **grads** (*np.ndarray*) - The gradients of the model parameters.
* **hotelling_statistics** (*list*) - The Hotelling's T-squared statistics for the model parameters. 

### print_model_parameters
```python
pyERGM.ERGM.print_model_parameters()
```
Prints the parameters of the ERGM model.
