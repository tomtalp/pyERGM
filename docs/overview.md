# Overview
## What is an ERGM ?
Exponential Random Graph Models (ERGMs) are a class of statistical models that describe the distribution of random graphs. An ERGM defines a random variable $\mathbf{Y}$, which is simply a random graph on $n$ nodes. The probability of observing a graph $y\in\mathcal{Y}$ is given by - 

$$\Pr(\mathbf{Y}=y | \theta) = \frac{\exp(\theta^Tg(y))}{\sum_{z\in\mathcal{Y}} \exp(\theta^Tg(z))}$$

where $\mathcal{Y}$ is the set of all $n$ node graphs, $g(y)$ is a vector of statistics that describe the graph $y$, and $\theta \in \mathbb{R}^q$ is a vector of model parameters. Each graph is represented by a binary adjacency matrix, where $y_{ij}=1$ if there is an edge between nodes $i$ and $j$ (and $y_{ji}=1$ in the undirected case).

An important property of ERGMs is that they are subject to the **maximum entropy principle** - the optimal model is the one that maximizes the entropy subject to the constraints imposed by the network statistics $g(\cdot)$. This makes ERGMs a powerful tool for modeling complex network structures, since they ...? #TODO

## How to fit an ERGM?
Given a graph $y_{\text{obs}}$ and a set of statistics $g(y)$, the model is fit by maximizing the likelihood of the observed graph under the model.
The log-likelihood function is defined as follows - 

$$\ell(\theta | y_{\text{obs}}) = \theta^Tg(y_{\text{obs}}) - \log(\sum_{z\in\mathcal{Y}} \exp(\theta^Tg(z)))$$

which is optimized using numerical optimization techniques (such as Gradient Descent, Newton-Raphson, etc.) to find the optimal parameter vector - 

$$\theta^* = \arg \max \ \ell(\theta | y_{\text{obs}}) = \arg \min \ - \ell(\theta | y_{\text{obs}})$$

## How to sample from an ERGM?
It is computationally infeasible to directly draw networks from the distribution - Calculating the normalization factor $Z=\sum_{z\in\mathcal{Y}} \exp(\theta^Tg(z))$ requires an iteration over the entire space of networks, which is exponential in size. Instead, we can use Markov Chain Monte Carlo (**MCMC**) methods to sample from the distribution. The Metropolis-Hastings algorithm is a popular choice for sampling from ERGMs, and is the default sampling method in pyERGM.

In it's simplest form the Metropolis-Hastings algorithm iteratively collects samples from the distribution by following these steps -

1. Start with a randomly picked initial graph $y_0 \in \mathcal{Y}$, and set as the current graph $y_\text{current}=y_0$.
2. Propose a new graph $y_\text{proposed}$ by making a small change to $y_\text{current}$  either by adding or removing an edge between randomly picked nodes $i, j$.
3. Calculate the *change score*, defined as $\delta_g(y)_{i,j} = g(y_{\text{proposed}}) - g(y_{\text{current}})$
4. Accept the proposed network with probability $p_{\text{accept}}=\min\left(1, \exp(\theta^T \delta_g(y)_{i,j}) \right)$.

Steps 2-4 are repeated for a large number of iterations, until a sufficient number of networks are collected.

## Further Reading
pyERGM is inspired by the `ergm` package in `R`. For a good introduction to ERGMs, we recommend the following resources:

* [ergm: A Package to Fit, Simulate and Diagnose Exponential-Family Models for Networks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2743438/), *Hunter et al. (2008)*
* [ergm 4: Computational Improvements](https://arxiv.org/abs/2203.08198), *Krivitsky et al. (2022)*