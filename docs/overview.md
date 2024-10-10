# Overview
## What is an ERGM ?
Exponential Random Graph Models (ERGMs) are a class of statistical models that describe the distribution of random graphs. An ERGM defines a random variable $\mathbf{Y}$, which is simply a random graph on $n$ nodes. The probability of observing a graph $y\in\mathcal{Y}$ is given by - 

$$\Pr(\mathbf{Y}=y | \theta) = \frac{\exp(\theta^Tg(y))}{\sum_{z\in\mathcal{Y}} \exp(\theta^Tg(z))}$$

where $\mathcal{Y}$ is the set of all $n$ node graphs, $g(y)$ is a vector of statistics that describe the graph $y$, and $\theta \in \mathbb{R}^q$ is a vector of model parameters. Each graph is represented by a binary adjacency matrix, where $y_{ij}=1$ if there is an edge between nodes $i$ and $j$ (and $y_{ji}=1$ in the undirected case).

An important property of ERGMs is that they are subject to the **maximum entropy principle** - the optimal model is the one that maximizes the entropy subject to the constraints imposed by the graph statistics $g(\cdot)$. This makes ERGMs a powerful tool for modeling complex graph structures, since they ...? #TODO

## How to fit an ERGM?
Given a graph $y_{\text{obs}}$ and a set of statistics $g(y)$, the model is fit by maximizing the likelihood of the observed graph under the model.
The log-likelihood function is defined as follows - 

$$\ell(\theta | y_{\text{obs}}) = \theta^Tg(y_{\text{obs}}) - \log(\sum_{z\in\mathcal{Y}} \exp(\theta^Tg(z)))$$

which is optimized using numerical optimization techniques (such as Gradient Descent, Newton-Raphson, etc.) to find the optimal parameter vector - 

$$\theta^* = \arg \max \ \ell(\theta | y_{\text{obs}}) = \arg \min \ - \ell(\theta | y_{\text{obs}})$$

Generally speaking, there are two ways for fitting an ERGM - MCMLE and MPLE.

### MCMLE
Monte Carlo maximum likelihood estimation (MCMLE) is the main method for fitting ERGMs, and can be used it any setting, no matter what statistics $g(y)$ are calculated on a graph.

The algorithm can be sketched as follows - 

1. Pick a random starting point $\theta_0$.
2. Calculate the gradient of the log-likelihood function $\nabla \ell(\theta | y_{\text{obs}}) = g(y_{\text{obs}})- \mathbb{E}_{z\sim\mathcal{Y}}[g(z)]$ *(see Appendix for the full derivation).*
<br><br>
Calculating the expectation $\mathbb{E}_{z\sim\mathcal{Y}}[g(z)]$ is computationally infeasible, so we approximate it by sampling graphs from the distribution. Graphs can be sampled using an MCMC algorithm, such as the Metropolis-Hastings algorithm.
3. When using the Newton-Raphson method, calculate the Hessian matrix of the log-likelihood function - 
$H_{i, j} = \mathbb{E}_{z}[g_i(z)]\mathbb{E}_{z}[g_j(z)] - \mathbb{E}_{z}[g_i(z)g_j(z)]$
4. In every iteration, update the parameter vector $\theta$ using the gradient and Hessian - $\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot H^{-1}\nabla \ell(\theta | y_{\text{obs}})$ where $\eta$ is the learning rate. (When using the Gradient Descent method, the Hessian is not used.)
5. Repeat until optimization converges or reaches a maximum number of iterations.

#### Convergence criterions
pyERGM supports two convergence criterions for the optimization -

* **zero_grad_norm** - The optimization stops when the L2 norm of the gradient falls below a threshold.
* **hotelling** - The optimization stops when the Hotelling's $T^2$ statistic falls below a threshold.

*TODO - Elaborate on the convergence criterions.*

### MPLE

## How to sample from an ERGM?
It is computationally infeasible to directly draw graphs from the distribution - Calculating the normalization factor $Z=\sum_{z\in\mathcal{Y}} \exp(\theta^Tg(z))$ requires an iteration over the entire space of graphs, which is exponential in size. Instead, we can use Markov Chain Monte Carlo (**MCMC**) methods to sample from the distribution. The Metropolis-Hastings algorithm is a popular choice for sampling from ERGMs, and is the default sampling method in pyERGM.

In it's simplest form the Metropolis-Hastings algorithm iteratively collects samples from the distribution by following these steps -

1. Start with a randomly picked initial graph $y_0 \in \mathcal{Y}$, and set as the current graph $y_\text{current}=y_0$.
2. Propose a new graph $y_\text{proposed}$ by making a small change to $y_\text{current}$  either by adding or removing an edge between randomly picked nodes $i, j$.
3. Calculate the *change score*, defined as $\delta_g(y)_{i,j} = g(y_{\text{proposed}}) - g(y_{\text{current}})$
4. Accept the proposed graph with probability $p_{\text{accept}}=\min\left(1, \exp(\theta^T \delta_g(y)_{i,j}) \right)$.

Steps 2-4 are repeated for a large number of iterations, until a sufficient number of graphs are collected.

## Further Reading
pyERGM is inspired by the `ergm` package in `R`. For a good introduction to ERGMs, we recommend the following resources:

* [ergm: A Package to Fit, Simulate and Diagnose Exponential-Family Models for Networks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2743438/), *Hunter et al. (2008)*
* [ergm 4: Computational Improvements](https://arxiv.org/abs/2203.08198), *Krivitsky et al. (2022)*
* [ERGM introduction for Social Network Analysis](https://eehh-stanford.github.io/SNA-workshop/ergm-intro.html) is a workshop on Social Network Analysis from Stanford University, which provides a good example for using ERGMs in practice, using the `R` package.

## Appendix
### Deriving the gradient and Hessian 
Given a graph $y_{\text{obs}}$, we can treat the probability function as a likelihood function $\ell(\theta | y_{\text{obs}})$, which defines the log likelihood function $\ell(\theta)$ - 

$$\ell(\theta | y_{\text{obs}}) = \theta^Tg(y_{\text{obs}}) - \log(\sum_{z\in\mathcal{Y}} \exp(\theta^Tg(z)))$$

which can be optimized to obtain - 

$$ \theta^* = \arg \max \ \ell(\theta | y_{\text{obs}}) = \arg \min \ - \ell(\theta | y_{\text{obs}}) $$

The log-likelihood funcrtion can be differentiated with respect to $\theta$ to obtain the gradient -

$$
\frac{\partial}{\partial \theta} \ \ell (\theta) = g(y_{\text{obs}}) - \frac{\sum_{z\in\mathcal{Y}} \exp(\theta^Tg(z))g(z)}{\sum_{z\in\mathcal{Y}} \exp(\theta^Tg(z))} 
$$

$$
= g(y_{\text{obs}}) - \sum_{z\in\mathcal{Y}} \frac{\exp(\theta^Tg(z))}{Z}g(z)
$$

$$
= g(y_{\text{obs}})- \sum_{z\in\mathcal{Y}}\Pr_{\theta, \mathcal{Y}}(\mathbf{Y}=z)g(z) 
$$

$$
= g(y_{\text{obs}})- \mathbb{E}_{z\sim\mathcal{Y}}[g(z)] 
$$

where $Z=\sum_{z\in\mathcal{Y}} \exp(\theta^Tg(z))$ is the normalization factor.

We can now take the second derivative, to find the $i, j$-th entry of the **Hessian matrix** -

$$
\frac{\partial^2\ell(\theta)}{\partial \theta_i \theta_j} = \frac{\partial}{\partial \theta_j} \big(g_i(y_{\text{obs}}) - \mathbb{E}_{z\sim\mathcal{Y}}[g_i(z)]  \big)
 = -\sum_{z\in \mathcal{Y}} \frac{\partial}{\partial \theta_j} \big( \frac{\exp(\theta^T g(z)) \cdot g_i(z)}{Z} \big)
$$

The derivative of the summed term can be calculated as follows - 

$$
\frac{\partial}{\partial \theta_j} \frac{\exp(\theta^T g(z)) }{Z}\cdot g_i(z) = \frac{\exp(\theta^T g(z))g_j(z)Z - \exp(\theta^T g(z)) \cdot \frac{\partial}{\partial \theta_j}Z }{Z^2} \cdot g_i(z)
$$

$$
= \frac{\exp(\theta^T g(z))}{Z^2} \cdot g_i(z) \cdot \big( g_j(z)Z - \sum_{z\in \mathcal{Y}} \exp(\theta^T g(z)) g_j(z)  \big)
$$

$$
= \frac{\exp(\theta^T g(z))}{Z} \cdot g_i(z)\cdot (g_j(z) - \sum_{z\in \mathcal{Y}}\Pr[Y=z]\cdot g_j(z))
$$

$$
= \frac{\exp(\theta^T g(z))}{Z} \cdot g_i(z)\cdot (g_j(z) - \mathbb{E}_{z\sim\mathcal{Y}}[g_j(z)])
$$

which can now be plugged back - 

$$
\frac{\partial^2\ell(\theta)}{\partial \theta_i \theta_j} = -\sum_{z\in\mathcal{Y}} \frac{\exp(\theta^T g(z))}{Z} \cdot g_i(z)\cdot (g_j(z) - \mathbb{E}_{z\sim\mathcal{Y}}[g_j(z)])
$$

$$
= -\sum_{z\in\mathcal{Y}}\Pr[Y=z]g_i(z)g_j(z) + \sum_{z\in\mathcal{Y}}\Pr[Y=z]g_i(z)\mathbb{E}_{z\sim\mathcal{Y}}[g_j(z)]
$$

$$
= \mathbb{E}_{z}[g_i(z)]\mathbb{E}_{z}[g_j(z)] - \mathbb{E}_{z}[g_i(z)g_j(z)]
$$

This derivative defines the $i, j$-th entry of the Hessian matrix.
