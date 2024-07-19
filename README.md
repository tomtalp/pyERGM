# pyERGM
pyERGM is a pure NumPy & NetworkX implementation of an **ERGM** distribution model.

## The ERGM Model
An ERGM distribution defines a random variable $\mathbf{Y}$, with its probability defined as 

$$\Pr_{\theta, \mathcal{Y}}(\mathbf{Y}=y) = \frac{\exp(\theta^Tg(y))}{\sum_{z\in\mathcal{Y}} \exp(\theta^Tg(z))}$$

This denotes the probability of sampling a network $y$ from the space of networks $\mathcal{Y}$, given model parameters $\theta$. The denominator is the normalization factor, also referred to as $\kappa$.
## Drawing a network from the distribution
It is computationally infeasible to directly draw networks from the distribution - Calculating the normalization factor $\kappa$ requires an iteration over the entire space of networks, which is exponential.

As a solution, we can use an MCMC sampler to approximate what a draw from the distribution would look like.

We implement the Metropolis-Hastings algorithm, which in it's simplest form works as follows -

### Step 1

Initialize a random network $y_0\in\mathcal{Y}$, which is set to be the first candidate of our MCMC. 

Different initializations can be used, the simplest one being a random sample from the Erdős–Rényi model with some probability.

### Step 2

Perturb the previous candidate network by adding or removing one of it's edges, between nodes $i, j$.
This creates two networks, $y^+ , y^-$.

### Step 3

Decide whether to keep the perturbation or not, according to the Acceptance Ratio *(AR)*, defined as - 

$$
\text{AR} = \frac{\Pr(\mathbf{Y}=y^+)}{\Pr(\mathbf{Y}=y^-)}\cdot \frac{q(y^- | y^+)}{q(y^+ | y^-)}
$$

where $q(a|b)$ is the probability of proposing network $a$ when the current network is $b$. In the simplest case $q(a|b) = q(b|a$), which is the case when we toggle whether an edge exists or not.  This is also called a *symmetric proposal* [[1]](#1).

We accept the change with probability $p_\text{accept} = 
\min(1, \text{AR})
$.
The intuition is that if $\text{AR}>1$ then $y^+$ has a higher probability of appearing in the distribution and is thus immediately selected. Otherwise we randomly pick a candidate with probability equal to ratio. 

#### A simplification for calculating the AR
Observe the quantity $\log(\text{AR})$ in the symmetric proposal case - 

$$\log(\text{AR}) = \log(\frac{\Pr(\mathbf{Y}=y^+)}{\Pr(\mathbf{Y}=y^-)})
= \log(\frac{\exp(\theta^Tg(y^+)) / \kappa }{\exp(\theta^Tg(y^-)) / \kappa }) = \theta^T \delta_g(y)_{i,j}$$

where $\delta_g(y)_{i,j} = g(y^+) - g(y^-)$ is the *change score*.

Now all that's left to do is accept the suggested change with probability 

$$
p_{\text{accept}} = \min(1, \exp(\theta^T \delta_g(y)_{i,j}))
$$

### Step 4
Steps 1-3 create a Markov chain over possible samples from the ERGM distribution. Repeat them for $n$ steps, and eventually the chain will converge to an appropriate sample from the ERGM distribution.



## Fitting

## References
<a id="1">[1]</a> 
Krivitsky, Hunter, et al. (2022). 
ergm 4: Computational Improvements