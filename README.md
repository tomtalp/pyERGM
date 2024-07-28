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
This creates two networks, $y_{\text{current}}\  , \  y_{\text{proposed}}$.

### Step 3

Decide whether to keep the perturbation or not, according to the Acceptance Ratio *(AR)*, defined as - 

$$
\text{AR} = \frac{\Pr(\mathbf{Y}=y_{\text{proposed}})}{\Pr(\mathbf{Y}=y_{\text{current}})}\cdot \frac{q(y_{\text{current}} | y_{\text{proposed}})}{q(y_{\text{proposed}} | y_{\text{current}})}
$$

where $q(a|b)$ is the probability of proposing network $a$ when the current network is $b$. In the simplest case $q(a|b) = q(b|a$), which is the case when we toggle whether an edge exists or not.  This is also called a *symmetric proposal* [[1]](#1).

We accept the change with probability $p_\text{accept} = 
\min(1, \text{AR})
$.
The intuition is that if $\text{AR}>1$ then $y_{\text{proposed}}$ has a higher probability of appearing in the distribution and is thus immediately selected. Otherwise we randomly pick a candidate with probability equal to ratio. 

#### A simplification for calculating the AR
Observe the quantity $\log(\text{AR})$ in the symmetric proposal case - 

$$\log(\text{AR}) = \log(\frac{\Pr(\mathbf{Y}=y_{\text{proposed}})}{\Pr(\mathbf{Y}=y_{\text{current}})})
= \log(\frac{\exp(\theta^Tg(y_{\text{proposed}})) / \kappa }{\exp(\theta^Tg(y_{\text{current}})) / \kappa }) = \theta^T \delta_g(y)_{i,j}$$

where $\delta_g(y)_{i,j}$ is the *change score* - 

$$
\delta_g(y)_{i,j} = g(y\_{\text{proposed}}) - g(y\_{\text{current}})
$$

Now all that's left to do is accept the suggested change with probability 

$$
p_{\text{accept}} = \min(1, \exp(\theta^T \delta_g(y)_{i,j}))
$$

### Step 4
Steps 1-3 create a Markov chain over possible samples from the ERGM distribution. Repeat them for $n$ steps, and eventually the chain will converge to an appropriate sample from the ERGM distribution.



## Fitting
The simplest way to fit an ERGM is by performing a Maximum Likelihood estimation (**MLE**).

Given a network $y_{\text{obs}}$, we can treat the probability function as a likelihood function $\ell(\theta | y_{\text{obs}})$, which defines the log likelihood function $\ell(\theta)$ - 

$$
\ell(\theta | y_{\text{obs}}) = \theta^Tg(y_{\text{obs}}) - \log(\sum_{z\in\mathcal{Y}} \exp(\theta^Tg(z)))
$$

which can be optimized to obtain - 

$$
\theta^* = \arg \max \ \ell(\theta | y_{\text{obs}}) = \arg \min \ - \ell(\theta | y_{\text{obs}})
$$

We now take the derivative - 

$$
\frac{\partial}{\partial \theta} \ \ell (\theta) = g(y_{\text{obs}}) - \frac{\sum_{z\in\mathcal{Y}} \exp(\theta^Tg(z)))g(z)}{\sum_{z\in\mathcal{Y}} \exp(\theta^Tg(z)))} = g(y_{\text{obs}}) - \sum_{z\in\mathcal{Y}} \frac{\exp(\theta^Tg(z)))}{\kappa}g(z)
= g(y_{\text{obs}})- \sum_{z\in\mathcal{Y}}\Pr_{\theta, \mathcal{Y}}(\mathbf{Y}=z)g(z) = g(y_{\text{obs}})- \mathbb{E}_{z\sim\mathcal{Y}}[g(z)] 
$$


## References
<a id="1">[1]</a> 
Krivitsky, Hunter, et al. (2022). 
ergm 4: Computational Improvements
