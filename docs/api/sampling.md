# Sampling Module

This module provides MCMC sampling algorithms for generating networks from ERGM models.

## Sampler

Base class for samplers.

::: pyERGM.sampling.Sampler
    options:
      members:
        - __init__
        - sample

## NaiveMetropolisHastings

Implementation of the Metropolis-Hastings algorithm for ERGM sampling.

::: pyERGM.sampling.NaiveMetropolisHastings
    options:
      members:
        - __init__
        - sample
        - set_thetas
