# ERGM Module

The core module containing the ERGM model class and related utilities.

## ERGM

The main class for fitting Exponential Random Graph Models.

::: pyERGM.ergm.ERGM
    options:
      members:
        - __init__
        - fit
        - generate_networks_for_sample
        - get_model_parameters
        - get_ignored_features
        - calculate_probability
        - calculate_weight
        - calc_model_log_likelihood
        - calc_model_entropy
        - get_mcmc_diagnostics
        - get_mple_prediction
        - get_mple_reciprocity_prediction
        - print_model_parameters

## BruteForceERGM

A class for exact ERGM calculations on small networks (for testing purposes).

::: pyERGM.ergm.BruteForceERGM
    options:
      members:
        - __init__
        - fit
        - generate_networks_for_sample
        - calc_expected_features

## ConvergenceTester

Convergence testers for MCMLE optimization. Each criterion is implemented as a
stateful subclass behind a common abstract API with a factory method.

::: pyERGM.convergence.ConvergenceTester
    options:
      members:
        - create
        - update
        - test
        - requires_covariance_estimation
