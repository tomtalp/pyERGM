"""
Enum constants for ERGM configuration options.

All enums inherit from str for backward compatibility - string values
can be used interchangeably with enum values.
"""
from enum import Enum


class OptimizationScheme(str, Enum):
    """Scheme for model fitting (determines MPLE vs MCMLE path)."""
    AUTO = "AUTO"
    MPLE = "MPLE"
    MPLE_RECIPROCITY = "MPLE_RECIPROCITY"
    MCMLE = "MCMLE"


class OptimizationMethod(str, Enum):
    """Optimization method for MCMLE."""
    NEWTON_RAPHSON = "newton_raphson"
    GRADIENT_DESCENT = "gradient_descent"


class MPLEOptimizationMethod(str, Enum):
    """Optimization method for MPLE (scipy.optimize methods)."""
    L_BFGS_B = "L-BFGS-B"
    NEWTON_CG = "Newton-CG"


class ConvergenceCriterion(str, Enum):
    """Convergence criterion for MCMLE."""
    HOTELLING = "hotelling"
    ZERO_GRAD_NORM = "zero_grad_norm"
    OBSERVED_BOOTSTRAP = "observed_bootstrap"
    MODEL_BOOTSTRAP = "model_bootstrap"


class CovMatrixEstimationMethod(str, Enum):
    """Method for estimating the covariance matrix."""
    NAIVE = "naive"
    BATCH = "batch"
    MULTIVARIATE_INITIAL_SEQUENCE = "multivariate_initial_sequence"


class ThetaInitMethod(str, Enum):
    """Method for initializing theta parameters."""
    UNIFORM = "uniform"
    MPLE = "mple"
    USE_EXISTING = "use_existing"


class EdgeProposalMethod(str, Enum):
    """Edge proposal distribution for MCMC sampling."""
    UNIFORM = "uniform"
    FEATURES_INFLUENCE_SUM = "features_influence__sum"
    FEATURES_INFLUENCE_SOFTMAX = "features_influence__softmax"


class SamplingMethod(str, Enum):
    """Network sampling method."""
    METROPOLIS_HASTINGS = "metropolis_hastings"
    EXACT = "exact"


class Reduction(str, Enum):
    """Reduction method for aggregating values."""
    SUM = "sum"
    MEAN = "mean"
    NONE = "none"
