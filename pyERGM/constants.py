"""
Enum constants for ERGM configuration options.
"""
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional


@dataclass
class OptimizationResult:
    """Result of an ERGM optimization (fitting) procedure."""
    success: bool
    statistic: Optional[float] = None
    threshold: Optional[float] = None


class OptimizationScheme(StrEnum):
    """Scheme for model fitting (determines MPLE vs MCMLE path)."""
    AUTO = "AUTO"
    MPLE = "MPLE"
    MPLE_RECIPROCITY = "MPLE_RECIPROCITY"
    MCMLE = "MCMLE"


class OptimizationMethod(StrEnum):
    """Optimization method for MCMLE."""
    NEWTON_RAPHSON = "newton_raphson"
    GRADIENT_DESCENT = "gradient_descent"


class MPLEOptimizationMethod(StrEnum):
    """Optimization method for MPLE (scipy.optimize methods)."""
    L_BFGS_B = "L-BFGS-B"
    NEWTON_CG = "Newton-CG"


class ConvergenceCriterion(StrEnum):
    """Convergence criterion for MCMLE."""
    HOTELLING = "hotelling"
    ZERO_GRAD_NORM = "zero_grad_norm"
    OBSERVED_BOOTSTRAP = "observed_bootstrap"
    MODEL_BOOTSTRAP = "model_bootstrap"


class CovMatrixEstimationMethod(StrEnum):
    """Method for estimating the covariance matrix."""
    NAIVE = "naive"
    BATCH = "batch"
    MULTIVARIATE_INITIAL_SEQUENCE = "multivariate_initial_sequence"


class ThetaInitMethod(StrEnum):
    """Method for initializing theta parameters."""
    MPLE = "mple"
    USE_EXISTING = "use_existing"


class EdgeProposalMethod(StrEnum):
    """Edge proposal distribution for MCMC sampling."""
    UNIFORM = "uniform"
    FEATURES_INFLUENCE_SUM = "features_influence__sum"
    FEATURES_INFLUENCE_SOFTMAX = "features_influence__softmax"


class SamplingMethod(StrEnum):
    """Network sampling method."""
    METROPOLIS_HASTINGS = "metropolis_hastings"
    EXACT = "exact"


class Reduction(StrEnum):
    """Reduction method for aggregating values."""
    SUM = "sum"
    MEAN = "mean"
    NONE = "none"


class DataBootstrapSplittingMethod(StrEnum):
    """Splitting method for bootstrapping neurons"""
    UNIFORM = "uniform"
