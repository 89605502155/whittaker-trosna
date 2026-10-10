"""Сглаживание Уиттекера, взвешенное по свежести точек временного ряда."""

from .core import (
    SmoothResult,
    difference_matrix,
    fit_recency_whittaker,
    objective,
    penalty_matrix,
    recency_whittaker,
    smooth_with_mobility,
    whittaker,
)
from .lambda_selection import (
    LambdaSearch,
    cutoff_for_lambda,
    lambda_for_cutoff,
    select_lambda,
)
from .streaming import StreamingWhittaker, StreamSnapshot
from .weights import (
    WEIGHT_FUNCTIONS,
    normalized_age,
    normalized_age_from_times,
    recency_coefficients,
)

__all__ = [
    "WEIGHT_FUNCTIONS",
    "LambdaSearch",
    "SmoothResult",
    "StreamSnapshot",
    "StreamingWhittaker",
    "cutoff_for_lambda",
    "difference_matrix",
    "fit_recency_whittaker",
    "lambda_for_cutoff",
    "normalized_age",
    "normalized_age_from_times",
    "objective",
    "penalty_matrix",
    "recency_coefficients",
    "recency_whittaker",
    "select_lambda",
    "smooth_with_mobility",
    "whittaker",
]
