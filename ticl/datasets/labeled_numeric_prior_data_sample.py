"""
Backward compatibility module for labeled numeric prior data.

This module is retained for backward compatibility.
New code should use labeled_numeric_prior_data_loader.py instead.
"""

import warnings
from ticl.datasets.labeled_numeric_prior_data_loader import (
    labeled_numeric_data,
    column_metadata,
    get_random_value,
    get_random_batch,
    get_columns_by_domain,
    get_noisy_value
)

warnings.warn(
    "labeled_numeric_prior_data_sample is deprecated. Use labeled_numeric_prior_data_loader instead.",
    DeprecationWarning,
    stacklevel=2
)