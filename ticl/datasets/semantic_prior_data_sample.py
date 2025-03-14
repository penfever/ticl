"""
Backward compatibility module for semantic prior data.

This module is retained for backward compatibility.
New code should use semantic_prior_data_loader.py instead.
"""

import warnings
from ticl.datasets.semantic_prior_data_loader import get_random_semantic_data

warnings.warn(
    "semantic_prior_data_sample is deprecated. Use semantic_prior_data_loader instead.",
    DeprecationWarning,
    stacklevel=2
)

# For backward compatibility
random_tensor = get_random_semantic_data()