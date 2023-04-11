"""MultiCons module"""

# flake8: noqa

from multicons.consensus import (
    consensus_function_10,
    consensus_function_12,
    consensus_function_13,
    consensus_function_14,
    consensus_function_15,
)
from multicons.core import MultiCons
from multicons.utils import (
    build_membership_matrix,
    in_ensemble_similarity,
    linear_closed_itemsets_miner,
)

__version__ = "0.2.0"
