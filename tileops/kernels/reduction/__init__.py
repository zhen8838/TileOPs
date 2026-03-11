"""Reduction kernel primitives and shared utilities.

This package provides the foundational building blocks (macros, constants,
and utility functions) used by all reduction sub-category kernels.
"""

from ._primitives import (
    DEFAULT_ALIGNMENT,
    _align_up,
    make_cumulative_scan,
    make_reduce_epilogue,
    make_softmax_epilogue,
    make_welford_update,
)

__all__ = [
    "_align_up",
    "DEFAULT_ALIGNMENT",
    "make_reduce_epilogue",
    "make_welford_update",
    "make_softmax_epilogue",
    "make_cumulative_scan",
]
