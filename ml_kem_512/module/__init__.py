"""
Module operations for ML-KEM-512 (k=2).
"""

from ml_kem_512.module.vector import (
    K,
    PolyVec,
    dot_ntt,
    inv_ntt_vec,
    mat_vec_mul,
    ntt_vec,
    transpose,
    zero_vec,
)

__all__ = [
    "PolyVec",
    "zero_vec",
    "ntt_vec",
    "inv_ntt_vec",
    "dot_ntt",
    "mat_vec_mul",
    "transpose",
    "K",
]
