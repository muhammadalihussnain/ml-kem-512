"""
Number Theoretic Transform (NTT) for ML-KEM-512.

Implements FIPS 203 Algorithms 9, 10, 11 exactly:
  n = 256, q = 3329
  zeta = 17  (primitive 512th root of unity mod q, so zeta^512 = 1 mod q)

The NTT decomposes Z_q[x]/(x^256+1) into 128 quadratic residue rings.
Each "layer" of the butterfly uses twiddle factors zeta^BitRev7(k).

Forward NTT  : ntt(poly)           -> list[int] of 256 NTT coefficients
Inverse NTT  : inv_ntt(f_hat)      -> Polynomial
NTT multiply : ntt_mul(a_hat, b_hat) -> list[int]
"""

from __future__ import annotations

from ml_kem_512.polynomial.poly import N, Polynomial, Q

# Primitive 512th root of unity mod q=3329
ZETA = 17

# 128^-1 mod 3329  (inv_ntt scales by (N/2)^-1 due to butterfly structure)
_N_INV = pow(N // 2, -1, Q)  # = 3303


def _bit_rev_7(n: int) -> int:
    """Reverse the 7 least-significant bits of n."""
    result = 0
    for _ in range(7):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result


# Precomputed twiddle factors: ZETAS[k] = zeta^BitRev7(k) mod q  for k=0..127
ZETAS = [pow(ZETA, _bit_rev_7(k), Q) for k in range(128)]


def ntt(poly: Polynomial) -> list:
    """
    Forward NTT (FIPS 203 Algorithm 9).

    Transforms a polynomial in Z_q[x]/(x^256+1) into NTT domain.

    Args:
        poly: Polynomial with N=256 coefficients

    Returns:
        list of 256 integers in [0, Q-1]
    """
    f = list(poly.coeffs)
    k = 1
    length = 128
    while length >= 2:
        for start in range(0, N, 2 * length):
            zeta = ZETAS[k]
            k += 1
            for j in range(start, start + length):
                t = (zeta * f[j + length]) % Q
                f[j + length] = (f[j] - t) % Q
                f[j] = (f[j] + t) % Q
        length //= 2
    return f


def inv_ntt(f_hat: list) -> Polynomial:
    """
    Inverse NTT (FIPS 203 Algorithm 10).

    Transforms NTT coefficients back into a polynomial.

    Args:
        f_hat: list of 256 NTT coefficients

    Returns:
        Polynomial with N=256 coefficients in [0, Q-1]
    """
    f = list(f_hat)
    k = 127
    length = 2
    while length <= 128:
        for start in range(0, N, 2 * length):
            zeta = ZETAS[k]
            k -= 1
            for j in range(start, start + length):
                t = f[j]
                f[j] = (t + f[j + length]) % Q
                f[j + length] = (zeta * (f[j + length] - t)) % Q
        length *= 2
    # Final scaling by n^-1 mod q
    f = [(c * _N_INV) % Q for c in f]
    return Polynomial(f)


def ntt_mul(a_hat: list, b_hat: list) -> list:
    """
    Pointwise multiplication in NTT domain (FIPS 203 Algorithm 11).

    Multiplies two NTT representations using base-case multiply on each
    pair of 2-element NTT units:
      (a0 + a1*X) * (b0 + b1*X) mod (X^2 - gamma)
      = (a0*b0 + a1*b1*gamma) + (a0*b1 + a1*b0)*X
    where gamma = zeta^(2*BitRev7(i)+1) for pair i.

    Args:
        a_hat: NTT representation of a (256 ints)
        b_hat: NTT representation of b (256 ints)

    Returns:
        NTT representation of a*b (256 ints)
    """
    result = [0] * N
    for i in range(128):
        a0, a1 = a_hat[2 * i], a_hat[2 * i + 1]
        b0, b1 = b_hat[2 * i], b_hat[2 * i + 1]
        gamma = pow(ZETA, 2 * _bit_rev_7(i) + 1, Q)
        result[2 * i] = (a0 * b0 + a1 * b1 * gamma) % Q
        result[2 * i + 1] = (a0 * b1 + a1 * b0) % Q
    return result


def poly_from_ntt(f_hat: list) -> Polynomial:
    """Convenience: wrap a raw NTT list back into a Polynomial."""
    return Polynomial(f_hat)
