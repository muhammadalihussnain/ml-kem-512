"""
Polynomial arithmetic for ML-KEM-512.

Polynomials live in the ring R_q = Z_q[x] / (x^256 + 1)
  n = 256  (degree bound)
  q = 3329 (modulus)

All coefficients are kept in the canonical range [0, q-1].
"""

from __future__ import annotations

# ML-KEM-512 parameters
N = 256
Q = 3329


class Polynomial:
    """
    Element of R_q = Z_q[x] / (x^256 + 1).

    Internally stored as a list of N=256 integers in [0, Q-1].
    """

    def __init__(self, coeffs: list[int] | None = None):
        """
        Create a polynomial.

        Args:
            coeffs: list of up to N integers. Missing trailing coefficients
                    are zero-padded. If None, creates the zero polynomial.
        """
        if coeffs is None:
            self.coeffs = [0] * N
        else:
            if len(coeffs) > N:
                raise ValueError(f"Polynomial has at most {N} coefficients, got {len(coeffs)}")
            # zero-pad and reduce each coefficient into [0, Q-1]
            self.coeffs = [int(c) % Q for c in coeffs] + [0] * (N - len(coeffs))

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: Polynomial) -> Polynomial:
        """Coefficient-wise addition mod q."""
        return Polynomial([(self.coeffs[i] + other.coeffs[i]) % Q for i in range(N)])

    def __sub__(self, other: Polynomial) -> Polynomial:
        """Coefficient-wise subtraction mod q."""
        return Polynomial([(self.coeffs[i] - other.coeffs[i]) % Q for i in range(N)])

    def __neg__(self) -> Polynomial:
        """Additive inverse: -p mod q."""
        return Polynomial([(-c) % Q for c in self.coeffs])

    def __mul__(self, other: Polynomial) -> Polynomial:
        """
        Schoolbook multiplication in R_q = Z_q[x] / (x^256 + 1).

        For each pair of terms a*x^i and b*x^j:
          - If i+j < N  : contributes  a*b to coefficient i+j
          - If i+j >= N : contributes -a*b to coefficient (i+j-N)
            because x^N ≡ -1 (mod x^N + 1)

        Complexity: O(N^2) — schoolbook, replaced by NTT in milestone 2.3.
        """
        result = [0] * N
        for i in range(N):
            if self.coeffs[i] == 0:
                continue
            for j in range(N):
                if other.coeffs[j] == 0:
                    continue
                idx = i + j
                if idx < N:
                    result[idx] = (result[idx] + self.coeffs[i] * other.coeffs[j]) % Q
                else:
                    # x^N ≡ -1  =>  subtract instead of add
                    result[idx - N] = (result[idx - N] - self.coeffs[i] * other.coeffs[j]) % Q
        return Polynomial(result)

    # ------------------------------------------------------------------
    # Comparison / helpers
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Polynomial):
            return NotImplemented
        return self.coeffs == other.coeffs

    def __repr__(self) -> str:
        # show only non-zero coefficients for readability
        terms = [(i, c) for i, c in enumerate(self.coeffs) if c != 0]
        if not terms:
            return "Polynomial(0)"
        return f"Polynomial({terms[:8]}{'...' if len(terms) > 8 else ''})"

    def __getitem__(self, index: int) -> int:
        return self.coeffs[index]

    def __len__(self) -> int:
        return N

    def is_zero(self) -> bool:
        """Return True if all coefficients are zero."""
        return all(c == 0 for c in self.coeffs)

    def reduce(self) -> Polynomial:
        """Return a new polynomial with all coefficients reduced mod q."""
        return Polynomial(self.coeffs)


def zero_poly() -> Polynomial:
    """Return the zero polynomial."""
    return Polynomial()


def one_poly() -> Polynomial:
    """Return the constant polynomial 1."""
    p = Polynomial()
    p.coeffs[0] = 1
    return p
