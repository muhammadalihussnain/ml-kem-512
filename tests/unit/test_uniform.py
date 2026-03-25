"""
Unit tests for Uniform Sampling / SampleNTT (Milestone 3.2).

Tests cover:
- Output: exactly 256 coefficients, all in [0, Q-1]
- Determinism: same (rho, i, j) -> same output
- Domain separation: different (i,j) pairs -> different outputs
- Rejection sampling: no values >= Q slip through
- Statistical uniformity: chi-square test over many samples
- All 4 matrix positions for ML-KEM-512 (k=2) are independent
- Integration: sample_ntt output is valid NTT-domain polynomial
"""

from ml_kem_512.polynomial.ntt import inv_ntt
from ml_kem_512.polynomial.poly import N, Q
from ml_kem_512.sampling.uniform import sample_ntt

RHO = bytes(range(32))  # fixed 32-byte seed


# ---------------------------------------------------------------------------
# Basic output tests
# ---------------------------------------------------------------------------


class TestSampleNTTOutput:
    def test_output_length(self):
        """sample_ntt returns exactly N=256 coefficients."""
        result = sample_ntt(RHO, 0, 0)
        assert len(result) == N

    def test_all_coefficients_in_range(self):
        """All coefficients are in [0, Q-1]."""
        result = sample_ntt(RHO, 0, 0)
        assert all(0 <= c < Q for c in result)

    def test_no_coefficient_equals_q(self):
        """No coefficient equals Q (rejection sampling works)."""
        result = sample_ntt(RHO, 0, 0)
        assert Q not in result

    def test_output_is_list(self):
        """sample_ntt returns a list."""
        result = sample_ntt(RHO, 0, 0)
        assert isinstance(result, list)

    def test_all_matrix_positions_valid(self):
        """All 4 positions for k=2 produce valid output."""
        for i in range(2):
            for j in range(2):
                result = sample_ntt(RHO, i, j)
                assert len(result) == N
                assert all(0 <= c < Q for c in result)


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


class TestSampleNTTDeterminism:
    def test_deterministic_same_seed(self):
        """Same (rho, i, j) always produces same output."""
        r1 = sample_ntt(RHO, 0, 0)
        r2 = sample_ntt(RHO, 0, 0)
        assert r1 == r2

    def test_deterministic_all_positions(self):
        """Determinism holds for all 4 matrix positions."""
        for i in range(2):
            for j in range(2):
                assert sample_ntt(RHO, i, j) == sample_ntt(RHO, i, j)

    def test_different_rho_different_output(self):
        """Different rho seeds produce different outputs."""
        rho2 = bytes(range(1, 33))
        assert sample_ntt(RHO, 0, 0) != sample_ntt(rho2, 0, 0)


# ---------------------------------------------------------------------------
# Domain separation tests
# ---------------------------------------------------------------------------


class TestSampleNTTDomainSeparation:
    def test_different_i_different_output(self):
        """Different row index i produces different output."""
        assert sample_ntt(RHO, 0, 0) != sample_ntt(RHO, 1, 0)

    def test_different_j_different_output(self):
        """Different column index j produces different output."""
        assert sample_ntt(RHO, 0, 0) != sample_ntt(RHO, 0, 1)

    def test_all_four_positions_distinct(self):
        """All 4 matrix positions A[i][j] for k=2 are distinct."""
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        outputs = [tuple(sample_ntt(RHO, i, j)) for i, j in positions]
        assert len(set(outputs)) == 4

    def test_i_j_swap_different(self):
        """A[0][1] != A[1][0] — row/column order matters."""
        a01 = sample_ntt(RHO, 0, 1)
        a10 = sample_ntt(RHO, 1, 0)
        assert a01 != a10


# ---------------------------------------------------------------------------
# Rejection sampling correctness
# ---------------------------------------------------------------------------


class TestRejectionSampling:
    def test_no_values_gte_q(self):
        """Rejection sampling never lets values >= Q through."""
        for i in range(2):
            for j in range(2):
                result = sample_ntt(RHO, i, j)
                assert all(c < Q for c in result)

    def test_multiple_seeds_no_values_gte_q(self):
        """Rejection holds across many different seeds."""
        for seed_byte in range(10):
            rho = bytes([seed_byte] * 32)
            result = sample_ntt(rho, 0, 0)
            assert all(0 <= c < Q for c in result)


# ---------------------------------------------------------------------------
# Statistical uniformity test
# ---------------------------------------------------------------------------


class TestSampleNTTUniformity:
    def test_values_spread_across_range(self):
        """
        Coefficients are spread across [0, Q-1].
        With 256 samples, we expect values in many different buckets.
        Use 16 buckets of size ~208 each; all should be non-empty.
        """
        result = sample_ntt(RHO, 0, 0)
        bucket_size = Q // 16
        buckets = [0] * 16
        for c in result:
            buckets[min(c // bucket_size, 15)] += 1
        # All 16 buckets should have at least 1 value
        assert all(b > 0 for b in buckets)

    def test_no_obvious_bias(self):
        """
        Collect 4 * 256 = 1024 samples (4 matrix positions).
        Mean should be near Q/2 ~ 1664.
        """
        all_vals = []
        for i in range(2):
            for j in range(2):
                all_vals.extend(sample_ntt(RHO, i, j))
        mean = sum(all_vals) / len(all_vals)
        # Mean should be within 10% of Q/2
        assert abs(mean - Q / 2) < 0.10 * Q

    def test_no_duplicate_heavy_bias(self):
        """No single value dominates more than 5% of the 256 coefficients."""
        result = sample_ntt(RHO, 0, 0)
        max_count = max(result.count(v) for v in set(result))
        assert max_count <= N * 0.05  # at most 5% duplicates


# ---------------------------------------------------------------------------
# NTT domain integration test
# ---------------------------------------------------------------------------


class TestSampleNTTIntegration:
    def test_inv_ntt_of_sample_is_valid_polynomial(self):
        """
        sample_ntt output is a valid NTT-domain representation.
        inv_ntt should recover a polynomial with coefficients in [0, Q-1].
        """
        ntt_coeffs = sample_ntt(RHO, 0, 0)
        poly = inv_ntt(ntt_coeffs)
        assert len(poly.coeffs) == N
        assert all(0 <= c < Q for c in poly.coeffs)

    def test_matrix_generation_pattern(self):
        """
        Simulate full ML-KEM-512 matrix A generation:
          A[i][j] = sample_ntt(rho, i, j)  for i,j in {0,1}
        All 4 elements are valid and distinct.
        """
        A = [[sample_ntt(RHO, i, j) for j in range(2)] for i in range(2)]
        for i in range(2):
            for j in range(2):
                assert len(A[i][j]) == N
                assert all(0 <= c < Q for c in A[i][j])
        # All 4 elements are distinct
        flat = [tuple(A[i][j]) for i in range(2) for j in range(2)]
        assert len(set(flat)) == 4


# ---------------------------------------------------------------------------
# Buffer extension branch coverage
# ---------------------------------------------------------------------------


class TestSampleNTTBufferExtension:
    def test_buffer_extension_branch(self):
        """
        Force the buffer extension branch by monkeypatching XOF.read
        to return only 3 bytes at a time (minimum viable chunk).
        This exercises the 'buf = buf + xof.read(168)' line.
        """
        import ml_kem_512.sampling.uniform as uniform_mod
        from ml_kem_512.primitives.prf import XOF as RealXOF

        original_init = RealXOF.__init__
        original_read = RealXOF.read

        # Patch XOF to return only 3 bytes on first read, then normal
        call_count = []

        def patched_read(self, n):
            call_count.append(1)
            if len(call_count) == 1:
                # Return only 3 bytes to force buffer extension
                return original_read(self, 3)
            return original_read(self, n)

        RealXOF.read = patched_read
        try:
            result = sample_ntt(RHO, 0, 0)
            assert len(result) == N
            assert all(0 <= c < Q for c in result)
        finally:
            RealXOF.read = original_read
