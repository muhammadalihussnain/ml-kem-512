"""
Unit tests for cryptographic primitives (hash functions).

Tests SHAKE-128, SHAKE-256, SHA3-256, and SHA3-512 implementations
against NIST FIPS 202 test vectors.
"""

import pytest

from ml_kem_512.primitives.hash import sha3_256, sha3_512, shake128, shake256


class TestSHAKE128:
    """Test SHAKE-128 extendable output function."""

    def test_empty_input(self):
        """Test SHAKE-128 with empty input."""
        # NIST test vector for empty message
        result = shake128(b"", 32)
        expected = bytes.fromhex("7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26")
        assert result == expected

    def test_short_message(self):
        """Test SHAKE-128 with short message."""
        # Test with simple message
        result = shake128(b"abc", 32)
        expected = bytes.fromhex("5881092dd818bf5cf8a3ddb793fbcba74097d5c526a6d35f97b83351940f2cc8")
        assert result == expected

    def test_variable_output_length(self):
        """Test SHAKE-128 can generate arbitrary-length output."""
        data = b"test"

        # Generate different lengths
        output_16 = shake128(data, 16)
        output_32 = shake128(data, 32)
        output_64 = shake128(data, 64)
        output_128 = shake128(data, 128)

        assert len(output_16) == 16
        assert len(output_32) == 32
        assert len(output_64) == 64
        assert len(output_128) == 128

        # Verify that shorter outputs are prefixes of longer ones
        assert output_32[:16] == output_16
        assert output_64[:32] == output_32
        assert output_128[:64] == output_64

    def test_deterministic(self):
        """Test SHAKE-128 produces deterministic output."""
        data = b"deterministic test"
        result1 = shake128(data, 32)
        result2 = shake128(data, 32)
        assert result1 == result2

    def test_different_inputs_different_outputs(self):
        """Test different inputs produce different outputs."""
        result1 = shake128(b"input1", 32)
        result2 = shake128(b"input2", 32)
        assert result1 != result2


class TestSHAKE256:
    """Test SHAKE-256 extendable output function."""

    def test_empty_input(self):
        """Test SHAKE-256 with empty input."""
        # NIST test vector for empty message
        result = shake256(b"", 32)
        expected = bytes.fromhex("46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f")
        assert result == expected

    def test_short_message(self):
        """Test SHAKE-256 with short message."""
        result = shake256(b"abc", 32)
        expected = bytes.fromhex("483366601360a8771c6863080cc4114d8db44530f8f1e1ee4f94ea37e78b5739")
        assert result == expected

    def test_variable_output_length(self):
        """Test SHAKE-256 can generate arbitrary-length output."""
        data = b"test"

        output_16 = shake256(data, 16)
        output_32 = shake256(data, 32)
        output_64 = shake256(data, 64)
        output_256 = shake256(data, 256)

        assert len(output_16) == 16
        assert len(output_32) == 32
        assert len(output_64) == 64
        assert len(output_256) == 256

        # Verify prefixes
        assert output_32[:16] == output_16
        assert output_64[:32] == output_32
        assert output_256[:64] == output_64

    def test_deterministic(self):
        """Test SHAKE-256 produces deterministic output."""
        data = b"deterministic test"
        result1 = shake256(data, 64)
        result2 = shake256(data, 64)
        assert result1 == result2

    def test_different_inputs_different_outputs(self):
        """Test different inputs produce different outputs."""
        result1 = shake256(b"input1", 32)
        result2 = shake256(b"input2", 32)
        assert result1 != result2


class TestSHA3_256:
    """Test SHA3-256 hash function."""

    def test_empty_input(self):
        """Test SHA3-256 with empty input."""
        # NIST test vector
        result = sha3_256(b"")
        expected = bytes.fromhex("a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a")
        assert result == expected

    def test_short_message(self):
        """Test SHA3-256 with 'abc'."""
        result = sha3_256(b"abc")
        expected = bytes.fromhex("3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532")
        assert result == expected

    def test_output_length(self):
        """Test SHA3-256 always outputs 32 bytes."""
        assert len(sha3_256(b"")) == 32
        assert len(sha3_256(b"short")) == 32
        assert len(sha3_256(b"a" * 1000)) == 32

    def test_deterministic(self):
        """Test SHA3-256 produces deterministic output."""
        data = b"test message"
        result1 = sha3_256(data)
        result2 = sha3_256(data)
        assert result1 == result2

    def test_different_inputs_different_outputs(self):
        """Test different inputs produce different outputs."""
        result1 = sha3_256(b"message1")
        result2 = sha3_256(b"message2")
        assert result1 != result2

    def test_long_message(self):
        """Test SHA3-256 with longer message."""
        # Test with a longer message
        message = b"a" * 200
        result = sha3_256(message)
        assert len(result) == 32
        # Verify it's deterministic
        assert result == sha3_256(message)


class TestSHA3_512:
    """Test SHA3-512 hash function."""

    def test_empty_input(self):
        """Test SHA3-512 with empty input."""
        # NIST test vector
        result = sha3_512(b"")
        expected = bytes.fromhex(
            "a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc1475c80a6"
            "15b2123af1f5f94c11e3e9402c3ac558f500199d95b6d3e301758586281dcd26"
        )
        assert result == expected

    def test_short_message(self):
        """Test SHA3-512 with 'abc'."""
        result = sha3_512(b"abc")
        expected = bytes.fromhex(
            "b751850b1a57168a5693cd924b6b096e08f621827444f70d884f5d0240d2712e"
            "10e116e9192af3c91a7ec57647e3934057340b4cf408d5a56592f8274eec53f0"
        )
        assert result == expected

    def test_output_length(self):
        """Test SHA3-512 always outputs 64 bytes."""
        assert len(sha3_512(b"")) == 64
        assert len(sha3_512(b"short")) == 64
        assert len(sha3_512(b"a" * 1000)) == 64

    def test_deterministic(self):
        """Test SHA3-512 produces deterministic output."""
        data = b"test message"
        result1 = sha3_512(data)
        result2 = sha3_512(data)
        assert result1 == result2

    def test_different_inputs_different_outputs(self):
        """Test different inputs produce different outputs."""
        result1 = sha3_512(b"message1")
        result2 = sha3_512(b"message2")
        assert result1 != result2

    def test_long_message(self):
        """Test SHA3-512 with longer message."""
        message = b"The quick brown fox jumps over the lazy dog"
        result = sha3_512(message)
        expected = bytes.fromhex(
            "01dedd5de4ef14642445ba5f5b97c15e47b9ad931326e4b0727cd94cefc44fff"
            "23f07bf543139939b49128caf436dc1bdee54fcb24023a08d9403f9b4bf0d450"
        )
        assert result == expected


class TestCrossFunction:
    """Test interactions between different hash functions."""

    def test_shake_vs_sha3_different(self):
        """Verify SHAKE and SHA3 produce different outputs for same input."""
        data = b"test"
        shake128_out = shake128(data, 32)
        shake256_out = shake256(data, 32)
        sha3_256_out = sha3_256(data)

        # All should be different
        assert shake128_out != shake256_out
        assert shake128_out != sha3_256_out
        assert shake256_out != sha3_256_out

    def test_all_functions_work_with_binary_data(self):
        """Test all functions handle arbitrary binary data."""
        binary_data = bytes(range(256))

        # All should work without errors
        shake128(binary_data, 32)
        shake256(binary_data, 32)
        sha3_256(binary_data)
        sha3_512(binary_data)
