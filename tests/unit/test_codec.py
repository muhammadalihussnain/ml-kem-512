"""
Unit tests for ByteEncode / ByteDecode (Milestone 3.4).

Tests cover:
- byte_encode: correct output byte length (32*d)
- byte_decode: correct output list length (N=256)
- Roundtrip: ByteDecode(ByteEncode(values)) == values
- Bit packing correctness for d=1,4,10,12
- Input validation: wrong list length / wrong byte length
- encode_poly / decode_poly wrappers
- ML-KEM-512 specific sizes: pk=800, sk=768, ciphertext=768
"""

import pytest

from ml_kem_512.encoding.codec import byte_decode, byte_encode, decode_poly, encode_poly
from ml_kem_512.polynomial.poly import N, Polynomial, Q, zero_poly

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_values(d: int) -> list:
    """Generate N values cycling through [0, 2^d - 1]."""
    return [i % (1 << d) for i in range(N)]


# ---------------------------------------------------------------------------
# byte_encode output length tests
# ---------------------------------------------------------------------------


class TestByteEncodeLength:
    def test_length_d1(self):
        """d=1: 256 bits = 32 bytes."""
        assert len(byte_encode(1, make_values(1))) == 32

    def test_length_d4(self):
        """d=4: 256*4 bits = 128 bytes."""
        assert len(byte_encode(4, make_values(4))) == 128

    def test_length_d10(self):
        """d=10: 256*10 bits = 320 bytes."""
        assert len(byte_encode(10, make_values(10))) == 320

    def test_length_d12(self):
        """d=12: 256*12 bits = 384 bytes."""
        assert len(byte_encode(12, make_values(12))) == 384

    def test_wrong_input_length_raises(self):
        """byte_encode raises ValueError if not exactly N values."""
        with pytest.raises(ValueError, match="256"):
            byte_encode(4, [0] * 10)


# ---------------------------------------------------------------------------
# byte_decode output length tests
# ---------------------------------------------------------------------------


class TestByteDecodeLength:
    def test_length_d1(self):
        """d=1: 32 bytes -> 256 values."""
        result = byte_decode(1, bytes(32))
        assert len(result) == N

    def test_length_d4(self):
        """d=4: 128 bytes -> 256 values."""
        result = byte_decode(4, bytes(128))
        assert len(result) == N

    def test_length_d10(self):
        """d=10: 320 bytes -> 256 values."""
        result = byte_decode(10, bytes(320))
        assert len(result) == N

    def test_length_d12(self):
        """d=12: 384 bytes -> 256 values."""
        result = byte_decode(12, bytes(384))
        assert len(result) == N

    def test_wrong_byte_length_raises(self):
        """byte_decode raises ValueError for wrong byte length."""
        with pytest.raises(ValueError, match="128"):
            byte_decode(4, bytes(64))


# ---------------------------------------------------------------------------
# Roundtrip tests
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_roundtrip_d1(self):
        """ByteDecode(ByteEncode(v, 1), 1) == v."""
        values = make_values(1)
        assert byte_decode(1, byte_encode(1, values)) == values

    def test_roundtrip_d4(self):
        """ByteDecode(ByteEncode(v, 4), 4) == v."""
        values = make_values(4)
        assert byte_decode(4, byte_encode(4, values)) == values

    def test_roundtrip_d10(self):
        """ByteDecode(ByteEncode(v, 10), 10) == v."""
        values = make_values(10)
        assert byte_decode(10, byte_encode(10, values)) == values

    def test_roundtrip_d12(self):
        """ByteDecode(ByteEncode(v, 12), 12) == v."""
        values = make_values(12)
        assert byte_decode(12, byte_encode(12, values)) == values

    def test_roundtrip_all_zeros(self):
        """All-zero values roundtrip correctly."""
        for d in [1, 4, 10, 12]:
            values = [0] * N
            assert byte_decode(d, byte_encode(d, values)) == values

    def test_roundtrip_all_max(self):
        """All-max values (2^d - 1) roundtrip correctly."""
        for d in [1, 4, 10, 12]:
            values = [(1 << d) - 1] * N
            assert byte_decode(d, byte_encode(d, values)) == values

    def test_roundtrip_alternating(self):
        """Alternating 0 and max values roundtrip correctly."""
        for d in [4, 10, 12]:
            max_val = (1 << d) - 1
            values = [max_val if i % 2 == 0 else 0 for i in range(N)]
            assert byte_decode(d, byte_encode(d, values)) == values


# ---------------------------------------------------------------------------
# Bit packing correctness tests
# ---------------------------------------------------------------------------


class TestBitPacking:
    def test_single_bit_value_1(self):
        """d=1: value=1 at position 0 sets bit 0 of byte 0."""
        values = [0] * N
        values[0] = 1
        encoded = byte_encode(1, values)
        assert encoded[0] & 1 == 1  # LSB of first byte is set

    def test_single_bit_value_0(self):
        """d=1: all zeros -> all bytes are 0."""
        encoded = byte_encode(1, [0] * N)
        assert all(b == 0 for b in encoded)

    def test_d4_first_value(self):
        """d=4: first value occupies low nibble of first byte."""
        values = [0] * N
        values[0] = 0xA  # 0b1010
        encoded = byte_encode(4, values)
        assert encoded[0] & 0x0F == 0xA

    def test_d4_second_value(self):
        """d=4: second value occupies high nibble of first byte."""
        values = [0] * N
        values[1] = 0x5  # 0b0101
        encoded = byte_encode(4, values)
        assert (encoded[0] >> 4) & 0x0F == 0x5

    def test_d8_single_byte_per_value(self):
        """d=8: each value maps to exactly one byte."""
        values = list(range(N))  # 0..255
        encoded = byte_encode(8, values)
        assert len(encoded) == N
        decoded = byte_decode(8, encoded)
        assert decoded == values

    def test_d12_known_value(self):
        """d=12: value 0xABC encodes and decodes correctly."""
        values = [0] * N
        values[0] = 0xABC
        encoded = byte_encode(12, values)
        decoded = byte_decode(12, encoded)
        assert decoded[0] == 0xABC

    def test_deterministic(self):
        """Same values always produce same bytes."""
        values = make_values(10)
        assert byte_encode(10, values) == byte_encode(10, values)


# ---------------------------------------------------------------------------
# encode_poly / decode_poly wrapper tests
# ---------------------------------------------------------------------------


class TestPolyCodec:
    def test_encode_poly_length_d12(self):
        """encode_poly(12, poly) produces 384 bytes."""
        poly = Polynomial([i % Q for i in range(N)])
        assert len(encode_poly(12, poly)) == 384

    def test_encode_poly_length_d10(self):
        """encode_poly(10, poly) produces 320 bytes."""
        poly = Polynomial([i % (1 << 10) for i in range(N)])
        assert len(encode_poly(10, poly)) == 320

    def test_decode_poly_returns_polynomial(self):
        """decode_poly returns a Polynomial."""
        data = bytes(384)
        result = decode_poly(12, data)
        assert isinstance(result, Polynomial)

    def test_poly_roundtrip_d12(self):
        """decode_poly(12, encode_poly(12, p)) == p for d=12."""
        # d=12 is lossless for coefficients in [0, 2^12-1]
        poly = Polynomial([i % (1 << 12) for i in range(N)])
        assert decode_poly(12, encode_poly(12, poly)) == poly

    def test_poly_roundtrip_d4(self):
        """decode_poly(4, encode_poly(4, p)) == p for d=4."""
        poly = Polynomial([i % (1 << 4) for i in range(N)])
        assert decode_poly(4, encode_poly(4, poly)) == poly

    def test_zero_poly_roundtrip(self):
        """Zero polynomial encodes and decodes to zero."""
        for d in [1, 4, 10, 12]:
            assert decode_poly(d, encode_poly(d, zero_poly())) == zero_poly()


# ---------------------------------------------------------------------------
# ML-KEM-512 size verification tests
# ---------------------------------------------------------------------------


class TestMLKEM512Sizes:
    def test_public_key_t_encoding_size(self):
        """
        Public key t_hat: 2 polynomials * 384 bytes = 768 bytes.
        Plus 32-byte rho seed = 800 bytes total.
        """
        poly = Polynomial([i % (1 << 12) for i in range(N)])
        t0 = encode_poly(12, poly)
        t1 = encode_poly(12, poly)
        ek = t0 + t1 + bytes(32)  # rho
        assert len(ek) == 800

    def test_secret_key_encoding_size(self):
        """
        Secret key s_hat: 2 polynomials * 384 bytes = 768 bytes.
        """
        poly = Polynomial([i % (1 << 12) for i in range(N)])
        s0 = encode_poly(12, poly)
        s1 = encode_poly(12, poly)
        dk = s0 + s1
        assert len(dk) == 768

    def test_ciphertext_u_encoding_size(self):
        """
        Ciphertext u: 2 polynomials * 320 bytes (d=10) = 640 bytes.
        """
        poly = Polynomial([i % (1 << 10) for i in range(N)])
        u0 = encode_poly(10, poly)
        u1 = encode_poly(10, poly)
        assert len(u0 + u1) == 640

    def test_ciphertext_v_encoding_size(self):
        """
        Ciphertext v: 1 polynomial * 128 bytes (d=4) = 128 bytes.
        """
        poly = Polynomial([i % (1 << 4) for i in range(N)])
        v = encode_poly(4, poly)
        assert len(v) == 128

    def test_full_ciphertext_size(self):
        """
        Full ciphertext c = encode(u0) + encode(u1) + encode(v)
        = 320 + 320 + 128 = 768 bytes.
        """
        u_poly = Polynomial([i % (1 << 10) for i in range(N)])
        v_poly = Polynomial([i % (1 << 4) for i in range(N)])
        c = encode_poly(10, u_poly) + encode_poly(10, u_poly) + encode_poly(4, v_poly)
        assert len(c) == 768
