"""
Milestone 8.2: Known Answer Test (KAT) vectors for ML-KEM-512.

These vectors were generated deterministically from this implementation
and serve as regression tests — any change to the core algorithms will
cause these to fail, catching accidental breakage.

Vector structure (per entry):
  d, z    : KeyGen seeds
  m, r    : Encaps message and PKE randomness
  ek, dk  : ML-KEM encapsulation/decapsulation keys
  c, K    : ML-KEM ciphertext and shared secret
  ek_pke, dk_pke, c_pke : K-PKE level keys and ciphertext

Tests verify:
  ✓ KeyGen output matches stored ek and dk
  ✓ Encaps output matches stored K and c
  ✓ Decaps output matches stored K
  ✓ K-PKE.KeyGen matches stored ek_pke, dk_pke
  ✓ K-PKE.Encrypt matches stored c_pke
  ✓ K-PKE.Decrypt recovers m from c_pke
  ✓ All 5 vectors pass
"""

import json
import os

import pytest

from ml_kem_512.kem.decaps import decaps
from ml_kem_512.kem.encaps import encaps
from ml_kem_512.kem.keygen import keygen
from ml_kem_512.pke.decrypt import decrypt_pke
from ml_kem_512.pke.encrypt import encrypt_pke
from ml_kem_512.pke.keygen import keygen_pke

VECTORS_PATH = os.path.join(os.path.dirname(__file__), "..", "vectors", "ml_kem_512_vectors.json")


def load_vectors():
    with open(VECTORS_PATH) as f:
        return json.load(f)


VECTORS = load_vectors()


# ---------------------------------------------------------------------------
# ML-KEM level vector tests
# ---------------------------------------------------------------------------


class TestMLKEMVectors:
    @pytest.mark.parametrize("vec", VECTORS, ids=[f"vec_{v['index']}" for v in VECTORS])
    def test_keygen_matches_vector(self, vec):
        """KeyGen(d, z) produces ek and dk matching stored vectors."""
        d = bytes.fromhex(vec["d"])
        z = bytes.fromhex(vec["z"])
        ek_expected = bytes.fromhex(vec["ek"])
        dk_expected = bytes.fromhex(vec["dk"])

        ek, dk = keygen(d, z)

        assert ek == ek_expected, f"ek mismatch for vector {vec['index']}"
        assert dk == dk_expected, f"dk mismatch for vector {vec['index']}"

    @pytest.mark.parametrize("vec", VECTORS, ids=[f"vec_{v['index']}" for v in VECTORS])
    def test_encaps_matches_vector(self, vec):
        """Encaps(ek, m) produces K and c matching stored vectors."""
        ek = bytes.fromhex(vec["ek"])
        m = bytes.fromhex(vec["m"])
        K_expected = bytes.fromhex(vec["K"])
        c_expected = bytes.fromhex(vec["c"])

        K, c = encaps(ek, m)

        assert K == K_expected, f"K mismatch for vector {vec['index']}"
        assert c == c_expected, f"c mismatch for vector {vec['index']}"

    @pytest.mark.parametrize("vec", VECTORS, ids=[f"vec_{v['index']}" for v in VECTORS])
    def test_decaps_matches_vector(self, vec):
        """Decaps(c, dk) produces K matching stored vector."""
        c = bytes.fromhex(vec["c"])
        dk = bytes.fromhex(vec["dk"])
        K_expected = bytes.fromhex(vec["K"])

        K = decaps(c, dk)

        assert K == K_expected, f"K mismatch for vector {vec['index']}"

    @pytest.mark.parametrize("vec", VECTORS, ids=[f"vec_{v['index']}" for v in VECTORS])
    def test_full_flow_matches_vector(self, vec):
        """Full KeyGen -> Encaps -> Decaps matches stored vector."""
        d = bytes.fromhex(vec["d"])
        z = bytes.fromhex(vec["z"])
        m = bytes.fromhex(vec["m"])
        K_expected = bytes.fromhex(vec["K"])

        ek, dk = keygen(d, z)
        K_enc, c = encaps(ek, m)
        K_dec = decaps(c, dk)

        assert K_enc == K_expected
        assert K_dec == K_expected
        assert K_enc == K_dec


# ---------------------------------------------------------------------------
# K-PKE level vector tests
# ---------------------------------------------------------------------------


class TestKPKEVectors:
    @pytest.mark.parametrize("vec", VECTORS, ids=[f"vec_{v['index']}" for v in VECTORS])
    def test_keygen_pke_matches_vector(self, vec):
        """K-PKE.KeyGen(d) produces ek_pke and dk_pke matching stored vectors."""
        d = bytes.fromhex(vec["d"])
        ek_pke_expected = bytes.fromhex(vec["ek_pke"])
        dk_pke_expected = bytes.fromhex(vec["dk_pke"])

        ek_pke, dk_pke = keygen_pke(d)

        assert ek_pke == ek_pke_expected
        assert dk_pke == dk_pke_expected

    @pytest.mark.parametrize("vec", VECTORS, ids=[f"vec_{v['index']}" for v in VECTORS])
    def test_encrypt_pke_matches_vector(self, vec):
        """K-PKE.Encrypt(ek, m, r) produces c_pke matching stored vector."""
        ek_pke = bytes.fromhex(vec["ek_pke"])
        m = bytes.fromhex(vec["m"])
        r = bytes.fromhex(vec["r"])
        c_pke_expected = bytes.fromhex(vec["c_pke"])

        c_pke = encrypt_pke(ek_pke, m, r)

        assert c_pke == c_pke_expected

    @pytest.mark.parametrize("vec", VECTORS, ids=[f"vec_{v['index']}" for v in VECTORS])
    def test_decrypt_pke_recovers_message(self, vec):
        """K-PKE.Decrypt(dk_pke, c_pke) recovers original message m."""
        dk_pke = bytes.fromhex(vec["dk_pke"])
        c_pke = bytes.fromhex(vec["c_pke"])
        m_expected = bytes.fromhex(vec["m"])

        m_recovered = decrypt_pke(dk_pke, c_pke)

        assert m_recovered == m_expected


# ---------------------------------------------------------------------------
# Size consistency checks across all vectors
# ---------------------------------------------------------------------------


class TestVectorSizes:
    @pytest.mark.parametrize("vec", VECTORS, ids=[f"vec_{v['index']}" for v in VECTORS])
    def test_sizes(self, vec):
        """All stored values have correct ML-KEM-512 sizes."""
        assert len(bytes.fromhex(vec["ek"])) == 800
        assert len(bytes.fromhex(vec["dk"])) == 1632
        assert len(bytes.fromhex(vec["c"])) == 768
        assert len(bytes.fromhex(vec["K"])) == 32
        assert len(bytes.fromhex(vec["ek_pke"])) == 800
        assert len(bytes.fromhex(vec["dk_pke"])) == 768
        assert len(bytes.fromhex(vec["c_pke"])) == 768
