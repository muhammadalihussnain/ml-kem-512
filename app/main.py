"""
ML-KEM Interactive Desktop Application.

Select ML-KEM-512 / 768 / 1024 and run each major cryptographic component
step-by-step from the UI.
"""

import os
import sys
import time
import tkinter as tk
from tkinter import ttk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml_kem_generic import (
    MLKEM,
    PARAMS,
    _G,
    _H,
    _J,
    _byte_decode,
    _byte_encode,
    _cbd,
    _compress,
    _decompress,
    _inv_ntt,
    _ntt,
    _poly_add,
    _poly_sub,
    _prf,
    _sample_ntt,
    _sha3_256,
    _sha3_512,
    _shake128,
    _shake256,
)

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
BG = "#0d1117"
PANEL = "#161b22"
BORDER = "#30363d"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
YELLOW = "#d29922"
RED = "#f85149"
PURPLE = "#bc8cff"
CYAN = "#79c0ff"
TEXT = "#e6edf3"
MUTED = "#8b949e"
STEP_BG = "#1c2128"


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class MLKEMApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ML-KEM Interactive Workflow — FIPS 203")
        self.geometry("1100x780")
        self.configure(bg=BG)
        self.resizable(True, True)

        self._engine = MLKEM("ML-KEM-512")
        self._output_rows = []
        self._component_buttons = {}
        self._state = {}

        self._build_ui()
        self._on_variant_change()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=PANEL, pady=12)
        hdr.pack(fill="x")

        tk.Label(hdr, text="ML-KEM", font=("Courier", 22, "bold"),
                 bg=PANEL, fg=ACCENT).pack(side="left", padx=20)
        tk.Label(hdr, text="Module-Lattice Key Encapsulation Mechanism  |  FIPS 203",
                 font=("Courier", 11), bg=PANEL, fg=MUTED).pack(side="left")

        # ── Controls ────────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=BG, pady=10)
        ctrl.pack(fill="x", padx=20)

        tk.Label(ctrl, text="Select Variant:", font=("Courier", 12, "bold"),
                 bg=BG, fg=TEXT).pack(side="left")

        self._variant = tk.StringVar(value="ML-KEM-512")
        combo = ttk.Combobox(ctrl, textvariable=self._variant,
                             values=list(PARAMS.keys()),
                             state="readonly", width=16,
                             font=("Courier", 12))
        combo.pack(side="left", padx=10)

        combo.bind("<<ComboboxSelected>>", lambda _e: self._on_variant_change())

        # param badge
        self._param_label = tk.Label(ctrl, text="", font=("Courier", 10),
                                     bg=BG, fg=YELLOW)
        self._param_label.pack(side="left", padx=20)

        clear_btn = tk.Button(ctrl, text="Clear Output",
                              font=("Courier", 11, "bold"),
                              bg=PANEL, fg=TEXT, relief="flat",
                              padx=12, pady=4, command=self._clear_output)
        clear_btn.pack(side="right")

        # ── Workflow map + dynamic components ───────────────────────────
        workflow_row = tk.Frame(self, bg=BG, pady=6)
        workflow_row.pack(fill="x", padx=20)

        self._workflow_tree = tk.Frame(
            workflow_row,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1,
            padx=10,
            pady=8,
        )
        self._workflow_tree.pack(side="left", fill="y")

        right_panel = tk.Frame(workflow_row, bg=BG)
        right_panel.pack(side="left", fill="both", expand=True, padx=(10, 0))
        tk.Label(
            right_panel,
            text="Component Actions",
            font=("Courier", 11, "bold"),
            bg=BG,
            fg=TEXT,
        ).pack(anchor="w")
        self._component_wrap = tk.Frame(right_panel, bg=BG)
        self._component_wrap.pack(fill="x", pady=(6, 0))

        # ── Scrollable output area ───────────────────────────────────────
        outer = tk.Frame(self, bg=BG)
        outer.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        self._scroll_frame = tk.Frame(canvas, bg=BG)

        self._scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self._scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self._canvas = canvas

        # mouse wheel scroll
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        # ── Status bar ──────────────────────────────────────────────────
        self._status = tk.Label(self, text="Ready. Select a variant and click any component button.",
                                font=("Courier", 10), bg=PANEL, fg=MUTED,
                                anchor="w", padx=12, pady=4)
        self._status.pack(fill="x", side="bottom")

    # ------------------------------------------------------------------
    # Variant / workflow control
    # ------------------------------------------------------------------

    def _on_variant_change(self):
        variant = self._variant.get()
        self._engine = MLKEM(variant)
        self._state = {
            # These will be generated by the RNG button (or implicitly by
            # component buttons via ensure_* helpers).
            "d": None,
            "z": None,
            "m": None,
            "ek": None,
            "dk": None,
            "c": None,
            "k_enc": None,
            # derived intermediates (populated by component buttons)
            "rho": None,
            "sigma": None,
            "g_input": None,
            "s_poly": None,
            "e_poly": None,
            "s_hat": None,
            "a00": None,
        }
        self._render_workflow_tree()
        self._render_component_buttons()
        self._status_update(f"Switched to {variant}.")
        self._clear_output()
        p = PARAMS[variant]
        self._param_label.config(
            text=f"k={p['k']}  eta1={p['eta1']}  eta2={p['eta2']}  du={p['du']}  dv={p['dv']}"
        )

    def _clear_output(self):
        for w in self._scroll_frame.winfo_children():
            w.destroy()
        self._output_rows.clear()

    # ------------------------------------------------------------------
    # Dynamic component buttons
    # ------------------------------------------------------------------

    def _component_definitions(self):
        return [
            ("Phase 1: Primitives", "Random Number Generation", self._run_rng),
            ("Phase 1: Primitives", "SHA3-512", self._run_sha3_512),
            ("Phase 1: Primitives", "SHAKE (128/256)", self._run_shake),
            ("Phase 1: Primitives", "PRF", self._run_prf),
            ("Phase 1: Primitives", "XOF / SampleNTT", self._run_xof),
            ("Phase 2: Arithmetic", "NTT / InvNTT", self._run_ntt),
            ("Phase 2: Arithmetic", "Polynomial Ops", self._run_poly_ops),
            ("Phase 3: KEM Core", "KDF (G,H,J)", self._run_kdf),
            ("Phase 3: KEM Core", "Key Generation", self._run_keygen),
            ("Phase 3: KEM Core", "Encapsulation", self._run_encapsulation),
            ("Phase 3: KEM Core", "Decapsulation", self._run_decapsulation),
            ("Phase 4: Validation", "End-to-End Check", self._run_end_to_end),
        ]

    def _render_workflow_tree(self):
        for child in self._workflow_tree.winfo_children():
            child.destroy()
        tk.Label(
            self._workflow_tree,
            text=f"{self._variant.get()} Workflow",
            font=("Courier", 10, "bold"),
            bg=PANEL,
            fg=YELLOW,
            anchor="w",
        ).pack(fill="x", pady=(0, 6))
        phases = [
            "Phase 1: Primitives",
            "Phase 2: Arithmetic",
            "Phase 3: KEM Core",
            "Phase 4: Validation",
        ]
        for phase in phases:
            tk.Label(
                self._workflow_tree,
                text=f"-> {phase}",
                font=("Courier", 10),
                bg=PANEL,
                fg=TEXT,
                anchor="w",
            ).pack(fill="x", pady=1)

    def _render_component_buttons(self):
        for child in self._component_wrap.winfo_children():
            child.destroy()
        self._component_buttons.clear()
        grouped = {}
        for phase, label, handler in self._component_definitions():
            grouped.setdefault(phase, []).append((label, handler))

        for idx, phase in enumerate(grouped):
            block = tk.LabelFrame(
                self._component_wrap,
                text=phase,
                font=("Courier", 10, "bold"),
                bg=BG,
                fg=CYAN,
                padx=8,
                pady=6,
                highlightbackground=BORDER,
                highlightthickness=1,
            )
            block.grid(row=idx // 2, column=idx % 2, padx=5, pady=5, sticky="nsew")
            for label, handler in grouped[phase]:
                b = tk.Button(
                    block,
                    text=label,
                    font=("Courier", 9, "bold"),
                    bg=STEP_BG,
                    fg=TEXT,
                    relief="flat",
                    padx=8,
                    pady=4,
                    command=handler,
                )
                b.pack(fill="x", pady=2)
                self._component_buttons[label] = b

    # ------------------------------------------------------------------
    # Component handlers
    # ------------------------------------------------------------------

    def _run_rng(self):
        self._step("Component", "Random Number Generation")
        d = os.urandom(32)
        z = os.urandom(32)
        m = os.urandom(32)
        self._state["d"] = d
        self._state["z"] = z
        self._state["m"] = m
        # reset derived values (they depend on d)
        self._state["ek"] = None
        self._state["dk"] = None
        self._state["c"] = None
        self._state["k_enc"] = None
        self._state["rho"] = None
        self._state["sigma"] = None
        self._state["g_input"] = None
        self._state["s_poly"] = None
        self._state["e_poly"] = None
        self._state["s_hat"] = None
        self._state["a00"] = None
        self._result("d (32B)", d.hex()[:32] + "...", CYAN)
        self._result("z (32B)", z.hex()[:32] + "...", CYAN)
        self._result("m (32B)", m.hex()[:32] + "...", CYAN)
        self._status_update("Random seeds refreshed.")

    # ------------------------------------------------------------------
    # Ensure helpers (avoid mock data; derive from current state)
    # ------------------------------------------------------------------
    def _ensure_rng_inputs(self):
        if self._state.get("d") is None or self._state.get("z") is None or self._state.get("m") is None:
            self._run_rng()

    def _ensure_g_keygen(self):
        """
        Compute and cache (rho, sigma) = G(d || k).
        This is the SHA3-512-based output that drives KeyGen sampling.
        """
        if self._state.get("d") is None:
            self._run_rng()
        if self._state.get("rho") is None or self._state.get("sigma") is None:
            d = self._state["d"]
            k = self._engine.k
            g_input = d + bytes([k])
            rho, sigma = _G(g_input)
            self._state["rho"] = rho
            self._state["sigma"] = sigma
            self._state["g_input"] = g_input

    def _ensure_keygen_sample_polys(self):
        """Compute and cache s_poly and e_poly for visualization."""
        self._ensure_g_keygen()
        if self._state.get("s_poly") is None or self._state.get("e_poly") is None:
            sigma = self._state["sigma"]
            eta1 = self._engine.eta1
            k = self._engine.k
            # In keygen_pke:
            #   s = sample_vec(sigma, start=0, eta=eta1)
            #   e = sample_vec(sigma, start=k, eta=eta1)
            prf_s = _prf(sigma, 0, 64 * eta1)
            prf_e = _prf(sigma, k, 64 * eta1)
            self._state["s_poly"] = _cbd(eta1, prf_s)
            self._state["e_poly"] = _cbd(eta1, prf_e)

    def _ensure_s_hat(self):
        """Compute and cache NTT(s_poly) for visualization."""
        self._ensure_keygen_sample_polys()
        if self._state.get("s_hat") is None:
            self._state["s_hat"] = _ntt(self._state["s_poly"])

    def _ensure_a00(self):
        """Compute and cache SampleNTT(rho, i=0, j=0)."""
        self._ensure_g_keygen()
        if self._state.get("a00") is None:
            self._state["a00"] = _sample_ntt(self._state["rho"], 0, 0)

    def _run_sha3_512(self):
        self._step("Component", "SHA3-512")
        self._ensure_g_keygen()
        g_input = self._state["g_input"]
        rho = self._state["rho"]
        sigma = self._state["sigma"]
        self._result("Input for G(d||k)", f"{g_input.hex()[:32]}...", MUTED)
        self._result("rho = G()[0]", rho.hex()[:32] + "...", GREEN)
        self._result("sigma = G()[1]", sigma.hex()[:32] + "...", GREEN)
        self._status_update("SHA3-512 executed for KeyGen (G(d||k)).")

    def _run_shake(self):
        self._step("Component", "SHAKE-128 / SHAKE-256")
        # SHAKE-128 is used by SampleNTT (XOF stream for A_hat),
        # and SHAKE-256 is used by PRF (for CBD sampling).
        self._ensure_a00()
        self._ensure_g_keygen()

        rho = self._state["rho"]
        sigma = self._state["sigma"]

        # SampleNTT uses shake128(rho || i || j) as an XOF stream.
        shake128_preview = _shake128(rho + bytes([0]) + bytes([0]), 32)
        # PRF uses shake256(sigma || b).
        prf_preview = _prf(sigma, 0, 32)

        self._result("SHAKE-128 input (rho||0||0) preview", shake128_preview.hex()[:32] + "...", GREEN)
        self._result("SHAKE-256 input (sigma||b=0) preview", prf_preview.hex()[:32] + "...", GREEN)
        self._status_update("SHAKE executed for XOF/PRF (derived from d).")

    def _run_prf(self):
        self._step("Component", "PRF")
        self._ensure_g_keygen()
        sigma = self._state["sigma"]
        eta1 = self._engine.eta1
        out = _prf(sigma, 0, 64 * eta1)
        self._result("eta1", str(eta1), MUTED)
        self._result("PRF input b", "b=0", MUTED)
        self._result("PRF output length", f"{len(out)} bytes", GREEN)
        self._result("PRF output prefix", out.hex()[:32] + "...", GREEN)
        self._status_update("PRF executed.")

    def _run_xof(self):
        self._step("Component", "XOF / SampleNTT")
        self._ensure_a00()
        self._ensure_g_keygen()
        rho = self._state["rho"]
        sampled = self._state["a00"]
        self._result("rho", rho.hex()[:32] + "...", CYAN)
        self._result("SampleNTT[0:8]", str(sampled[:8]), GREEN)
        self._result("Range check", f"0 <= coeff <= 3328, max={max(sampled)}", GREEN)
        self._status_update("XOF/SampleNTT executed.")

    def _run_ntt(self):
        self._step("Component", "NTT / InvNTT")
        self._ensure_s_hat()
        self._ensure_keygen_sample_polys()
        poly = self._state["s_poly"]
        transformed = self._state["s_hat"]
        recovered = _inv_ntt(transformed)
        self._result("CBD secret poly s [0:6]", str(poly[:6]), CYAN)
        self._result("NTT [0:6]", str(transformed[:6]), GREEN)
        self._result("InvNTT [0:6]", str(recovered[:6]), GREEN)
        self._status_update("NTT executed.")

    def _run_poly_ops(self):
        self._step("Component", "Polynomial Add/Sub + Compression")
        self._ensure_keygen_sample_polys()
        p1 = self._state["s_poly"]
        p2 = self._state["e_poly"]
        p_add = _poly_add(p1, p2)
        p_sub = _poly_sub(p1, p2)

        du = self._engine.du
        dv = self._engine.dv

        # Use an actual sampled coefficient as compression input (for visualization).
        x = p_add[0]
        cu = _compress(du, x)
        cd_u = _decompress(du, cu)
        cv = _compress(dv, x)
        cd_v = _decompress(dv, cv)

        self._result("s_poly+e_poly [0:6]", str(p_add[:6]), GREEN)
        self._result("s_poly-e_poly [0:6]", str(p_sub[:6]), GREEN)
        self._result(f"Compress du={du}({x})", str(cu), CYAN)
        self._result("Decompress du result", str(cd_u), CYAN)
        self._result(f"Compress dv={dv}({x})", str(cv), CYAN)
        self._result("Decompress dv result", str(cd_v), CYAN)
        self._status_update("Polynomial operations executed.")

    def _run_kdf(self):
        self._step("Component", "KDF (G, H, J)")
        # This button is context-sensitive:
        # - If `ek` is not available yet, show KeyGen-side G(d||k) only.
        # - If `ek` and `m` are available, show Encapsulation-side k_bar/r = G(m||H(ek)).
        # - If ciphertext `c` exists, also show J(k_bar || H(c)) = K_enc.
        if self._state.get("ek") is None:
            self._ensure_g_keygen()
            rho = self._state["rho"]
            sigma = self._state["sigma"]
            self._result("G(d||k)->rho", rho.hex()[:32] + "...", GREEN)
            self._result("G(d||k)->sigma", sigma.hex()[:32] + "...", GREEN)
            self._result("Note", "J requires encapsulation; run Encapsulation next.", YELLOW)
            self._status_update("KDF executed for KeyGen (G only).")
            return

        ek = self._state["ek"]
        h_ek = _H(ek)
        k_bar, r = _G(self._state["m"] + h_ek)
        self._result("H(ek) preview", h_ek.hex()[:32] + "...", GREEN)
        self._result("G(m||H(ek))->k_bar", k_bar.hex()[:32] + "...", GREEN)
        self._result("G(m||H(ek))->r", r.hex()[:32] + "...", GREEN)

        if self._state.get("c") is not None:
            c = self._state["c"]
            h_c = _H(c)
            k_enc = _J(k_bar + h_c)
            self._result("H(c) preview", h_c.hex()[:32] + "...", GREEN)
            self._result("J(k_bar||H(c))=K_enc", k_enc.hex(), PURPLE)
            self._status_update("KDF executed for Encapsulation (G,H,J).")
        else:
            self._status_update("KDF executed for Encapsulation-side G/H (run Encapsulation for J).")

    def _run_keygen(self):
        self._step("Component", "Key Generation")
        self._ensure_rng_inputs()
        t0 = time.perf_counter()
        ek, dk = self._engine.keygen(self._state["d"], self._state["z"])
        elapsed = (time.perf_counter() - t0) * 1000
        self._state["ek"] = ek
        self._state["dk"] = dk
        self._result("ek size", f"{len(ek)} bytes (spec {self._engine.ek_size})", GREEN)
        self._result("dk size", f"{len(dk)} bytes (spec {self._engine.dk_size})", GREEN)
        self._result("elapsed", f"{elapsed:.2f} ms", CYAN)
        self._status_update("Key generation complete.")

    def _run_encapsulation(self):
        if self._state["ek"] is None:
            self._run_keygen()
        if self._state.get("m") is None:
            self._run_rng()
        self._step("Component", "Encapsulation")
        t0 = time.perf_counter()
        k_enc, c = self._engine.encaps(self._state["ek"], self._state["m"])
        elapsed = (time.perf_counter() - t0) * 1000
        self._state["k_enc"] = k_enc
        self._state["c"] = c
        self._result("ciphertext size", f"{len(c)} bytes (spec {self._engine.ct_size})", GREEN)
        self._result("shared secret K", k_enc.hex(), PURPLE)
        self._result("elapsed", f"{elapsed:.2f} ms", CYAN)
        self._status_update("Encapsulation complete.")

    def _run_decapsulation(self):
        if self._state["dk"] is None:
            self._run_keygen()
        if self._state["c"] is None:
            self._run_encapsulation()
        # dk/c are enough for decapsulation (z is embedded in dk).
        self._step("Component", "Decapsulation")
        t0 = time.perf_counter()
        k_dec = self._engine.decaps(self._state["c"], self._state["dk"])
        elapsed = (time.perf_counter() - t0) * 1000
        match = (self._state["k_enc"] == k_dec)
        self._state["k_dec"] = k_dec
        self._result("Recovered K", k_dec.hex(), PURPLE)
        self._result("match with encaps", "YES" if match else "NO", GREEN if match else RED)
        self._result("elapsed", f"{elapsed:.2f} ms", CYAN)
        self._status_update("Decapsulation complete.")

    def _run_end_to_end(self):
        self._step("Component", "End-to-End Workflow")
        # Start from current RNG-generated seeds and derive a couple of real intermediates.
        self._ensure_g_keygen()
        self._ensure_keygen_sample_polys()
        self._ensure_s_hat()

        rho = self._state["rho"]
        sigma = self._state["sigma"]
        s0 = self._state["s_poly"]

        self._result("G(d||k)->rho preview", rho.hex()[:32] + "...", GREEN)
        self._result("G(d||k)->sigma preview", sigma.hex()[:32] + "...", GREEN)
        self._result("s_poly (CBD) [0:8]", str(s0[:8]), GREEN)

        # ByteEncode/Decode roundtrip for 12-bit secret polynomial coefficients.
        enc = _byte_encode(12, self._state["s_hat"])
        dec = _byte_decode(12, enc)
        ok = dec[:10] == self._state["s_hat"][:10]
        self._result("ByteEncode_12(s_hat) roundtrip", "OK" if ok else "FAIL", CYAN)

        self._run_keygen()
        self._run_encapsulation()
        self._run_decapsulation()
        self._status_update(f"{self._variant.get()} component workflow executed.")

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _status_update(self, msg: str):
        self._status.config(text=msg)

    def _section(self, title: str):
        f = tk.Frame(self._scroll_frame, bg=BG, pady=6)
        f.pack(fill="x", padx=10)
        sep = tk.Frame(f, bg=BORDER, height=1)
        sep.pack(fill="x", pady=(4, 2))
        tk.Label(f, text=f"  {title}",
                 font=("Courier", 13, "bold"),
                 bg=BG, fg=YELLOW, anchor="w").pack(fill="x")
        sep2 = tk.Frame(f, bg=BORDER, height=1)
        sep2.pack(fill="x", pady=(2, 0))
        self.update_idletasks()
        self._canvas.yview_moveto(1.0)

    def _step(self, milestone: str, description: str, color: str = ACCENT):
        outer = tk.Frame(self._scroll_frame, bg=STEP_BG,
                         highlightbackground=BORDER, highlightthickness=1)
        outer.pack(fill="x", padx=10, pady=4)
        header = tk.Frame(outer, bg=color, padx=8, pady=3)
        header.pack(fill="x")
        tk.Label(header, text=f"  {milestone}",
                 font=("Courier", 11, "bold"),
                 bg=color, fg=BG).pack(side="left")
        tk.Label(header, text=description,
                 font=("Courier", 10),
                 bg=color, fg=BG).pack(side="left", padx=10)
        self._current_step_frame = outer
        self.update_idletasks()
        self._canvas.yview_moveto(1.0)

    def _result(self, label: str, value: str, color: str = TEXT):
        row = tk.Frame(self._current_step_frame, bg=STEP_BG, padx=12, pady=2)
        row.pack(fill="x")
        tk.Label(row, text=f"{label}:",
                 font=("Courier", 10),
                 bg=STEP_BG, fg=MUTED, width=36, anchor="w").pack(side="left")
        tk.Label(row, text=value,
                 font=("Courier", 10, "bold"),
                 bg=STEP_BG, fg=color, anchor="w").pack(side="left")
        self.update_idletasks()
        self._canvas.yview_moveto(1.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = MLKEMApp()
    app.mainloop()
