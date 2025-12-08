import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt

# ===================== 1) Anchor: main-cursor time (sub-sample refine) =====================
def find_main_time_from_pulse(pulse: np.ndarray, dt: float) -> float:
    i = int(np.argmax(np.abs(pulse)))
    if 1 <= i < len(pulse)-1:
        y1, y2, y3 = pulse[i-1], pulse[i], pulse[i+1]
        denom = (y1 - 2.0*y2 + y3)
        if denom != 0.0:
            delta = 0.5*(y1 - y3)/denom
            i = i + float(np.clip(delta, -0.5, 0.5))
    return float(i)*dt

# ===================== 2) Baud-spaced taps at absolute time =====================
def sample_cursors_at_reference(pulse: np.ndarray, dt: float, ui: float,
                                t_ref: float, n_pre: int, n_post: int) -> np.ndarray:
    ks  = np.arange(-n_pre, n_post+1)
    t   = t_ref + ks*ui
    idx = t/dt
    i0  = np.floor(idx).astype(int)
    a   = idx - i0
    i0  = np.clip(i0, 0, len(pulse)-2)
    return (1.0 - a)*pulse[i0] + a*pulse[i0+1]

def compare_eye_heights_at_center(pulse, ui, dt, n_pre=12, n_post=24, ber_target=1e-15):
    # reuse your existing helpers
    t_ref   = find_main_time_from_pulse(pulse, dt)
    cursors = sample_cursors_at_reference(pulse, dt, ui, t_ref, n_pre, n_post)
    c0      = float(cursors[n_pre])
    inter   = np.delete(cursors, n_pre)
    N_eff   = int(np.count_nonzero(np.abs(inter) > 0))

    # deterministic worst-case (lower bound)
    EH_wc = 2.0 * (abs(c0) - float(np.sum(np.abs(inter))))

    # statistical (Gaussian proxy)
    sigma_isi = float(np.sqrt(np.sum(inter**2)))
    
    # Q^{-1}(p) ~ sqrt(2) * erfcinv(2p). For p=1e-15 it's ~7.94; hard-code to avoid SciPy.
    Qinv = 7.94 if ber_target == 1e-15 else np.sqrt(2.0) * np.sqrt(-np.log(2.0*max(1e-300, ber_target)))
    
    EH_stat = 2.0 * (c0 - Qinv * sigma_isi)
    if EH_stat < 0.0:
        EH_stat = 0.0

    P_wc = 2.0 ** (-N_eff)

    print(f"c0(center)     : {c0:+.3f}")
    # np.set_printoptions(precision=3)
    # print(f"cursors        : {cursors}")
    print(f"Σ|ck|(no main) : {np.sum(np.abs(inter)):.3f}")
    print(f"σ_ISI          : {sigma_isi:.3f}")
    print(f"N_eff taps     : {N_eff:d}  →  P(worst-case pattern) ≈ 2^(-N) = {P_wc:.3e}")
    print(f"EH_worst-case  : {EH_wc:.3f}  (lower bound)")
    print(f"EH_stat @BER={ber_target:.1e} ≈ {EH_stat:.3f}  (Gaussian approx: much worse the ISI tail)\n")

# ===================== 3) Voltage-grid sizing =====================
def _choose_bins_span_alltaus(all_inters_abs_min: float,
                              all_AMAX_max: float,
                              min_bins: int = 1000,
                              dv_max_fraction: float = 0.1) -> Tuple[int, float]:
    v_span = max(1.2, 1.05 * float(all_AMAX_max))
    if all_inters_abs_min > 0:
        dv_max = float(all_inters_abs_min) * dv_max_fraction
        v_bins_by_dv = int(np.ceil((2.0 * v_span) / max(dv_max, 1e-15)))
        v_bins = max(min_bins, v_bins_by_dv)
    else:
        v_bins = max(min_bins, 1001)
    if v_bins % 2 == 0:
        v_bins += 1
    return v_bins, v_span

def recommend_vspan_include_main(pulse, ui, dt, n_pre=1, n_post=1, n_tau=61):
    t_ref  = find_main_time_from_pulse(pulse, dt)
    taus   = np.linspace(0.0, 1.0, n_tau, endpoint=False)
    amax_max = 0.0
    c0_max   = 0.0
    for u in taus:
        t0 = t_ref + (u - 0.5)*ui   # if you already shifted to center at 0.5 UI; else drop (-0.5)
        curs = sample_cursors_at_reference(pulse, dt, ui, t0, n_pre, n_post)
        c0   = float(curs[n_pre])
        inter= np.delete(curs, n_pre)
        amax_max = max(amax_max, float(np.sum(np.abs(inter))))
        c0_max   = max(c0_max, abs(c0))
    return 1.05*(amax_max + c0_max)

# ===================== 4) ISI PDFs via CF (characteristic function) =====================
def _shift_pdf(pdf: np.ndarray, support: np.ndarray, delta: float) -> np.ndarray:
    dv = float(support[1] - support[0])
    sb = delta / dv
    k  = int(np.floor(sb))
    f  = sb - k
    r0 = np.roll(pdf, k)
    r1 = np.roll(pdf, k+1)
    out = (1.0 - f)*r0 + f*r1
    s = out.sum()
    return out/(s + 1e-20)

def _convolve_plusminus(pdf: np.ndarray, support: np.ndarray, c: float) -> np.ndarray:
    if abs(c) < 1e-18:
        return pdf
    out = 0.5*_shift_pdf(pdf, support, +c) + 0.5*_shift_pdf(pdf, support, -c)
    return out/(out.sum() + 1e-20)

# ---------- ISI PDFs via CF (characteristic function) ----------
def _isi_pdf_cf_from_taps(interferers: np.ndarray, V: np.ndarray, omega: np.ndarray) -> np.ndarray:
    if interferers.size == 0:
        pdf = np.zeros_like(V, dtype=np.float32)
        pdf[np.argmin(np.abs(V))] = 1.0
        return pdf
    c = interferers.astype(np.float64, copy=False)
    Phi = np.ones_like(omega, dtype=np.float64)
    for ck in c:
        Phi *= np.cos(omega * ck)
    Phi_shift = np.fft.ifftshift(Phi)
    pdf = np.fft.ifft(Phi_shift).real
    pdf = np.fft.fftshift(pdf)
    pdf = np.maximum(pdf, 0.0)
    s = pdf.sum()
    if s > 0:
        pdf /= s
    return pdf.astype(np.float32, copy=False)

# =====================================================================
# A) Iterative builder (±c convolution) — original, readable reference
# =====================================================================
def build_isi_pdf_grid_from_pulse_anchored(
    pulse: np.ndarray, ui: float, dt: float,
    n_precursor: int = 12, n_postcursor: int = 24,
    n_tau: int = 81,
    v_bins: Optional[int] = None, v_span: Optional[float] = None,
    prune_energy: float = 0.995,            # keep taps covering ≥99.5% of Σ c^2
    prune_abs_floor: Optional[float] = None,# drop |c| below this (set to ΔV/4 if None)
    dtype=np.float32
):
    """
    Anchored multi-τ ISI PDFs using iterative ±c convolution (Sanders Loop #1+#2).
    Returns: tau_ui, v_support, isi_pdfs (float32), main_cursors (float32)
    """
    t_ref  = find_main_time_from_pulse(pulse, dt)
    tau_ui = np.linspace(0.0, 1.0, int(n_tau), endpoint=False)

    # Pass 1: collect taps and grid stats
    n_taps = n_precursor + 1 + n_postcursor
    all_cursors = np.empty((n_tau, n_taps), dtype=np.float32)
    all_AMAX_max = 0.0
    inter_abs_min = np.inf
    for i, u in enumerate(tau_ui):
        # t0   = t_ref + u*ui
        t0 = t_ref + (u - 0.5) * ui
        curs = sample_cursors_at_reference(pulse, dt, ui, t0, n_precursor, n_postcursor).astype(np.float32)
        all_cursors[i, :] = curs
        inter = np.delete(curs, n_precursor)
        AMAX  = float(np.sum(np.abs(inter)))
        all_AMAX_max = max(all_AMAX_max, AMAX)
        nz = np.abs(inter); nz = nz[nz > 0]
        if nz.size > 0:
            inter_abs_min = min(inter_abs_min, float(nz.min()))

    # Common voltage grid
    if (v_bins is None) or (v_span is None):
        v_bins_auto, v_span_auto = _choose_bins_span_alltaus(
            all_inters_abs_min = 0.0 if not np.isfinite(inter_abs_min) else inter_abs_min,
            all_AMAX_max = all_AMAX_max, min_bins=1000, dv_max_fraction=0.1
        )
        v_bins = v_bins if v_bins is not None else v_bins_auto
        v_span = v_span if (v_span is not None and v_span > 0) else v_span_auto

    V  = np.linspace(-v_span, v_span, int(v_bins), dtype=dtype)
    dv = float(V[1] - V[0])
    if prune_abs_floor is None:
        prune_abs_floor = 0.25 * dv

    # Build PDFs (row-by-row, low memory)
    isi_pdfs     = np.empty((n_tau, len(V)), dtype=dtype)
    main_cursors = np.empty(n_tau, dtype=dtype)
    delta0 = np.zeros_like(V); delta0[np.argmin(np.abs(V))] = 1.0

    for i in range(n_tau):
        curs = all_cursors[i, :].astype(np.float64, copy=False)
        main = float(curs[n_precursor]); main_cursors[i] = np.float32(main)
        inter = np.delete(curs, n_precursor)

        # prune by energy
        abs2 = inter*inter
        if abs2.size:
            order = np.argsort(abs2)[::-1]
            cum = np.cumsum(abs2[order])
            keep_n = int(np.searchsorted(cum, prune_energy * cum[-1], side='right') + 1)
            inter_kept = inter[order[:keep_n]]
        else:
            inter_kept = inter
        # prune by absolute floor
        inter_kept = inter_kept[np.abs(inter_kept) >= prune_abs_floor]

        pdf = delta0.copy()
        for c in inter_kept:
            pdf = _convolve_plusminus(pdf, V, float(c))
        isi_pdfs[i, :] = pdf.astype(dtype, copy=False)

    return tau_ui.astype(dtype), V, isi_pdfs, main_cursors

# ===================== 6) Dual-Dirac τ-averaging (Loop #3) =====================

def _norm_pdf_periodic_centered(n_tau: int, sigma_ui: float, mean_ui: float) -> np.ndarray:
    if sigma_ui <= 0:
        ker = np.zeros(n_tau, dtype=np.float64); ker[0] = 1.0
        return ker
    nu = (np.arange(n_tau) - n_tau//2) / float(n_tau)  # [-0.5,0.5)
    ker = np.exp(-0.5*((nu - mean_ui)/sigma_ui)**2) / (np.sqrt(2*np.pi)*sigma_ui)
    ker /= ker.sum() + 1e-20
    return np.roll(ker, n_tau//2)

def dual_dirac_tau_kernel(n_tau: int, sigma_rj_ui: float, dj_pp_ui: float) -> np.ndarray:
    """
    Dual-Dirac kernel on τ grid: K(ν) = 0.5*N(ν;+w,σ) + 0.5*N(ν;−w,σ), w=DJpp/2 (in UI)
    """
    w = 0.5*float(dj_pp_ui)
    kpos = _norm_pdf_periodic_centered(n_tau, sigma_rj_ui, +w)
    kneg = _norm_pdf_periodic_centered(n_tau, sigma_rj_ui, -w)
    ker = 0.5*(kpos + kneg)
    ker /= ker.sum() + 1e-20
    return ker.astype(np.float64)

def apply_tau_kernel_dual_dirac(isi_pdfs: np.ndarray, sigma_rj_ui: float, dj_pp_ui: float) -> np.ndarray:
    """
    Circular convolution along τ with Dual-Dirac kernel (vectorized via FFT).
    isi_pdfs: shape (n_tau, v_bins)  →  returns same shape (float32)
    """
    n_tau = isi_pdfs.shape[0]
    ker = dual_dirac_tau_kernel(n_tau, sigma_rj_ui, dj_pp_ui)[:, None]   # (n_tau,1)
    Fp  = np.fft.rfft(isi_pdfs.astype(np.float64, copy=False), axis=0)
    Fk  = np.fft.rfft(ker, n=n_tau, axis=0)
    out = np.fft.irfft(Fp*Fk, n=n_tau, axis=0)
    out[out < 0] = 0.0
    out /= (out.sum(axis=1, keepdims=True) + 1e-20)
    return out.astype(np.float32, copy=False)

# ===================== 7) BER (streamed, low memory) =====================

def _shift_pdf_row(base: np.ndarray, V: np.ndarray, delta: float, out: np.ndarray):
    """Shift 1 row by delta into 'out' using linear interp on same grid (in-place)."""
    dv = float(V[1] - V[0])
    sb = delta / dv
    k  = int(np.floor(sb))
    f  = sb - k
    np.copyto(out, (1.0 - f) * np.roll(base, k) + f * np.roll(base, k+1))

def _gaussian_kernel_v_fullgrid(V: np.ndarray, sigma: float):
    if sigma <= 0:
        g = np.zeros_like(V); g[np.argmin(np.abs(V))] = 1.0
        return g.astype(np.float32)
    dv = float(V[1] - V[0])
    half = int(np.ceil(6.0*sigma/dv))
    kx = np.arange(-half, half+1) * dv
    g  = np.exp(-0.5*(kx/sigma)**2) / (np.sqrt(2*np.pi)*sigma)
    g /= (g.sum()*dv)  # area=1 under discrete integration
    G = np.zeros_like(V, dtype=np.float64)
    c = len(V)//2
    for i,val in enumerate(g):
        G[(c - half + i) % len(V)] = val
    return G.astype(np.float32)

def build_ber_grid_from_isi_streaming(
    tau_ui: np.ndarray,
    v_support: np.ndarray,
    isi_pdfs: np.ndarray,       # (n_tau, v_bins) float32 recommended
    main_cursors: np.ndarray,   # (n_tau,)
    sigma_v: float = 0.0,
    chunk_tau: int = 16,
    dtype=np.float32
):
    """
    Streamed BER(τ,V) computation:
      - bit-conditional shift by ±c0(τ)
      - optional vertical Gaussian noise (FFT per row)
      - tails → BER
    """
    n_tau, v_bins = int(isi_pdfs.shape[0]), int(isi_pdfs.shape[1])
    V  = v_support.astype(dtype, copy=False)
    dv = float(V[1] - V[0])

    # optional vertical-noise kernel (FFT once)
    if sigma_v > 0:
        ker = _gaussian_kernel_v_fullgrid(V.astype(np.float64), sigma_v).astype(dtype)
        Fk  = np.fft.rfft(ker)

    BER = np.empty((n_tau, v_bins), dtype=dtype)
    plus  = np.empty(v_bins, dtype=dtype)
    minus = np.empty(v_bins, dtype=dtype)

    for i0 in range(0, n_tau, int(chunk_tau)):
        i1 = min(i0 + int(chunk_tau), n_tau)
        for i in range(i0, i1):
            base = isi_pdfs[i, :].astype(dtype, copy=False)
            m    = float(main_cursors[i])

            _shift_pdf_row(base, V, +m, plus)
            _shift_pdf_row(base, V, -m, minus)

            if sigma_v > 0:
                Fp = np.fft.rfft(plus);  plus[:]  = np.fft.irfft(Fp * Fk, n=v_bins).astype(dtype, copy=False)
                Fm = np.fft.rfft(minus); minus[:] = np.fft.irfft(Fm * Fk, n=v_bins).astype(dtype, copy=False)
                plus[plus < 0] = 0; minus[minus < 0] = 0
                s1 = plus.sum();  s2 = minus.sum()
                if s1 > 0: plus  /= s1
                if s2 > 0: minus /= s2

            F_plus     = np.cumsum(plus,  dtype=np.float64) * dv
            tail_minus = np.cumsum(minus[::-1].astype(np.float64))[::-1] * dv
            row = 0.5*F_plus + 0.5*tail_minus
            np.copyto(BER[i, :], np.clip(row, 0.0, 1.0).astype(dtype, copy=False))

    t_ui = tau_ui.astype(dtype, copy=False)
    return t_ui, V, BER

# ===================== 8) Plotting helpers =====================

def _sanitize_levels(levels, Z):
    """
    Prepare contour levels for BER surface Z.

    - If any requested levels are inside (min(Z), max(Z)), keep those
      (strictly increasing).
    - If *none* are inside, fall back to 4 levels:
        L0 = smallest drawable level ≈ nextafter(min(Z), +inf)
        L1 = min(L0*1e3, max(Z)-ε)
        L2 = min(L1*1e3, max(Z)-ε)
        L3 = min(L2*1e3, max(Z)-ε)
      This guarantees you see the eye at the lowest achievable BER plus
      3 additional contours at “1e-3 resolution”.
    """
    Z = np.asarray(Z, float)
    finite = Z[np.isfinite(Z)]
    zmin, zmax = float(finite.min()), float(finite.max())
    eps = max(np.finfo(float).eps, 1e-16) * max(1.0, abs(zmin), abs(zmax))

    # 1) Use any requested levels that are inside the data range.
    req = sorted({float(lv) for lv in np.ravel(levels) if np.isfinite(lv)})
    kept = [lv for lv in req if (zmin + eps) < lv < (zmax - eps)]
    if len(kept) >= 1:
        out, last = [], -np.inf
        for lv in kept:
            if lv <= last:
                lv = np.nextafter(last, np.inf)
            out.append(lv); last = lv
        return out

    # 2) Fallback: lowest achievable level + 3 more at ×1e3 (1e-3 decade spacing).
    L0 = np.nextafter(zmin, np.inf)              # smallest drawable iso-BER
    out = [L0]
    step = 1e1                                  # each next contour is 3 decades higher
    for _ in range(3):
        cand = out[-1] * step
        if cand >= (zmax - eps):
            cand = np.nextafter(zmax - eps, -np.inf)
        if cand <= out[-1]:
            cand = np.nextafter(out[-1], np.inf)
        out.append(cand)
    return out

def plot_isi_bar_at_tau_fast(taus_ui, v_support, isi_pdfs, tau_index: int):
    """Bar-plot a single τ slice of ISI PDF (no BER computations)."""
    V   = v_support
    pdf = isi_pdfs[tau_index, :]
    dv  = float(V[1] - V[0])
    tau_ui = float(taus_ui[tau_index])

    plt.figure(figsize=(6,4))
    plt.bar(V, pdf, width=dv, align='center', edgecolor='k', linewidth=0.25)
    plt.title(f"ISI PDF(fast) at τ = {tau_ui:.3f} UI")
    plt.xlabel("Voltage [arb]"); plt.ylabel("Probability density")
    plt.grid(True, axis='y', linestyle=':')
    plt.tight_layout(); plt.show()

def plot_eye_center_isi_bar_from_grid(taus_ui, v_support, isi_pdfs):
    """Bar-plot the ISI PDF at the τ closest to 0.5 UI from an existing grid."""
    idx = int(np.argmin(np.abs(taus_ui - 0.5)))
    V, pdf = v_support, isi_pdfs[idx, :]
    dv = float(V[1] - V[0])
    tau_ui = float(taus_ui[idx])

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.bar(V, pdf, width=dv, align='center', edgecolor='k', linewidth=0.25)
    plt.title(f"ISI PDF at Eye Center (τ ≈ {tau_ui:.3f} UI)")
    plt.xlabel("Voltage [arb]"); plt.ylabel("Probability density")
    plt.grid(True, axis='y', linestyle=':')
    plt.tight_layout(); plt.show()

def plot_ber_contours_ui01(
    t_ui, V, BER,
    levels=(1e-7,1e-8,1e-9,1e-10,1e-12,1e-15),
    title="StatEye BER contours (center at 0.5 UI)",
    main_cursors=None,
    print_table=True,
    palette="rainbow",          # base colormap to sample
    reverse=False,              # reverse the palette
    use_filled=False,           # filled vs line contours
    show_legend=True            # legend for line contours
):
    """
    Plots BER(t,V) contours with τ in [0,1) and prints Eye Height & Eye Width per BER level.

    Eye Height @ level L:  max over τ of the vertical opening where BER(τ,V) ≤ L.
    Eye Width  @ level L:  max over threshold V of the (circular) time opening where BER(τ,V) ≤ L.
    """

    # ---------- helpers ----------
    def _max_true_run(mask):
        """Longest contiguous True run in 1D boolean mask (non-circular)."""
        n = len(mask); best_len = 0; best_s = -1; best_e = -1; i = 0
        while i < n:
            if mask[i]:
                j = i
                while j < n and mask[j]:
                    j += 1
                if (j - i) > best_len:
                    best_len = j - i; best_s = i; best_e = j - 1
                i = j
            else:
                i += 1
        return best_len, best_s, best_e

    def _max_true_run_circular(mask):
        """Longest contiguous True run treating axis as circular (cap length at n)."""
        n = len(mask)
        if n == 0:
            return 0, -1, -1
        mm = np.concatenate([mask, mask])
        N = mm.size
        best_len = 0; best_s = -1; best_e = -1; i = 0
        while i < N:
            if mm[i]:
                j = i
                while j < N and mm[j]:
                    j += 1
                length = min(j - i, n)
                if length > best_len:
                    best_len = length; best_s = i; best_e = i + length - 1
                i = j
            else:
                i += 1
        if best_s < 0:
            return 0, -1, -1
        return best_len, best_s % n, best_e % n

    # ---------- plot contours ----------
    levels_in = _sanitize_levels(levels, BER)
    # --- build a discrete color list (one color per level) ---
    cmap = plt.get_cmap(palette)
    if reverse:
        cmap = cmap.reversed()

    nlev = len(levels_in)
    if use_filled:
        # contourf has nlev-1 filled bands between levels
        colors_fill = [cmap(x) for x in np.linspace(0, 1, max(nlev-1, 1))]
        fig, ax = plt.subplots(figsize=(5.8, 4.4))
        CSf = ax.contourf(t_ui, V, BER.T, levels=levels_in, colors=colors_fill)
        # thin black isolines on top (optional)
        CSl = ax.contour(t_ui, V, BER.T, levels=levels_in, colors='k', linewidths=0.5)
        cbar = fig.colorbar(CSf, ax=ax, label="BER")
    else:
        # contour lines: exactly one color per level
        colors_line = [cmap(x) for x in np.linspace(0, 1, nlev)]
        fig, ax = plt.subplots(figsize=(5.8, 4.4))
        CSl = ax.contour(t_ui, V, BER.T, levels=levels_in, colors=colors_line, linewidths=1.5)
        # label the contours; color labels to match lines (Matplotlib cycles colors)
        ax.clabel(CSl, inline=True, fmt="%1.0e", colors=colors_line)
        # optional legend mapping each color to its BER level
        if show_legend:
            from matplotlib.lines import Line2D
            handles = [Line2D([0],[0], color=colors_line[i], lw=2, label=f"{levels_in[i]:.1e}")
                       for i in range(nlev)]
            ax.legend(handles=handles, title="BER level", frameon=True, fontsize=8, loc="upper right")

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Time [UI]"); ax.set_ylabel("Voltage [arb]")
    ax.set_title(title); ax.grid(True, ls=":")
    ax.axvline(0.5, ls="--", alpha=0.3)
    plt.tight_layout(); plt.show()

    # ---------- main cursor summary (if available) ----------
    if main_cursors is not None and len(main_cursors) == len(t_ui):
        i_center = int(np.argmin(np.abs(t_ui - 0.5)))
        print(f"Main cursor @0.5 UI: {float(main_cursors[i_center]):+.6f} "
              f"(min={float(np.min(main_cursors)):+.6f}, max={float(np.max(main_cursors)):+.6f})")

    if not print_table:
        return

    # ---------- eye metrics table ----------
    dv = float(V[1] - V[0]) if len(V) > 1 else 0.0
    dt = float(t_ui[1] - t_ui[0]) if len(t_ui) > 1 else 0.0
    n_tau, n_V = BER.shape[0], BER.shape[1]

    rows = []
    for L in levels_in:
        # Eye Height: scan times, take max vertical opening
        best_h = 0.0; best_i = -1; sV = -1; eV = -1
        for i in range(n_tau):
            mask_v = BER[i, :] <= L
            run_len, s, e = _max_true_run(mask_v)
            if run_len > 0:
                # Height as (#bins in run) * ΔV
                h = run_len * dv
                if h > best_h:
                    best_h = h; best_i = i; sV = s; eV = e

        # Eye Width: scan thresholds, take max circular time opening
        best_w = 0.0; best_j = -1; sT = -1; eT = -1
        for j in range(n_V):
            mask_t = BER[:, j] <= L
            run_len, s, e = _max_true_run_circular(mask_t)
            if run_len > 0:
                # Width as (#bins in run) * Δt (cap at 1 UI)
                w = min(run_len, n_tau) * dt
                if w > best_w:
                    best_w = w; best_j = j; sT = s; eT = e

        row = {
            "BER": L,
            "EH[V]": best_h,
            "EH_at_tau[UI]": (float(t_ui[best_i]) if best_i >= 0 else np.nan),
            "EH_V_low": (float(V[sV]) if sV >= 0 else np.nan),
            "EH_V_high": (float(V[eV]) if eV >= 0 else np.nan),
            "EW[UI]": best_w,
            "EW_at_Vth": (float(V[best_j]) if best_j >= 0 else np.nan),
        }
        rows.append(row)

    # Pretty-print table
    header = f"{'BER':>10} | {'EH[V]':>10} | {'EH@τ[UI]':>10} | {'V_low':>10} | {'V_high':>10} | {'EW[UI]':>10} | {'EW@Vth':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['BER']:>10.1e} | {r['EH[V]']:>10.6f} | {r['EH_at_tau[UI]']:>10.6f} | "
              f"{r['EH_V_low']:>10.6f} | {r['EH_V_high']:>10.6f} | {r['EW[UI]']:>10.6f} | {r['EW_at_Vth']:>10.6f}")


# ===================== Usage (quick start) =====================
# 1) You already have: pulse (pulse response), ui = 1/data_rate, dt (time step).
# 2) Build ISI PDFs (anchored, CF) on a common V-grid:
#    taus_ui, V, isi_pdfs, main = build_isi_pdf_grid_from_pulse_anchored_cf(
#        pulse, ui, dt, n_precursor=12, n_postcursor=24, n_tau=61, v_bins=801,
#        prune_energy=0.995, chunk_tau=32, dtype=np.float32
#    )
# 3) (Optional) τ-average with Dual-Dirac jitter:
#    isi_avg = apply_tau_kernel_dual_dirac(isi_pdfs, sigma_rj_ui=0.004, dj_pp_ui=0.10)
#    # Use isi_avg instead of isi_pdfs below if you want jitter-averaged contours.
# 4) Build BER with low memory:
#    t_ui, Vg, BER = build_ber_grid_from_isi_streaming(
#        taus_ui, V, isi_pdfs, main, sigma_v=0.0, chunk_tau=16, dtype=np.float32
#    )
# 5) Plot:
#    plot_ber_contours(t_ui, Vg, BER, levels=(1e-7,1e-8,1e-9,1e-10,1e-12,1e-15))
#    # Or inspect one τ slice as a bar chart:
#    plot_isi_bar_at_tau_fast(taus_ui, V, isi_pdfs, tau_index=int(0.5*len(taus_ui)))