import numpy as np

# ---------- PAM4 LUT-FFE support ----------

def gray_to_pam4(g1: int, g0: int) -> int:
    """Gray (g1,g0) -> PAM4 amplitude {-3,-1,+1,+3}."""
    if g1 == 0 and g0 == 0:
        return -3
    if g1 == 0 and g0 == 1:
        return -1
    if g1 == 1 and g0 == 1:
        return +1
    if g1 == 1 and g0 == 0:
        return +3
    raise ValueError("Invalid Gray code")


def build_pam4_2tap_lut(h0: float, h1: float):
    """
    For tap pair (h0,h1), build 16-entry LUT:
      addr[3:2] = Gray of symbol a0
      addr[1:0] = Gray of symbol a1
      LUT[addr] = h0*a0 + h1*a1
    """
    lut = np.zeros(16, dtype=float)
    for addr in range(16):
        g1a = (addr >> 3) & 1
        g0a = (addr >> 2) & 1
        g1b = (addr >> 1) & 1
        g0b = addr & 1
        a0 = gray_to_pam4(g1a, g0a)
        a1 = gray_to_pam4(g1b, g0b)
        lut[addr] = h0 * a0 + h1 * a1
    return lut


class Pam4LutFfe2Tap:
    """
    8-tap PAM4 TX FFE:
        y[n] = sum_{k=0..7} h_k * a[n-k]
    implemented as 4x 2-tap LUTs (DA style).
    """

    def __init__(self, coeffs, y_max=7.0):
        coeffs = np.asarray(coeffs, dtype=float)
        assert coeffs.size == 8, "Need exactly 8 FFE taps"
        self.coeffs = coeffs
        self.y_max = float(y_max)

        # Build 4 LUTs for tap pairs (h0,h1), (h2,h3), ...
        self.luts = []
        for i in range(4):
            h0 = coeffs[2*i]
            h1 = coeffs[2*i+1]
            self.luts.append(build_pam4_2tap_lut(h0, h1))

        # 8-symbol Gray history; index 0 is current symbol
        self.hist_g1 = [0]*8
        self.hist_g0 = [0]*8

    def step_analog(self, g1: int, g0: int) -> float:
        """One symbol step: update history, use LUTs, return analog y[n]."""
        # shift in new Gray symbol
        self.hist_g1 = [g1] + self.hist_g1[:-1]
        self.hist_g0 = [g0] + self.hist_g0[:-1]

        y = 0.0
        for bank in range(4):
            idx0 = 2*bank
            idx1 = 2*bank + 1

            g1a = self.hist_g1[idx0]
            g0a = self.hist_g0[idx0]
            g1b = self.hist_g1[idx1]
            g0b = self.hist_g0[idx1]

            addr = (g1a << 3) | (g0a << 2) | (g1b << 1) | g0b
            y += self.luts[bank][addr]

        return y

def compute_tx_ffe_zero_forcing(
        h_row,
        num_taps=8,
        main_index=None,
        lam=1e-4,
):
    """
    Given:
        h_row    : 1-D array, first row of Toeplitz channel matrix
                   (symbol-spaced channel impulse, including pre/post)
        num_taps : number of TX FFE taps (length of f)
        main_idx : index in h_row where you want the *equalized* main cursor
                   (default = argmax |h|)
        lam      : small Tikhonov regularization for numerical stability

    Solves (regularized LS):
        H_full f ≈ d,   with d[k_main] = 1, others 0

    where H_full is the convolution matrix of the channel.

    Returns:
        f   : TX FFE taps (length = num_taps)
        g   : effective channel impulse g = h * f
    """
    h = np.array(h_row, dtype=float)
    Lh = len(h)

    if main_index is None:
        main_index = int(np.argmax(np.abs(h)))  # cursor with max magnitude

    # Build full convolution matrix (Toeplitz) for g = H_full f
    K = Lh + num_taps - 1                 # length of g
    H_full = np.zeros((K, num_taps))
    for r in range(K):
        for c in range(num_taps):
            idx = r - c
            if 0 <= idx < Lh:
                H_full[r, c] = h[idx]

    # Target: delta at main_index -> d[k_main] = 1, others 0
    d = np.zeros(K)
    d[main_index] = 1.0

    # Regularized LS solution: (H^T H + λI) f = H^T d
    HtH = H_full.T @ H_full
    rhs = H_full.T @ d
    f = np.linalg.solve(HtH + lam * np.eye(num_taps), rhs)

    # Normalize so that g[main_index] is exactly 1
    g = np.convolve(f, h)
    scale = 1.0 / g[main_index]
    f *= scale
    g *= scale

    return f, g
