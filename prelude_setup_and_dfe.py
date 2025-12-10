"""
Prelude file for full_link series scripts
Contains common setup and DFE processing that can be reused across multiple scripts.

Usage:
    Place this file in the same directory as your main scripts.
    In your main script, include:
        exec(open('prelude_setup_and_dfe.py').read())
    
    This will:
    1. Initialize global variables (g, Ts, t, pulse_response_length, etc.)
    2. Generate PRBS signal and apply jitter if needed
    3. Load channel transfer function and impulse response
    4. Generate pulse response and frequency plots
    5. Take signal_jitter as input and produce signal_filtered as output
"""

# ============================================================================
# PART 1: COMMON SETUP (Original lines 48-166)
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.interpolate import PchipInterpolator
import skrf as rf
import warnings
from pathlib import Path

import serdespy as sdp
import sparam_modeling as sm
from sparam_modeling import gen_channel, frd_imp, cconv, impinterp, get_crossings
from plot_functions import plot_pulse_response, analyze_and_plot_cursors

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Initialize global dictionary (if not already defined in main script)
# ============================================================================
if 'g' not in globals():
    g = {
        'pulse_signal': None,
        'f': None,
        'pulse_resp_ch': None,
        'ratio_oversampling': None,
        'ui': None,
        'os': None,
        'H_ch': None,
        'tx_launch_amp': None,
        'pulse_signal_length': None,
        'num_pre_cursor': None,
        'num_post_cursor': None,
        'rterm_source': None,
        'rterm_sink': None,
    }

# ============================================================================
# Initialize parameters (if not already set)
# ============================================================================
if 'data_rate' not in globals():
    data_rate = 56e9  # NRZ
if 'PLOT_FREQ_RESP' not in globals():
    PLOT_FREQ_RESP = True
if 'PLOT_PULSE_RESP' not in globals():
    PLOT_PULSE_RESP = True
if 'ADD_RAND_JITTER' not in globals():
    ADD_RAND_JITTER = False

f_nyq = data_rate / 2
g['ui'] = 1 / data_rate
g['os'] = 128  # samples per symbol
g['tx_launch_amp'] = 0.6
g['num_pre_cursor'] = 1
g['num_post_cursor'] = 4
g['rterm_source'] = 50
g['rterm_sink'] = 50
print("Variable Initialized.\n")

# ============================================================================
# Input Pulse Generation
# ============================================================================
Ts = g['ui'] / g['os']  # Time step
pulse_response_length = 100
total_data_width = pulse_response_length * g['ui']
pulse_start = 3 * g['ui']

# Create time vector with precise number of samples
num_samples = int(total_data_width / Ts) + 1
t = np.linspace(0, total_data_width, num_samples)

g['pulse_signal'] = np.zeros_like(t)
start_index = int(pulse_start / Ts)
end_index = int(start_index + (1 * g['ui']) / Ts)
g['pulse_signal'][start_index:end_index] = g['tx_launch_amp']
g['pulse_signal_length'] = int(total_data_width / Ts)
print("Input pulse is defined.\n")

# ============================================================================
# Generate binary data and apply jitter
# ============================================================================
data = sdp.prbs13(1)
signal_BR = sdp.nrz_input_BR(data)  # generate Baud-Rate sampled signal from data
signal_ideal = 0.5 * g['tx_launch_amp'] * np.repeat(signal_BR, g['os'])  # oversample to get Tx signal
print("PRBS signal train is generated.\n")

if ADD_RAND_JITTER:
    signal_jitter = sdp.gaussian_jitter(signal_ideal, g['ui'], len(data), g['os'], stdev=1000e-15)
    print("Random Jitter is added.\n")
else:
    signal_jitter = signal_ideal

# ============================================================================
# TX to RX link - Load S-parameters and generate channel
# ============================================================================
if '__file__' in globals():
    script_dir = Path(__file__).resolve().parent
else:
    # When using exec(), __file__ might not be defined, use current working directory
    script_dir = Path.cwd()
    
s_param_dir = script_dir / "Channels"
if not s_param_dir.is_dir():
    print(f"ERROR: S-parameter directory not found at '{s_param_dir}'")
    print("Please ensure the 'Channels' directory exists in the same directory as this script.")
    raise FileNotFoundError(f"S-parameter directory not found at '{s_param_dir}'")

# Baseline Channel without t-coils
H_base, f_base, S11_s_base, S11_l_base = gen_channel(
    # Source
    r_s=g['rterm_source'],
    c_die_s=150e-15,
    L1_s=100e-12,
    c_esd1_s=200e-15,
    L2_s=300e-12,
    c_esd2_s=200e-15,
    L3_s=300e-12,
    c_pad_s=100e-15,
    km_s=-0.5,
    # Sink
    c_pad_l=100e-15,
    L1_l=50e-12,
    c_esd_l=400e-15,
    L2_l=200e-12,
    c_die_l=150e-15,
    km_l=-0.4,
    r_l=g['rterm_sink'],
    pkg_s=s_param_dir / 'PKG100GEL_95ohm_30mm_50ohmPort.s4p',  # Source Package
    pkg_l=s_param_dir / 'PKG100GEL_95ohm_30mm_50ohmPort.s4p',  # Sink Package
    s_tcoil=False,
    s_tcoil_split=True,
    l_tcoil=False,
    l_tcoil_split=True,
    pkg_s_portswap=True,
    pkg_l_portswap=True,
    ch_portswap=False
)

# Full Link Channel with t-coils
g['H_ch'], g['f'], S11_s, S11_l = gen_channel(
    # Source
    r_s=g['rterm_source'],
    c_die_s=150e-15,
    L1_s=250e-12,
    c_esd1_s=200e-15,
    L2_s=100e-12,
    c_esd2_s=200e-15,
    L3_s=150e-12,
    c_pad_s=100e-15,
    km_s=-0.4,
    # Sink
    c_pad_l=100e-15,
    L1_l=50e-12,
    c_esd_l=400e-15,
    L2_l=200e-12,
    c_die_l=150e-15,
    km_l=-0.4,
    r_l=g['rterm_sink'],
    pkg_s=s_param_dir / 'PKG100GEL_95ohm_30mm_50ohmPort.s4p',  # Source Package
    pkg_l=s_param_dir / 'PKG100GEL_95ohm_30mm_50ohmPort.s4p',  # Sink Package
    s_tcoil=True,
    s_tcoil_split=True,
    l_tcoil=False,
    l_tcoil_split=True,
    pkg_s_portswap=True,
    pkg_l_portswap=True,
    ch_portswap=False
)
if g['H_ch'] is None:
    raise RuntimeError("Failed to load channel transfer function")
print("Full link: Transfer Function evaluation completed.\n")

# ============================================================================
# Impulse Response
# ============================================================================
imp_ch_raw, Fs_ntwk = frd_imp(g['H_ch'], g['f'] * 2 * np.pi)
Fs = 1 / Ts
g['ratio_oversampling'] = round(Fs / (2 * Fs_ntwk))
print("Full link: Impulse Response evaluation completed.\n")

# ============================================================================
# Output - Frequency Response Plot
# ============================================================================
fi_nyq = np.argmin(np.abs(g['f'] - f_nyq))
H_ch_loss_at_nyquist = 20 * np.log10(np.abs(g['H_ch'][fi_nyq]))
fi_rate = np.argmin(np.abs(g['f'] - data_rate))
H_ch_loss_at_rate = 20 * np.log10(np.abs(g['H_ch'][fi_rate]))
print(f"Loss at {f_nyq/1e9:.1f} GHz for end to end full link: {H_ch_loss_at_nyquist:.2f}dB")
print(f"Loss at {data_rate/1e9:.1f} GHz for end to end full link: {H_ch_loss_at_rate:.2f}dB\n")
if PLOT_FREQ_RESP:
    plt.figure()
    plt.plot(g['f']/1e9, 20 * np.log10(np.abs(g['H_ch'])), label="Channel Loss")
    plt.plot(f_base/1e9, 20 * np.log10(np.abs(H_base)), label="TX-RX Link")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Frequency Response")
    plt.legend()
    plt.grid(True)

# ============================================================================
# PART 2: DFE PROCESSING (Original lines 168-205)
# ============================================================================
# This section takes signal_jitter as input and produces signal_filtered as output

imp_ch = impinterp(np.fft.irfft(g['H_ch']), g['ratio_oversampling'])
imp_ch /= np.sum(np.abs(imp_ch))

# One bit response
pulse_resp_ch = cconv(g['pulse_signal'], imp_ch, g['pulse_signal_length'])

# Signal train after channel (MAIN OUTPUT: signal_filtered)
signal_filtered = sp.signal.fftconvolve(signal_jitter, imp_ch, mode="full")
signal_filtered = signal_filtered[0:len(signal_jitter)]  # trim to original length

# Plot pulse response using dedicated function (set plot=False to skip plotting here)
fig, ax = plot_pulse_response(t, g['pulse_signal'], pulse_resp_ch, g['os'], pulse_response_length,
                                num_left_cursors=5, num_right_cursors=9,
                                title=f"Pulse Response of End-to-End Channel for {pulse_response_length}UI duration",
                                plot=PLOT_PULSE_RESP)

# Analyze cursors and create table plot (set plot=False to skip plotting here)
fig_cursors, cursors, cursor_list, eye_h = analyze_and_plot_cursors(pulse_resp_ch, g['os'], 
                                                        num_pre=1, num_post=3,
                                                        title="Cursor Analysis with Values",
                                                        plot=PLOT_PULSE_RESP)

print("=" * 80)
print("PRELUDE SETUP AND DFE PROCESSING COMPLETE")
print("=" * 80)
print(f"Available global variables:")
print(f"  - g: Global dictionary with channel parameters")
print(f"  - signal_ideal: Ideal TX signal (without jitter)")
print(f"  - signal_jitter: TX signal with jitter applied")
print(f"  - signal_filtered: RX signal after channel filtering (MAIN OUTPUT)")
print(f"  - pulse_resp_ch: Pulse response of the channel")
print(f"  - imp_ch: Impulse response of the channel")
print(f"  - t: Time vector")
print(f"  - Ts: Sampling period")
print(f"  - data_rate: Data rate ({data_rate/1e9:.1f} Gbps)")
print("=" * 80)
