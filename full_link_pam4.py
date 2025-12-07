import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import skrf as rf
import warnings
import os
from pathlib import Path

## Custom Libraries
import serdespy as sdp
import sparam_modeling as sm
from sparam_modeling import gen_channel, frd_imp, cconv, impinterp, get_crossings
import sslms_dlev_adapt as sda
from sslms_dlev_adapt import sslms_dfe_dlev
from tx_8tap_ffe import *
from plot_functions import plot_pulse_response, analyze_and_plot_cursors

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Global variables dictionary to mimic MATLAB's global scope
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

def generate_pam4_signal(os, tx_launch_amp, ffe_taps=None, y_max=7.0):
    """
    Generate PAM4 signal using LUT-FFE with Gray coding.
    
    Parameters:
    -----------
    os : int
        Oversampling factor (samples per symbol)
    tx_launch_amp : float
        Transmitter launch amplitude
    ffe_taps : array-like, optional
        FFE tap weights (default: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    y_max : float
        Maximum output level for FFE (default: 7.0)
    
    Returns:
    --------
    signal_pam4_ideal : ndarray
        PAM4 signal repeated with oversampling
    """
    # Generate PRBS13 data and repeat 3 times
    data_pam4 = sdp.prbs13(1)
    data_pam4 = np.concatenate((data_pam4, data_pam4, data_pam4), axis=0)
    
    # Ensure even length
    if len(data_pam4) % 2 == 1:
        data_pam4 = data_pam4[:-1]
    
    # Split into MSB and LSB
    b1 = data_pam4[0::2]   # MSB
    b0 = data_pam4[1::2]   # LSB
    
    # Gray coding: g1 = MSB, g0 = MSB XOR LSB
    g1_bits = b1.astype(int)
    g0_bits = np.bitwise_xor(b1, b0).astype(int)
    
    # Set default FFE taps if not provided
    if ffe_taps is None:
        ffe_taps = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    
    # Create FFE and process signal
    ffe = Pam4LutFfe2Tap(ffe_taps, y_max=y_max)
    y_pam4 = np.empty_like(g1_bits, dtype=float)
    for i in range(len(g1_bits)):
        y_pam4[i] = ffe.step_analog(int(g1_bits[i]), int(g0_bits[i]))
    
    # Scale to launch amplitude
    scale = tx_launch_amp / 6.0  # 0.6/6 = 0.1 V per PAM4 step
    pam4_analog = scale * y_pam4  # -> {-0.3,-0.1,+0.1,+0.3} V for main-only case
    
    # Apply oversampling
    signal_pam4_ideal = np.repeat(pam4_analog, os)
    
    return signal_pam4_ideal

def main():
    # Constants from the main script
    PLOT_FREQ_RESP = True
    PLOT_PULSE_RESP = True
    ADD_RAND_JITTER = False
    TX_FFE = True

    # Global Variables
    data_rate = 112e9 #NRZ
    f_nyq = data_rate / 2
    g['ui'] = 1 / data_rate
    g['os'] = 128 #samples per symbol
    g['tx_launch_amp'] = 0.6
    g['num_pre_cursor'] = 1
    g['num_post_cursor'] = 4
    g['rterm_source'] = 50
    g['rterm_sink'] = 50
    print("Variable Initialized.\n")
    
    # Input Pulse Generation
    Ts = g['ui'] / g['os'] # Time step
    pulse_response_length = 100
    total_data_width = pulse_response_length * g['ui']
    pulse_start = 3 * g['ui']
    t = np.arange(0, total_data_width, Ts)
    g['pulse_signal'] = np.zeros_like(t)
    start_index = int(pulse_start / Ts)
    end_index = int(start_index + (1 * g['ui']) / Ts)
    g['pulse_signal'][start_index:end_index] = g['tx_launch_amp']
    g['pulse_signal_length'] = int(total_data_width / Ts)
    print("Input pulse is defined.\n")
    
    # TX to RX link
    script_dir = Path(__file__).resolve().parent
    s_param_dir = script_dir / "Channels"
    if not s_param_dir.is_dir():
        print(f"ERROR: S-parameter directory not found at '{s_param_dir}'")
        print("Please ensure the 'Channels' directory exists in the same directory as this script.")
        return
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
        pkg_s=s_param_dir / 'PKG100GEL_95ohm_30mm_50ohmPort.s4p', #Source Package
        pkg_l=s_param_dir / 'PKG100GEL_95ohm_30mm_50ohmPort.s4p', #Sink Package
        #ch=s_param_dir / '100G_PAM4_Cisco_c2c_thru_ch1.s4p', #Channel
        s_tcoil=False,
        s_tcoil_split = True,
        l_tcoil=False,
        l_tcoil_split = True,
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
        pkg_s=s_param_dir / 'PKG100GEL_95ohm_30mm_50ohmPort.s4p', #Source Package
        pkg_l=s_param_dir / 'PKG100GEL_95ohm_30mm_50ohmPort.s4p', #Sink Package
        #ch=s_param_dir / '100G_PAM4_Cisco_c2c_thru_ch1.s4p', #Channel
        s_tcoil=True,
        s_tcoil_split = True,
        l_tcoil=False,
        l_tcoil_split = True,
        pkg_s_portswap=True,
        pkg_l_portswap=True,
        ch_portswap=False
    )
    if g['H_ch'] is None: return # Exit if s-params failed to load
    print("Full link: Transfer Function evaluation completed.\n")
    
    # Impulse Response
    imp_ch_raw, Fs_ntwk = frd_imp(g['H_ch'], g['f'] * 2 * np.pi)
    Fs = 1/Ts
    g['ratio_oversampling'] = round(Fs / (2 * Fs_ntwk))
    print("Full link: Impulse Response evaluation completed.\n")
    
    # Output
    if PLOT_FREQ_RESP:
        fi_nyq = np.argmin(np.abs(g['f'] - f_nyq))
        H_ch_loss_at_nyquist = 20 * np.log10(np.abs(g['H_ch'][fi_nyq]))
        fi_rate = np.argmin(np.abs(g['f'] - data_rate))
        H_ch_loss_at_rate = 20 * np.log10(np.abs(g['H_ch'][fi_rate]))
        print(f"Loss at {f_nyq/1e9:.1f} GHz for end to end full link: {H_ch_loss_at_nyquist:.2f}dB")
        print(f"Loss at {data_rate/1e9:.1f} GHz for end to end full link: {H_ch_loss_at_rate:.2f}dB\n")
        plt.figure()
        plt.plot(g['f']/1e9, 20 * np.log10(np.abs(g['H_ch'])), label="Channel Loss")
        plt.plot(f_base/1e9, 20 * np.log10(np.abs(H_base)), label="TX-RX Link")
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("Frequency Response")
        plt.legend()
        plt.grid(True)
        
    if PLOT_PULSE_RESP:
        imp_ch = impinterp(np.fft.irfft(g['H_ch']), g['ratio_oversampling'])
        imp_ch /= np.sum(np.abs(imp_ch))
        #One bit response
        pulse_resp_ch = cconv(g['pulse_signal'], imp_ch, g['pulse_signal_length'])
        # Plot pulse response using dedicated function
        fig, ax = plot_pulse_response(t, g['pulse_signal'], pulse_resp_ch, g['os'], pulse_response_length,
                                      num_left_cursors=5, num_right_cursors=9,
                                      title=f"Pulse Response of End-to-End Channel for {pulse_response_length}UI duration")
        plt.tight_layout()
        
        # Analyze cursors and create table plot
        fig_cursors, cursors, cursors_list, eye_h = analyze_and_plot_cursors(pulse_resp_ch, g['os'], 
                                                               num_pre=1, num_post=6,
                                                               title="Cursor Analysis with Values")
        #Signal train after channel
        if TX_FFE:
            ffe_taps, g_eff = compute_tx_ffe_zero_forcing(cursors_list, num_taps=8)
            signal_pam4_ideal = generate_pam4_signal(os=g['os'], tx_launch_amp=g['tx_launch_amp'],ffe_taps=ffe_taps)
            signal_pam4_filtered = sp.signal.fftconvolve(signal_pam4_ideal, imp_ch, mode="full")
            signal_pam4_filtered = signal_pam4_filtered[0:len(signal_pam4_ideal)] #trim to original length
        else:
            signal_pam4_ideal = generate_pam4_signal(os=g['os'], tx_launch_amp=g['tx_launch_amp'])
            signal_pam4_filtered = sp.signal.fftconvolve(signal_pam4_ideal, imp_ch, mode="full")
            signal_pam4_filtered = signal_pam4_filtered[0:len(signal_pam4_ideal)] #trim to original length
        
    plt.show()

    #eye diagram of ideal NRZ signal
    if ADD_RAND_JITTER:
        sdp.simple_eye(signal_pam4_ideal, g['os']*3, 100, Ts, "{}Gbps Ideal PAM4 Signal".format(data_rate/1e9),linewidth=1.5)
    else:
        sdp.simple_eye(signal_pam4_ideal, g['os']*3, 100, Ts, "{}Gbps PAM4 Signal with Random Jitter".format(data_rate/1e9),linewidth=1.5)
                
    arr = signal_pam4_filtered[g['os']*100:g['os']*105]
    crossings = np.where(arr[:-1] * arr[1:] < 0)[0]
    zero_cross = crossings[0] if len(crossings) > 0 else 0
    
    #eye diagram of PAM4 signal after channel
    if TX_FFE:
        sdp.simple_eye(signal_pam4_filtered[g['os']*100+zero_cross+int(g['os']/2):], g['os']*2, 2000, Ts, "{}Gbps TX_FFE_PAM4 Signal after Channel".format(round(data_rate/1e9)))
    else:
        sdp.simple_eye(signal_pam4_filtered[g['os']*100+zero_cross+int(g['os']/2):], g['os']*2, 2000, Ts, "{}Gbps PAM4 Signal after Channel".format(round(data_rate/1e9)))

if __name__ == "__main__":
    main()
