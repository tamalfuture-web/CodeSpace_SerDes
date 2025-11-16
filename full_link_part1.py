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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

################### Plotting Functions #########################

def plot_pulse_response(t, pulse_signal, pulse_resp_ch, Ts, pulse_response_length, num_left_cursors=5, num_right_cursors=9, title="Pulse Response"):
    """
    Plot pulse response with cursor circles and data level labels (no grids).
    
    Parameters:
    -----------
    t : array
        Time vector
    pulse_signal : array
        Input pulse signal
    pulse_resp_ch : array
        Channel response signal
    Ts : float
        Sampling period (for cursor spacing)
    pulse_response_length : int
        Number of UI in the pulse response
    num_left_cursors : int
        Number of pre-cursors to the left of peak
    num_right_cursors : int
        Number of post-cursors to the right of peak
    title : str
        Title of the plot
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor('white')  # White background, no gray
    
    # Plot signals
    ax.plot(t/1e-12, pulse_signal, label="Input Pulse", linewidth=2, marker='o', markersize=4, alpha=0.7, color='blue')
    ax.plot(t/1e-12, pulse_resp_ch, label="Channel Response", linewidth=2.5, color='darkblue')
    
    # Find peak (argmax) of pulse response
    peak_idx = np.argmax(pulse_resp_ch)
    peak_time = t[peak_idx] / 1e-12  # Convert to ps
    peak_value = pulse_resp_ch[peak_idx]
    
    # Add circle at peak with label
    ax.plot(peak_time, peak_value, marker='o', markersize=12, color='green', markerfacecolor='none', 
            markeredgewidth=2.5, zorder=5, label=f'Peak h0={peak_value:.4f}')
    ax.text(peak_time, peak_value + 0.02, f'h0\n{peak_value:.4f}', 
           fontsize=9, ha='center', fontweight='bold', color='green')
    
    # Calculate index offset for one Ts
    Ts_samples = int(Ts / (t[1] - t[0]))  # Number of samples per Ts
    Ts_ps = Ts * 1e12  # Convert Ts to picoseconds
    
    # Left cursors (pre-cursors)
    for i in range(1, num_left_cursors + 1):
        cursor_idx = peak_idx - i * Ts_samples
        if 0 <= cursor_idx < len(pulse_resp_ch):
            cursor_time = t[cursor_idx] / 1e-12
            cursor_val = pulse_resp_ch[cursor_idx]
            
            # Circle marker
            ax.plot(cursor_time, cursor_val, marker='o', markersize=10, color='orange', 
                   markerfacecolor='none', markeredgewidth=2, zorder=5)
            # Data level label
            ax.text(cursor_time, cursor_val + 0.015, f'h-{i}\n{cursor_val:.4f}', 
                   fontsize=8, ha='center', color='orange', fontweight='bold')
    
    # Right cursors (post-cursors)
    for i in range(1, num_right_cursors + 1):
        cursor_idx = peak_idx + i * Ts_samples
        if 0 <= cursor_idx < len(pulse_resp_ch):
            cursor_time = t[cursor_idx] / 1e-12
            cursor_val = pulse_resp_ch[cursor_idx]
            
            # Circle marker
            ax.plot(cursor_time, cursor_val, marker='o', markersize=10, color='red', 
                   markerfacecolor='none', markeredgewidth=2, zorder=5)
            # Data level label
            ax.text(cursor_time, cursor_val - 0.025, f'h{i}\n{cursor_val:.4f}', 
                   fontsize=8, ha='center', color='red', fontweight='bold')
    
    ax.set_xlabel("Time (ps)", fontsize=11)
    ax.set_ylabel("Amplitude (V)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig, ax


def analyze_and_plot_cursors(pulse_resp_ch, os, num_pre=1, num_post=3, title="Cursor Analysis"):
    """
    Extract cursor values and plot with embedded table showing all cursor positions.
    
    Parameters:
    -----------
    pulse_resp_ch : array
        Channel pulse response signal
    os : int
        Oversampling factor (samples per symbol)
    num_pre : int
        Number of pre-cursors to display on plot (default 1)
    num_post : int
        Number of post-cursors to display on plot (default 3)
    title : str
        Title of the plot
    
    Returns:
    --------
    cursors : dict
        Dictionary of all extracted cursor values
    eye_h : float
        Calculated eye height
    """
    # Find peak
    peak_idx = np.argmax(pulse_resp_ch)
    pulse_resp_main_crsr = pulse_resp_ch[peak_idx]
    
    # Extract cursors
    cursors = {}
    cursors['main (h0)'] = pulse_resp_main_crsr
    
    # Pre-cursors (up to 5)
    for i in range(1, 6):
        idx = peak_idx - i * os
        if idx >= 0:
            cursors[f'pre (h-{i})'] = pulse_resp_ch[idx]
    
    # Post-cursors (up to 9)
    for i in range(1, 10):
        idx = peak_idx + i * os
        if idx < len(pulse_resp_ch):
            cursors[f'post (h{i})'] = pulse_resp_ch[idx]
    
    # Print to console
    print("Extracted Cursor Values:")
    total_cursor_variation = 0
    for name, val in cursors.items():
        print(f"  {name:<12}: {val:.4f}")
        if "main" not in name:
            total_cursor_variation += val
    
    eye_h = pulse_resp_main_crsr - total_cursor_variation
    if eye_h > 0:
        print(f"Eye height: {eye_h:.3f}V and Vref: +/-{pulse_resp_main_crsr:.3f}V")
    else:
        print("Eye is closed")
    
    # Create figure with plot and embedded table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor('white')
    
    # Plot pulse response with limited cursors (1 pre, 3 post on plot)
    sample_indices = np.arange(len(pulse_resp_ch))
    ax.plot(sample_indices, pulse_resp_ch, linewidth=2.5, color='darkblue', label='Pulse Response')
    
    # Peak
    ax.plot(peak_idx, pulse_resp_main_crsr, marker='o', markersize=12, color='green', 
           markerfacecolor='none', markeredgewidth=2.5, zorder=5, label=f'Peak h0={pulse_resp_main_crsr:.4f}')
    
    # Plot pre-cursors (only 1 on plot)
    if 1 * os <= peak_idx:
        idx = peak_idx - 1 * os
        val = pulse_resp_ch[idx]
        ax.plot(idx, val, marker='o', markersize=10, color='orange', markerfacecolor='none', 
               markeredgewidth=2, zorder=5)
        ax.text(idx, val + 0.015, f'h-1\n{val:.4f}', fontsize=9, ha='center', color='orange', fontweight='bold')
    
    # Plot post-cursors (up to 3 on plot)
    for i in range(1, min(4, num_post + 1)):
        idx = peak_idx + i * os
        if idx < len(pulse_resp_ch):
            val = pulse_resp_ch[idx]
            ax.plot(idx, val, marker='o', markersize=10, color='red', markerfacecolor='none', 
                   markeredgewidth=2, zorder=5)
            ax.text(idx, val - 0.025, f'h{i}\n{val:.4f}', fontsize=9, ha='center', color='red', fontweight='bold')
    
    ax.set_xlabel("Sample Index", fontsize=11)
    ax.set_ylabel("Amplitude (V)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Create table with all cursor values
    table_data = []
    for name, val in cursors.items():
        table_data.append([name, f'{val:.6f}'])
    
    # Add eye height info
    table_data.append(['Eye Height', f'{eye_h:.6f}V'])
    table_data.append(['Vref', f'{pulse_resp_main_crsr:.6f}V'])
    
    # Embed table in figure
    table = ax.table(cellText=table_data, 
                    colLabels=['Cursor', 'Value (V)'],
                    cellLoc='center',
                    loc='center left',
                    bbox=[1.05, 0.0, 0.35, 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.tight_layout()
    return fig, cursors, eye_h

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

def main():
    # Constants from the main script
    PLOT_FREQ_RESP = True
    PLOT_PULSE_RESP = True
    ADD_RAND_JITTER = False

    # Global Variables
    data_rate = 56e9 #NRZ
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

    #generate binary data
    data = sdp.prbs13(1)
    signal_BR = sdp.nrz_input_BR(data) #generate Baud-Rate sampled signal from data
    signal_ideal = 0.5*g['tx_launch_amp'] * np.repeat(signal_BR, g['os']) #oversample to get Tx signal
    print("PRBS signal train is generated.\n")
    if ADD_RAND_JITTER:
        signal_jitter = sdp.gaussian_jitter(signal_ideal, g['ui'], len(data), g['os'], stdev=1000e-15)
        print("Random Jitter is added.\n")
    else:
        signal_jitter = signal_ideal
    
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
        #Signal train after channel
        signal_filtered = sp.signal.fftconvolve(signal_jitter, imp_ch, mode="full")
        signal_filtered = signal_filtered[0:len(signal_jitter)] #trim to original length
        
        # Plot pulse response using dedicated function
        fig, ax = plot_pulse_response(t, g['pulse_signal'], pulse_resp_ch, g['os'], pulse_response_length,
                                      num_left_cursors=5, num_right_cursors=9,
                                      title=f"Pulse Response of End-to-End Channel for {pulse_response_length}UI duration")
        plt.tight_layout()
        
        # Analyze cursors and create table plot
        fig_cursors, cursors, eye_h = analyze_and_plot_cursors(pulse_resp_ch, g['os'], 
                                                               num_pre=1, num_post=3,
                                                               title="Cursor Analysis with Values")
        
    plt.show()

    #eye diagram of ideal NRZ signal
    if ADD_RAND_JITTER:
        sdp.simple_eye(signal_ideal, g['os']*3, 100, Ts, "{}Gbps Ideal NRZ Signal".format(data_rate/1e9),linewidth=1.5)
    else:
        sdp.simple_eye(signal_jitter, g['os']*3, 100, Ts, "{}Gbps NRZ Signal with Random Jitter".format(data_rate/1e9),linewidth=1.5)
                
    arr = signal_filtered[g['os']*100:g['os']*105]
    crossings = np.where(arr[:-1] * arr[1:] < 0)[0]
    zero_cross = crossings[0] if len(crossings) > 0 else None
    
    #Time domain SIgnal waveforms
    plt.figure()
    time = Ts*np.arange(len(signal_ideal[0:13000]))
    plt.plot(time[0:13000], signal_ideal[0:13000])
    plt.plot(time[0:13000], signal_filtered[0:13000])
    plt.title("Time Domain Signal Waveforms")
    #eye diagram of NRZ signal after channel
    sdp.simple_eye(signal_filtered[g['os']*100+zero_cross+int(g['os']/2):], g['os']*2, 2000, Ts, "{}Gbps NRZ Signal after Channel".format(round(data_rate/1e9)))
    
    # VREF and DFE taps adaptation

    sampling_offset = np.argmax(pulse_resp_ch)
    print(f"\nSampling offset: {sampling_offset} samples")
    dLev_init = g['tx_launch_amp'] / 2
    num_taps = g['num_post_cursor']
    mu_taps = 1e-3
    mu_dlev = 1e-5
    delta_dLev = 1e-3
    sslms_start_iter = 1000

    tap, dLev, tap_history, dLev_history = sslms_dfe_dlev(signal_filtered, g['os'], sampling_offset, num_taps, mu_taps, mu_dlev, dLev_init, delta_dLev, sslms_start_iter, plot=True)
    print("\nSSLMS DFE and Vref adaptation completed.\n")
    print(f"Final DFE taps: {tap} \nFinal Vref: {dLev:0.2f}\n")
    
if __name__ == "__main__":
    main()
