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
    # Define project paths relative to this script's location
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
        ch=s_param_dir / '100G_PAM4_Cisco_c2c_thru_ch1.s4p', #Channel
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
        ch=s_param_dir / '100G_PAM4_Cisco_c2c_thru_ch1.s4p', #Channel
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
        
        #Single One response
        pulse_resp_ch = cconv(g['pulse_signal'], imp_ch, g['pulse_signal_length'])

        #Signal train after channel
        signal_filtered = sp.signal.fftconvolve(signal_jitter, imp_ch, mode="full")
        
        plt.figure()
        plt.plot(t/1e-12, g['pulse_signal'], label="Input Pulse")
        plt.plot(t/1e-12, pulse_resp_ch, label="Channel Response")
        plt.title(f"Pulse Response of End-to-End Channel for {pulse_response_length}UI duration")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)

        # Cursor measurement
        pulse_resp_main_crsr = np.max(pulse_resp_ch)
        
        cursors = {}
        cursors['main (h0)'] = pulse_resp_main_crsr/1

        # Extract pre-cursors
        for i in range(1, 6):
            idx = np.argmax(pulse_resp_ch) - i * g['os']
            if idx >= 0:
                cursors[f'pre (h-{i})'] = pulse_resp_ch[idx]/1

        # Extract post-cursors
        for i in range(1, 10):
            idx = np.argmax(pulse_resp_ch) + i * g['os']
            if idx < len(pulse_resp_ch):
                cursors[f'post (h{i})'] = pulse_resp_ch[idx]/1

        # Print the results
        print("Extracted Cursor Values:")
        total_cursor_variation = 0
        for name, val in cursors.items():
            #print(f"  {name:<12}: {val:.4f}")
            if not "main" in name:
                total_cursor_variation += val
        
        # Eye height calculation
        eye_h = pulse_resp_main_crsr - total_cursor_variation
        if eye_h > 0:
            print(f"Eye height:{eye_h: 0.3f}V and Vref: +/-{pulse_resp_main_crsr: 0.3f}V")
        else:
            print("Eye is closed")
        
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
    
    

if __name__ == "__main__":
    main()
