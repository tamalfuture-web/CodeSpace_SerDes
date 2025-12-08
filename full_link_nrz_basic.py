"""
Full Link Analysis - Part 1: Basic NRZ Signal through Channel
This script uses the prelude_setup_and_dfe.py file to handle common setup and DFE processing.
The prelude initializes global variables and produces signal_filtered as output.
The script also uses plot_save_utils.py to automatically save all plots to the plots/ directory.
- Frequency response evaluation of the end-to-end channel with t-coils
- Impulse and pulse response analysis
- Time-domain signal waveform visualization and EYE diagrams
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from pathlib import Path
import serdespy as sdp

# Setup plot saving (this will intercept plt.show() and save plots instead)
from plot_save_utils import setup_plot_saving
setup_plot_saving()

# ============================================================================
# Load the prelude to do common setup
# ============================================================================
# This will initialize all global variables including:
# - g: global dictionary
# - signal_ideal, signal_jitter, signal_filtered
# - pulse_resp_ch, imp_ch, t, Ts, data_rate, etc.

exec(open('prelude_setup_and_dfe.py').read())

# ============================================================================
# Additional processing specific to full_link_part1.py
# ============================================================================

def main():

    # Eye diagram of ideal NRZ signal
    if ADD_RAND_JITTER:
        sdp.simple_eye(signal_ideal, g['os']*3, 100, Ts, "{}Gbps Ideal NRZ Signal".format(data_rate/1e9), linewidth=1.5)
    else:
        sdp.simple_eye(signal_jitter, g['os']*3, 100, Ts, "{}Gbps NRZ Signal with Random Jitter".format(data_rate/1e9), linewidth=1.5)
        
    # Time domain Signal waveforms
    plt.figure()
    time = Ts * np.arange(len(signal_ideal[0:13000]))
    plt.plot(time[0:13000], signal_ideal[0:13000], label='Ideal Signal')
    plt.plot(time[0:13000], signal_filtered[0:13000], label='Filtered Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    plt.title("Time Domain Signal Waveforms")
    plt.legend()
    plt.grid(True)
    
    # Eye diagram of NRZ signal after channel
    # Find zero crossing
    arr = signal_filtered[g['os']*100:g['os']*105]
    crossings = np.where(arr[:-1] * arr[1:] < 0)[0]
    zero_cross = crossings[0] if len(crossings) > 0 else 0
    sdp.simple_eye(signal_filtered[g['os']*100+zero_cross+int(g['os']/2):], g['os']*2, 2000, Ts, 
                   "{}Gbps NRZ Signal after Channel".format(round(data_rate/1e9)))
           
    plt.show()

if __name__ == "__main__":
    main()
