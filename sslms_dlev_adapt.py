import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def sslms_dfe_dlev(signal, os, sampling_offset=0, num_taps=5, mu_taps=0.01, mu_dlev=0.01, dLev_init=0.0, delta_dLev=1e-3, sslms_start_iter=0, plot=False, Ts=None):
    """
    SSLMS DFE with adaptive decision level.
    
    Returns:
        Tuple containing:
        - taps: Final DFE tap weights
        - dLev: Final decision level (Vref)
        - taps_history: History of tap values over time
        - dLev_history: History of decision level over time
        - equalized_signal: Full-resolution equalized signal (same length as input)
        - decisions: Full-resolution symbol decisions (same length as input)
    """
    signal_downsampled = signal[sampling_offset::os]

    #TAP weights
    taps = np.zeros(num_taps)
    taps_history = np.zeros((num_taps, len(signal)))

    #Vref or dLEV
    dLev = dLev_init
    dLev_history = np.zeros(len(signal))

    #Past decisions
    d_history = np.zeros(num_taps)

    # Index of history arrays
    history_idx = sampling_offset

    iteration = 0

    for x_n in signal_downsampled:

        y_n = x_n - np.dot(taps, d_history)

        #Data Slicer
        d_n = np.sign(y_n)

        #Error Slicer
        e_cont = y_n - dLev * d_n

        ##SSLMS
        if iteration > sslms_start_iter:
            #Update TAP weights
            taps += mu_taps * np.sign(e_cont) * d_history

        if iteration <= sslms_start_iter:
            #Update dLEV
            dLev += mu_dlev * e_cont
        else:
            up = 1 if (y_n * d_n > dLev + delta_dLev) else 0
            down = 1 if (y_n * d_n < dLev - delta_dLev) else 0
            dLev += mu_dlev * (up - down)

        d_history = np.roll(d_history, 1)
        d_history[0] = d_n

        # Store history
        if history_idx + os < len(signal):
            taps_history[:, history_idx + os] = taps
            dLev_history[history_idx:history_idx + os] = dLev
        else:
            taps_history[:, history_idx] = taps[:]
            dLev_history[history_idx] = dLev

        history_idx += os
        iteration += 1

    # Optional plotting
    if plot:
        # Ensure plots directory exists
        plots_dir = Path(__file__).resolve().parent / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Generate time axis
        if Ts is None:
            time_axis = np.arange(len(signal))
            time_label = 'Sample Index'
        else:
            time_axis = np.arange(len(signal)) * Ts
            time_label = 'Time (s)'
        
        # Plot 1: First 2 tap histories
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(time_axis, taps_history[0, :], label='Tap 1', linewidth=1.5)
        if num_taps > 1:
            ax1.plot(time_axis, taps_history[1, :], label='Tap 2', linewidth=1.5)
        ax1.set_xlabel(time_label)
        ax1.set_ylabel('Tap Value')
        ax1.set_title('Evolution of First 2 Tap Weights')
        ax1.legend()
        ax1.grid(True)
        plot1_path = plots_dir / 'sslms_tap_weights_history.png'
        fig.savefig(plot1_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {plot1_path}")
        
        # Plot 2: Signal filtered and dLev overlayed
        fig, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(time_axis, signal, label='Input Signal', linewidth=1, alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(time_axis, dLev_history, label='dLev (Slicer Level)', color='red', linewidth=1.5)
        ax2_twin.plot(time_axis, -dLev_history, label='-dLev', color='red', linewidth=1.5, linestyle='--')
        ax2.set_xlabel(time_label)
        ax2.set_ylabel('Signal Amplitude', color='blue')
        ax2_twin.set_ylabel('dLev (Slicer Level)', color='red')
        ax2.set_title('Signal with Adaptive Decision Levels (dLev)')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        plot2_path = plots_dir / 'sslms_signal_with_dlev.png'
        fig.savefig(plot2_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {plot2_path}")

    # Compute full-resolution equalized signal and decisions
    # Use final tap weights for full equalization
    equalized_signal = np.zeros(len(signal))
    decisions = np.zeros(len(signal))
    
    # Initialize decision history buffer
    d_history_full = np.zeros(num_taps)
    
    # Process all samples (not just downsampled)
    for i in range(len(signal)):
        # Get appropriate tap weights (interpolate from history if needed)
        if i >= len(taps_history[0]):
            taps_final = taps_history[:, -1]  # Use last known taps
        else:
            taps_final = taps_history[:, i]
        
        # Get appropriate decision level
        if i >= len(dLev_history):
            dLev_final = dLev_history[-1]  # Use last known dLev
        else:
            dLev_final = dLev_history[i]
        
        # Apply DFE equalization
        y_eq = signal[i] - np.dot(taps_final, d_history_full)
        equalized_signal[i] = y_eq
        
        # Make symbol decision
        d_i = np.sign(y_eq)
        decisions[i] = d_i
        
        # Update decision history
        d_history_full = np.roll(d_history_full, 1)
        d_history_full[0] = d_i
    
    return taps, dLev, taps_history, dLev_history, equalized_signal, decisions