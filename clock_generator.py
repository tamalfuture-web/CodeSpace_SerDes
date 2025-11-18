import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def generate_clock_signal(
    freq_hz: float,
    duration_s: float = 1e-6,
    fs_hz: float = 1e12,
    rj_rms_s: float = 0.0,
    dj_freq_hz: float = 0.0,
    dj_peak_s: float = 0.0,
    lpf_bw_factor: float = 10.0
):
    """
    Generates a complementary clock signal with optional RJ and DJ, passed through a low-pass filter.

    Args:
        freq_hz (float): The desired clock frequency in Hz.
        duration_s (float): The total duration of the signal in seconds.
        fs_hz (float): The sampling frequency in Hz. Must be high enough to avoid aliasing.
        rj_rms_s (float): The RMS value of Random Jitter in seconds.
        dj_freq_hz (float): The frequency of the Sinusoidal Jitter in Hz.
        dj_peak_s (float): The peak amplitude of the Sinusoidal Jitter in seconds.
        lpf_bw_factor (float): The factor by which to multiply the clock frequency to get the
                               low-pass filter's cutoff frequency.

    Returns:
        tuple: A tuple containing:
            - t (np.ndarray): The time vector for the signals.
            - clk_p (np.ndarray): The positive, filtered clock signal.
            - clk_n (np.ndarray): The negative, filtered clock signal.
            - f_welch (np.ndarray): Frequency offsets for the phase noise profile (in Hz).
            - phase_noise_dbchz (np.ndarray): SSB Phase Noise in dBc/Hz.
    """
    if not 1e9 <= freq_hz <= 30e9:
        raise ValueError("Frequency must be between 1 GHz and 30 GHz.")

    # 1. Create the time vector
    t = np.arange(0, duration_s, 1 / fs_hz)
    num_samples = len(t)

    # 2. Generate jitter signal j(t)
    jitter_signal = np.zeros(num_samples)
    # Add sinusoidal jitter (DJ)
    if dj_freq_hz > 0 and dj_peak_s > 0:
        jitter_signal += dj_peak_s * np.sin(2 * np.pi * dj_freq_hz * t)
    
    # Add random jitter (RJ)
    if rj_rms_s > 0:
        # Generate white noise and scale it to the correct RMS value
        # Note: This is a simplified model of broadband noise affecting timing.
        rj_noise = np.random.normal(0, 1, num_samples)
        # Scale noise to have the desired RMS value
        rj_noise = rj_noise * (rj_rms_s / np.sqrt(np.mean(rj_noise**2)))
        jitter_signal += rj_noise

    # 3. Generate the jittery clock
    # The jitter signal is added to the time vector inside the sine function's argument
    phase_jittered = 2 * np.pi * freq_hz * (t + jitter_signal)
    clk_p_ideal = np.sin(phase_jittered)
    clk_n_ideal = -clk_p_ideal

    # 4. Create and apply the low-pass filter
    lpf_cutoff_hz = freq_hz * lpf_bw_factor
    # Use a 2nd order Butterworth filter for a gentle rolloff
    b, a = signal.butter(2, lpf_cutoff_hz, btype='low', fs=fs_hz)
    clk_p = signal.lfilter(b, a, clk_p_ideal)
    clk_n = signal.lfilter(b, a, clk_n_ideal)

    # 5. Calculate the Phase Noise Profile
    # Use the Hilbert transform to get the analytic signal of the positive clock
    analytic_signal = signal.hilbert(clk_p)
    # Get the instantaneous phase and unwrap it
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    # The ideal phase is a linear ramp. Subtract it to get the phase deviation.
    ideal_phase = 2 * np.pi * freq_hz * t
    phase_deviation_rad = instantaneous_phase - ideal_phase
    
    # Calculate the Power Spectral Density (PSD) of the phase deviation
    # This gives us the phase noise in rad^2/Hz
    f_welch, psd_phase_deviation = signal.welch(
        phase_deviation_rad,
        fs=fs_hz,
        nperseg=fs_hz / (freq_hz / 100), # Use a window size that gives good resolution
        scaling='density'
    )
    
    # Convert to Single-Sideband (SSB) phase noise in dBc/Hz
    # L(f) = 10 * log10( 0.5 * S_phi(f) )
    phase_noise_dbchz = 10 * np.log10(0.5 * psd_phase_deviation)

    return t, clk_p, clk_n, f_welch, phase_noise_dbchz


if __name__ == '__main__':
    # --- Example Usage ---
    # Parameters
    clock_freq = 10e9  # 10 GHz
    sampling_freq = 2e12 # 2 THz sampling rate (200x oversampling)
    sim_duration = 2e-7 # 200 ns simulation time
    
    # Jitter components
    rj_rms = 100e-15    # 100 fs RMS
    dj_amp = 250e-15    # 250 fs peak
    dj_freq = 100e6     # 100 MHz

    print(f"Generating {clock_freq/1e9} GHz clock...")

    t, clk_p, clk_n, f_noise, pn_dbchz = generate_clock_signal(
        freq_hz=clock_freq,
        duration_s=sim_duration,
        fs_hz=sampling_freq,
        rj_rms_s=rj_rms,
        dj_freq_hz=dj_freq,
        dj_peak_s=dj_amp
    )

    print("Generation complete. Plotting results...")

    # --- Plotting ---
    
    # 1. Plot a small segment of the clock signals
    plt.figure(figsize=(12, 8))
    plt.suptitle(f"{clock_freq/1e9} GHz Clock with Jitter", fontsize=16)

    ax1 = plt.subplot(2, 1, 1)
    # Display 5 clock cycles
    num_cycles_to_plot = 5
    plot_end_index = int(num_cycles_to_plot / clock_freq * sampling_freq)
    
    ax1.plot(t[:plot_end_index] * 1e12, clk_p[:plot_end_index], label='Clock+')
    ax1.plot(t[:plot_end_index] * 1e12, clk_n[:plot_end_index], label='Clock-', alpha=0.8)
    ax1.set_title("Complementary Clock Waveform")
    ax1.set_xlabel("Time (ps)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)
    ax1.legend()

    # 2. Plot the Phase Noise Profile
    ax2 = plt.subplot(2, 1, 2)
    # Plot from 1 kHz to 1 GHz offset
    ax2.semilogx(f_noise, pn_dbchz)
    ax2.set_title("Phase Noise Profile")
    ax2.set_xlabel("Frequency Offset (Hz)")
    ax2.set_ylabel("SSB Phase Noise (dBc/Hz)")
    ax2.grid(True, which='both')
    ax2.set_xlim(1e3, 1e9)
    ax2.set_ylim(-160, -40)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("plots/clock_generator_output.png")
    print("\nPlot saved to 'plots/clock_generator_output.png'")
    
    # To avoid issues in headless environments, we save the plot instead of showing it.
    # If you are in a graphical environment, you can uncomment the next line.
    # plt.show()
