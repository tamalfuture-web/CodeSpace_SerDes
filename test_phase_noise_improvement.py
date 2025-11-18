"""
Test script to demonstrate the phase noise improvements
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Simple test: jitter components only
clock_freq = 10e9
samples_per_ui = 256
duration_ui = 2000
ui = 1 / clock_freq

fs_hz = samples_per_ui * clock_freq
num_samples = duration_ui * samples_per_ui
t = np.arange(0, num_samples) * (ui / samples_per_ui)

# Create jitter signal with known components
jitter_signal = np.zeros(num_samples)

# Add 100 MHz sinusoidal jitter (2.5% of UI)
dj_peak_ui = 0.025
dj_peak_s = dj_peak_ui * ui
dj_freq = 100e6
jitter_signal += dj_peak_s * np.sin(2 * np.pi * dj_freq * t)

# Add 1% RMS random jitter
rj_rms_ui = 0.01
rj_rms_s = rj_rms_ui * ui
rj_noise = np.random.normal(0, 1, num_samples)
rj_noise = rj_noise * (rj_rms_s / np.sqrt(np.mean(rj_noise**2)))
jitter_signal += rj_noise

# Phase deviation directly from jitter
phase_deviation = 2 * np.pi * clock_freq * jitter_signal

# Calculate PSD with improved parameters
nperseg = min(int(fs_hz / 1e6), len(t) // 4)
f_welch, psd_phase = signal.welch(
    phase_deviation,
    fs=fs_hz,
    nperseg=nperseg,
    noverlap=nperseg // 2,
    scaling='density',
    window='blackmanharris'
)

# Convert to dBc/Hz
phase_noise_dbchz = 10 * np.log10(np.maximum(0.5 * psd_phase, 1e-20))

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Jitter signal (first 100 ns)
plot_time = 100e-9  # 100 ns
plot_idx = int(plot_time * fs_hz)
ax1.plot(t[:plot_idx] * 1e9, jitter_signal[:plot_idx] * 1e12, linewidth=1)
ax1.set_xlabel('Time (ns)')
ax1.set_ylabel('Jitter (ps)')
ax1.set_title('Jitter Signal (RJ + DJ @ 100 MHz)')
ax1.grid(True, alpha=0.3)

# Plot 2: Phase noise with 100 MHz marked
ax2.semilogx(f_welch, phase_noise_dbchz, linewidth=2, label='Phase Noise')
ax2.axvline(x=100e6, color='red', linestyle='--', linewidth=2, label='100 MHz (DJ frequency)')
ax2.set_xlabel('Frequency Offset (Hz)')
ax2.set_ylabel('SSB Phase Noise (dBc/Hz)')
ax2.set_title('Phase Noise Profile - Showing 100 MHz DJ Peak')
ax2.grid(True, which='both', alpha=0.3)
ax2.set_xlim(1e3, 1e9)
ax2.legend()

# Find peak near 100 MHz
mask_100mhz = (f_welch > 50e6) & (f_welch < 150e6)
if np.any(mask_100mhz):
    peak_idx = np.argmax(phase_noise_dbchz[mask_100mhz])
    peak_freq = f_welch[np.where(mask_100mhz)[0][peak_idx]]
    peak_level = phase_noise_dbchz[np.where(mask_100mhz)[0][peak_idx]]
    ax2.plot(peak_freq, peak_level, 'ro', markersize=10, label=f'Peak @ {peak_freq/1e6:.1f} MHz: {peak_level:.1f} dBc/Hz')
    ax2.legend()
    print(f"✓ 100 MHz peak found:")
    print(f"  Frequency: {peak_freq/1e6:.2f} MHz")
    print(f"  Level: {peak_level:.1f} dBc/Hz")

plt.tight_layout()
plt.savefig('plots/phase_noise_improvement_test.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to 'plots/phase_noise_improvement_test.png'")
