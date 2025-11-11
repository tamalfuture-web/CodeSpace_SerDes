#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 18:57:52 2025

@author: tamaldas
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 10          # sampling frequency (Hz)
T = 1/fs
t = np.linspace(0, 1, 2000)  # continuous time
f0 = 2           # base frequency (Hz)

# Continuous signal: sin + sin^2
x = np.sin(2*np.pi*f0*t) 

# Sample times and samples
n = np.arange(0, 1, T)
xs = np.sin(2*np.pi*f0*n) 

# Dirac comb in time (aligned with sample times)
t_dirac = n
amp_time = np.ones_like(t_dirac)

# Frequency domain: continuous spectrum approximation
Nfft = 4096
X = np.fft.fftshift(np.fft.fft(x, Nfft))
freqs = np.fft.fftshift(np.fft.fftfreq(Nfft, d=t[1]-t[0]))

# Replicated spectrum after sampling
f = np.linspace(-3*fs, 3*fs, 6000)
Xs = np.zeros_like(f)
base_spectrum = np.interp(f, freqs, np.abs(X)/max(np.abs(X)), left=0, right=0)
for k in range(-3, 4):
    shifted = np.interp(f - k*fs, freqs, np.abs(X)/max(np.abs(X)), left=0, right=0)
    Xs += shifted

# Dirac comb in frequency
k = np.arange(-6, 7)
f_dirac = k * fs
amp_freq = np.ones_like(f_dirac)

# Plotting
fig, axs = plt.subplots(2, 3, figsize=(14, 6))

# Row 1: time domain
axs[0,0].plot(t, x, 'b')
axs[0,0].set_title("Continuous signal $x(t)=\\sin(2\\pi f_0 t)$")
axs[0,0].set_xlabel("Time [s]"); axs[0,0].set_ylabel("Amplitude")

axs[0,1].stem(t_dirac, amp_time, basefmt=" ", linefmt="C2-", markerfmt="C2o")
axs[0,1].set_title("Dirac comb in time")
axs[0,1].set_xlabel("Time [s]"); axs[0,1].set_ylabel("Amplitude")

axs[0,2].stem(n, xs, basefmt=" ", linefmt="C1-", markerfmt="C1o")
axs[0,2].set_title("Sampled signal $x_s(t)$")
axs[0,2].set_xlabel("Time [s]"); axs[0,2].set_ylabel("Amplitude")

# Row 2: frequency domain
axs[1,0].plot(freqs, np.abs(X)/max(np.abs(X)), 'b')
axs[1,0].set_xlim(-fs*2, fs*2)
axs[1,0].set_title("Spectrum of $x(t)$")
axs[1,0].set_xlabel("Frequency [Hz]"); axs[1,0].set_ylabel("Normalized |X(f)|")

axs[1,1].stem(f_dirac, amp_freq, basefmt=" ", linefmt="C2-", markerfmt="C2o")
axs[1,1].set_xlim(-3*fs, 3*fs)
axs[1,1].set_title("Dirac comb in frequency")
axs[1,1].set_xlabel("Frequency [Hz]"); axs[1,1].set_ylabel("Amplitude")

axs[1,2].plot(f, Xs, 'r')
axs[1,2].set_title("Spectrum after sampling")
axs[1,2].set_xlabel("Frequency [Hz]"); axs[1,2].set_ylabel("|Xs(f)|")

plt.tight_layout()
plt.show()
