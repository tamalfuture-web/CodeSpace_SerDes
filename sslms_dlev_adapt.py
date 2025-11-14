import numpy as np

def sslms_dfe_vref(signal, os, sampling_offset=0, num_taps=5, mu_taps=0.01, mu_dlev=0.01, dlev_init=0.0, sslms_start_iter=0):
    """
    """
    signal_downsampled = signal[sampling_offset::os]

    #TAP weights
    taps = np.zeros(num_taps)
    taps_history = np.zeros((num_taps, len(signal)))

    #Vref or dLEV
    dLev = dlev_init
    dLev_history = np.zeros(len(signal))

    #Past decisions
    d_history = np.zeros(num_taps)

    # Index of history arrays
    history_idx = sampling_offset

    iteration = 0

    for x_n in signal_downsampled:

    return taps, dLev, taps_history, dLev_history
