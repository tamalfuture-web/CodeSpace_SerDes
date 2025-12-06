################### Plotting Functions #########################
import numpy as np
import matplotlib.pyplot as plt

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
    fig : matplotlib figure
        Figure containing the plot
    cursors : dict
        Dictionary of all extracted cursor values (for detailed table display)
    cursors_list : numpy array
        Array of cursor values ordered as [pre5, pre4, pre3, pre2, pre1, main, post1, post2, post3, post4, post5, ...]
    eye_h : float
        Calculated eye height
    """
    # Find peak
    peak_idx = np.argmax(pulse_resp_ch)
    pulse_resp_main_crsr = pulse_resp_ch[peak_idx]
    
    # Extract cursors into both dict and list
    cursors = {}
    cursors['main (h0)'] = pulse_resp_main_crsr
    
    # Build cursors_list: pre-cursors (reversed order), main, then post-cursors
    cursors_list = []
    
    # Pre-cursors in reverse order (pre5, pre4, pre3, pre2, pre1)
    pre_values = []
    for i in range(1, 6):
        idx = peak_idx - i * os
        if idx >= 0:
            cursors[f'pre (h-{i})'] = pulse_resp_ch[idx]
            pre_values.insert(0, pulse_resp_ch[idx])  # Insert at beginning for reverse order
        else:
            pre_values.insert(0, 0.0)  # Pad with 0 if out of bounds
    
    cursors_list.extend(pre_values)
    cursors_list.append(pulse_resp_main_crsr)  # Main cursor
    
    # Post-cursors (post1, post2, post3, ...)
    for i in range(1, 10):
        idx = peak_idx + i * os
        if idx < len(pulse_resp_ch):
            cursors[f'post (h{i})'] = pulse_resp_ch[idx]
            cursors_list.append(pulse_resp_ch[idx])
        else:
            cursors_list.append(0.0)  # Pad with 0 if out of bounds
    
    # Convert to numpy array with float type and format to 4 decimal places
    cursors_list = np.array(cursors_list, dtype=np.float64)
    cursors_list = np.round(cursors_list, decimals=4)
    # Set print options to avoid scientific notation
    np.set_printoptions(suppress=True, precision=4, formatter={'float_kind': '{:.4f}'.format})
    
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
    return fig, cursors, cursors_list, eye_h
