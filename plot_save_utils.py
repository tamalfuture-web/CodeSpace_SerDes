"""
Plot Saving Utilities Module

This module provides functions to automatically save matplotlib figures to the plots/ directory
with timestamped filenames. Instead of the old approach where a wrapper called the main script,
this allows scripts to import and use the plotting utilities directly.

Usage:
    from plot_save_utils import setup_plot_saving, save_all_plots
    
    # Setup plt.show() to automatically save plots
    setup_plot_saving()
    
    # Your plotting code here...
    plt.plot(...)
    plt.show()  # This will automatically save instead of displaying
    
    # Or manually save all current figures at any time
    save_all_plots()
"""

import matplotlib.pyplot as plt
from pathlib import Path
import time
import re
import sys


# Ensure the workspace root is on sys.path
workspace_root = Path.cwd()


def _safe_filename(s):
    """Convert string to safe filename."""
    s = s.strip()
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^A-Za-z0-9_\-]', '', s)
    return s or None


def save_and_close(*args, **kwargs):
    """
    Save all open matplotlib figures to plots/ directory with timestamped filenames.
    This function is designed to replace plt.show().
    """
    outdir = workspace_root / 'plots'
    outdir.mkdir(exist_ok=True)
    saved = []
    ts = time.strftime('%Y%m%d_%H%M%S')
    
    for i, num in enumerate(plt.get_fignums(), start=1):
        fig = plt.figure(num)
        # Try to build a meaningful filename from figure title
        title = None
        try:
            if fig.axes and len(fig.axes) > 0:
                title = fig.axes[0].get_title()
        except Exception:
            title = None

        safe_title = _safe_filename(title) if title else None
        base = safe_title if safe_title else f'plot_{i}'
        # avoid overwriting by appending timestamp
        filename = f"{base}_{ts}.png"
        path = outdir / filename
        fig.savefig(path, dpi=200)
        saved.append(str(path))
    
    if saved:
        print('Saved figures:')
        for s in saved:
            print('  ', s)
    plt.close('all')


def setup_plot_saving():
    """
    Setup matplotlib to automatically save plots instead of displaying them.
    This replaces plt.show() with save_and_close().
    
    Call this function early in your script before any plotting code.
    """
    # Force headless matplotlib backend
    import matplotlib
    matplotlib.use('Agg')
    
    # Override plt.show to save plots
    plt.show = save_and_close


def save_all_plots(manual=True):
    """
    Manually save all currently open figures to plots/ directory.
    
    Parameters:
    -----------
    manual : bool
        If True, this is a manual save call (not from plt.show() override).
        Will preserve the message format.
    """
    if manual:
        print("\nManually saving all open figures...")
    save_and_close()


# Auto-setup when module is imported (optional)
# Uncomment the line below if you want automatic setup on import
# setup_plot_saving()
