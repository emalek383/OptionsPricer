""" Module for plotting the heatmap of the option's PnL as fct of vol/spot. """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

def plot_heatmap(option_pnl, spots, vols):
    """
    Calculate and plot a heatmap showing PnL of an option as a function of spot price and volatility.

    Parameters
    ----------
    option_pnl : nump.ndarray
        2D array of PnL values.
    spots : numpy.ndarray
        1D array of spot prices.
    vols : numpy.ndarray
        1D array of volatilities.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with the heatmap plot.
        
    """

    vmin = option_pnl.min()
    vmax = option_pnl.max()
    
    if vmin >= 0:  # All positive values
        colors = plt.cm.Greens(np.linspace(0.2, 0.8, 256))
        cmap = LinearSegmentedColormap.from_list('CustomGreen', colors)
        norm = Normalize(vmin=vmin, vmax=vmax)
    elif vmax <= 0:  # All negative values
        colors = plt.cm.Reds_r(np.linspace(0.2, 0.8, 256))
        cmap = LinearSegmentedColormap.from_list('CustomRed', colors)
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:  # Mix of positive and negative values
        neg_colors = plt.cm.Reds_r(np.linspace(0.2, 0.8, 128))
        pos_colors = plt.cm.Greens(np.linspace(0.2, 0.8, 128))
        all_colors = np.vstack((neg_colors, pos_colors))
        cmap = LinearSegmentedColormap.from_list('RedGreen', all_colors)
        abs_max = max(abs(vmin), abs(vmax))
        norm = Normalize(vmin=-abs_max, vmax=abs_max)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.pcolormesh(spots, vols, option_pnl, cmap=cmap, shading='auto', norm=norm)
    
    cbar = fig.colorbar(im)
    cbar.set_label('PnL')

    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Volatility')
    
    ax.set_xticks(spots)
    ax.set_yticks(vols)
    
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    
    x, y = np.meshgrid(spots, vols)
    for x_val, y_val, val in zip(x.flatten(), y.flatten(), option_pnl.flatten()):
        ax.annotate(f'{val:.2f}', (x_val, y_val), ha='center', va='center')
    
    return fig