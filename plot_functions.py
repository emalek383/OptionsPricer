
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from options_pricing import calculate_price

def plot_heatmap(option_params, heatmap_params, call = True):
    strike = option_params['strike']
    time_to_maturity = option_params['time']
    risk_free_rate = option_params['risk_free_rate']
    american = option_params['american']
    option_type = option_params['type'].lower()
    method = option_params['method']
    # Additional keyword arguments for barrier options
    kwargs = {}
    if option_params['type'].lower() == 'barrier':
        kwargs['barrier'] = option_params['barrier']
        kwargs['barrier_type'] = option_params['barrier_type'].lower()
    
    if call:
        price = option_params['call_value']
    else:
        price = option_params['put_value']
        
    vols = np.linspace(heatmap_params['min_vol'], heatmap_params['max_vol'], 10)
    spots = np.linspace(heatmap_params['min_spot'], heatmap_params['max_spot'], 10)
    
    # Create meshgrid for vectorized calculation
    spot_grid, vol_grid = np.meshgrid(spots, vols)
    
    if option_params['method'].lower() == 'bs':
        # vectorised calculation of prices for BS formula
        option_pnl = calculate_price(
            spot_grid, 
            strike, 
            time_to_maturity, 
            vol_grid, 
            risk_free_rate, 
            call, 
            american, 
            option_type,
            method,
            **kwargs
        ) - price
        
        #calculate_price(spot_grid, strike, time_to_maturity, vol_grid, risk_free_rate, call, american, option_type) - price
    
    else:
        # iterative calculation of prices for binomial trees
        option_pnl = np.zeros_like(spot_grid)
    
        for i in range(10):
            for j in range(10):
                option_pnl[i, j] = calculate_price(
                    spot_grid[i, j], 
                    strike, 
                    time_to_maturity, 
                    vol_grid[i, j], 
                    risk_free_rate, 
                    call, 
                    american, 
                    option_type,
                    method,
                    **kwargs
                ) - price
                
                #calculate_price(spot_grid[i, j], strike, time_to_maturity, vol_grid[i, j], risk_free_rate, call, american, option_type) - price
    
    # Determine the range of values
    vmin = option_pnl.min()
    vmax = option_pnl.max()
    
    # Create custom colormap based on the range of values
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
        # Use vmin and vmax symmetrically centered around 0
        abs_max = max(abs(vmin), abs(vmax))
        norm = Normalize(vmin=-abs_max, vmax=abs_max)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.pcolormesh(spots, vols, option_pnl, cmap=cmap, shading='auto', norm = norm)
    
    # Add colorbar
    cbar = fig.colorbar(im)
    cbar.set_label('PnL')

    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Volatility')
    
    # Set ticks
    ax.set_xticks(spots)
    ax.set_yticks(vols)
    
    # Use a formatter to display fewer decimal places
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    
    x, y = np.meshgrid(spots, vols)
    for x_val, y_val, val in zip(x.flatten(), y.flatten(), option_pnl.flatten()):
        ax.annotate(f'{val:.2f}', (x_val, y_val), ha='center', va='center')
    
    return fig