""" Module for processing the streamlit forms and computing the relevant options prices."""

import numpy as np
import streamlit as st 

from data_loader import download_data
from Options import UnderlyingAsset
from Options import OptionFactory, BlackScholesModel, BinomialTreeModel, MonteCarloModel

DEFAULT_SPOT = 100
DEFAULT_STRIKE = 100
DEFAULT_TIME = 1
DEFAULT_VOL = 0.2
DEFAULT_RISK_FREE_RATE = 0.05
DEFAULT_MIN_SPOT = 80
DEFAULT_MAX_SPOT = 120
DEFAULT_MIN_VOL = 0.01
DEFAULT_MAX_VOL = 1.0
DEFAULT_TYPE = 'Vanilla'

state = st.session_state

@st.cache_data
def cached_calculate_option_prices(option_params, method = 'bs'):
    """
    Cache and calculate option prices using Black-Scholes and Binomial Tree models.

    Parameters:
    -----------
    option_params : dict
        Dictionary containing option parameters with the following keys:
        - 'spot': float or np.array, Spot price(s) of the underlying asset
        - 'strike': float, Strike price of the option
        - 'time': float, Time to maturity in years
        - 'vol': float or np.array, Volatility of the underlying asset
        - 'risk_free_rate': float, Risk-free interest rate
        - 'call': bool, True for a call option, False for a put option
        - 'american': bool, True for an American option, False for a European option
        - 'type': str, Type of option ('vanilla', 'digital', or 'barrier')
        - 'barrier': float, Barrier level for barrier options (optional)
        - 'barrier_type': str, Type of barrier option (optional)
    method : str, optional
        Pricing method to use ('bs' for Black-Scholes or 'tree' for Binomial Tree).
    num_steps : int, optional
        Number of steps for the Binomial Tree model. Default is 1000.

    Returns:
    --------
    dict
        A dictionary containing the Black-Scholes and Binomial Tree prices.
        
    """
    
    return calculate_option_prices(option_params, method = method)

@st.cache_data
def cached_calculate_option_pnl(option_params, heatmap_params, method = 'bs'):
    """
    Cache and calculate the PnL of an option for a range of spot prices and volatilities.

    Parameters
    ----------
    option_params : dict
        The parameters of the option.
    heatmap_params : dict
        Min vol, max vol, min spot and max spot for the heatmaps.
    method : str, optional
        Pricing method to use ('bs' for Black-Scholes or 'tree' for Binomial Tree).
        Default is 'bs'.
    num_steps : int, optional
        Number of steps for the Binomial Tree model. Default is 1000.

    Returns
    -------
    numpy.ndarray
        2D array of PnL values.
    numpy.ndarray
        1D array of spot prices.
    numpy.ndarray
        1D array of volatilities.
        
    """
    
    return calculate_option_pnl(option_params, heatmap_params, method = method)

def calculate_option_prices(option_params, method = 'bs'):
    """
    Calculate option prices using Black-Scholes and Binomial Tree models.

    Parameters:
    -----------
    option_params : dict
        Dictionary containing option parameters with the following keys:
        - 'spot': float or np.array, Spot price(s) of the underlying asset
        - 'strike': float, Strike price of the option
        - 'time': float, Time to maturity in years
        - 'vol': float or np.array, Volatility of the underlying asset
        - 'risk_free_rate': float, Risk-free interest rate
        - 'call': bool, True for a call option, False for a put option
        - 'american': bool, True for an American option, False for a European option
        - 'type': str, Type of option ('vanilla', 'digital', or 'barrier')
        - 'barrier': float, Barrier level for barrier options (optional)
        - 'barrier_type': str, Type of barrier option (optional)
    method : str, optional
        Pricing method to use ('bs' for Black-Scholes or 'tree' for Binomial Tree).
    num_steps : int, optional
        Number of steps for the Binomial Tree model. Default is 1000.

    Returns:
    --------
    dict
        A dictionary containing the Black-Scholes and Binomial Tree prices.
        
    """

    # Create the underlying asset and option objects
    S = option_params['spot']
    K = option_params['strike']
    T = option_params['time']
    vol = option_params['vol']
    r = option_params['risk_free_rate']
    call = option_params['call']
    american = option_params['american']
    option_type = option_params['type'].lower()
    
    underlying = UnderlyingAsset(S, vol, r)
    
    if option_type =='barrier':
        barrier = option_params['barrier']
        barrier_type = option_params['barrier_type'].lower()
        option = OptionFactory.create_option(option_type, underlying, K, T, barrier, barrier_type, call, american)
    elif option_type == 'asian':
        averaging_type = option_params['averaging_type'].lower()
        averaging_freq = option_params['averaging_freq'].lower()
        option = OptionFactory.create_option(option_type, underlying, K, T, averaging_type = averaging_type, averaging_freq = averaging_freq, call = call, american = american)
    else:
        option = OptionFactory.create_option(option_type, underlying, K, T, call, american)
                
    if method == 'bs':
        return BlackScholesModel.price(option) # supports vectorised computation on vectorised inputs
    elif method == 'mc': # Also supports vectorised computation
        model = MonteCarloModel(num_simulations = option_params['num_simulations'], variance_reduction = option_params['variance_reduction'])
        return model.price(option)
    else: # method == 'tree' # Does not support vectorised computation
        model = BinomialTreeModel(num_steps = option_params['num_steps'])
        if isinstance(S, (np.ndarray, list)): # uses loops for tree computation which does not support vectorised input
            return np.array([model.price(OptionFactory.create_option(option_type, UnderlyingAsset(s, v, r), K, T, call, american))
                             for s, v in zip(S, vol)])
        
        else:
            return model.price(option)

def calculate_option_pnl(option_params, heatmap_params, method = 'bs'):
    """
    Calculate the PnL of an option for a range of spot prices and volatilities.

    Parameters
    ----------
    option_params : dict
        The parameters of the option.
    heatmap_params : dict
        Min vol, max vol, min spot and max spot for the heatmaps.
    method : str, optional
        Pricing method to use ('bs' for Black-Scholes or 'tree' for Binomial Tree).
        Default is 'bs'.
    num_steps : int, optional
        Number of steps for the Binomial Tree model. Default is 1000.

    Returns
    -------
    numpy.ndarray
        2D array of PnL values.
    numpy.ndarray
        1D array of spot prices.
    numpy.ndarray
        1D array of volatilities.
        
    """
    
    K = option_params['strike']
    T = option_params['time']
    r = option_params['risk_free_rate']
    call = option_params['call']
    american = option_params['american']
    option_type = option_params['type'].lower()
    current_price = option_params['price']

    kwargs = {}
    if option_type == 'barrier':
        kwargs['barrier'] = option_params['barrier']
        kwargs['barrier_type'] = option_params['barrier_type'].lower()
    elif option_type == 'asian':
        kwargs['averaging_type'] = option_params['averaging_type'].lower()
        kwargs['averaging_freq'] = option_params['averaging_freq'].lower()

    vols = np.linspace(heatmap_params['min_vol'], heatmap_params['max_vol'], 10)
    spots = np.linspace(heatmap_params['min_spot'], heatmap_params['max_spot'], 10)
    
    spot_grid, vol_grid = np.meshgrid(spots, vols)
    
    underlying_grid = UnderlyingAsset(spot_grid.flatten(), vol_grid.flatten(), r)

    if method == 'bs' or method == 'mc':
        # Vectorized calculation for Black-Scholes and Monte Carlo
        if method == 'bs':
            model = BlackScholesModel()
        else:
            model = MonteCarloModel(num_simulations = min(option_params['num_simulations'], 2_000), variance_reduction = option_params['variance_reduction'])
            
        if option_type == 'barrier':
            option = OptionFactory.create_option(option_type, underlying_grid, K, T,
                                                 kwargs['barrier'], kwargs['barrier_type'], call, american)
        elif option_type == 'asian':
            option = OptionFactory.create_option(option_type, underlying_grid, K, T,
                                                 averaging_type = kwargs['averaging_type'], averaging_freq = kwargs['averaging_freq'], call = call, american = american)
        else:
            option = OptionFactory.create_option(option_type, underlying_grid, K, T, call, american)
        
        option_prices = model.price(option)
        option_pnl = option_prices - current_price
        option_pnl = option_pnl.reshape(spot_grid.shape)
        
    else:  # method == 'tree'
        # Iterative calculation for Binomial Tree
        model = BinomialTreeModel(num_steps = option_params['num_steps'])
        option_pnl = np.zeros_like(spot_grid)
        for i in range(10):
            for j in range(10):
                if option_type == 'barrier':
                    option = OptionFactory.create_option(option_type, UnderlyingAsset(spot_grid[i, j], vol_grid[i, j], r) , K, T, 
                                                         kwargs['barrier'], kwargs['barrier_type'], call, american)
                else:
                    option = OptionFactory.create_option(option_type, UnderlyingAsset(spot_grid[i, j], vol_grid[i, j], r), K, T, call, american)
                option_pnl[i, j] = model.price(option) - current_price

    return option_pnl, spots, vols

def process_options_form(asset = None, 
                         spot = DEFAULT_SPOT,
                         strike = DEFAULT_STRIKE,
                         time_to_expiry = DEFAULT_TIME,
                         vol = DEFAULT_VOL,
                         risk_free_rate = DEFAULT_RISK_FREE_RATE,
                         american = False,
                         option_type = DEFAULT_TYPE,
                         method = 'bs',
                         **kwargs):
    """
    Process the options parameter form.

    Parameters
    ----------
    asset : str, optional
        Ticker of underlying asset. If not None, will override the spot price entered. The default is None.
    spot : float, optional
        Spot price. If a non-trivial ticker is entered, will be ignored. The default is DEFAULT_SPOT.
    strike : float, optional
        Strike price. The default is DEFAULT_STRIKE.
    time_to_expiry : float, optional
        Time to expiry in years. The default is DEFAULT_TIME.
    vol : float, optional
        Volatility. The default is DEFAULT_VOL.
    risk_free_rate : float, optional
        Risk free rate. The default is DEFAULT_RISK_FREE_RATE.
    american : bool, optional
        True if American exercise type, False if European. The default is False.
    option_type : str, optional
        Denotess the features of the option (Vanilla, Digital, Barrier). The default is DEFAULT_TYPE.
    barrier_type : str, optional
        Type of barrier option ('Up-and-out', 'Down-and-out', 'Up-and-in', 'Down-and-in'). The default is None.
    barrier : float, optional
        The barrier price. The default is None.
    method : str, optional
        Computational method to be used. The default is 'bs'.

    Returns
    -------
    error : str
        Error messages for the user.

    """
    
    error = ""
    
    if asset:
        asset = asset.upper()
    state.option_params['asset'] = asset
    if not asset:
        state.option_params['spot'] = spot
    else:
        state.option_params['spot'] = download_data(asset)
        if not state.option_params['spot']:
            state.option_params['spot'] = spot
            state.option_params['asset'] = None
            error = f"Could not download {asset}. Using manual spot price input."
    
    if american:
        method = 'tree'
    
    if option_type.lower() == 'asian' and kwargs['averaging_type'].lower() == 'arithmetic':
        method = 'mc'
        
    if option_type.lower() == 'asian' and kwargs['averaging_type'].lower() == 'geometric':
        if method == 'tree':
            method = 'bs'
    
    state.option_params['method'] = method
    state.option_params['num_steps'] = kwargs.get('num_steps') if method == 'tree' else None
    state.option_params['num_simulations'] = kwargs.get('num_simulations') if method == 'mc' else None
    state.option_params['variance_reduction'] = kwargs.get('variance_reduction') if method == 'mc' else None
    state.option_params['american'] = american
    state.option_params['type'] = option_type
    state.option_params['barrier_type'] = kwargs.get('barrier_type') if option_type.lower() == 'barrier' else None
    state.option_params['barrier'] = kwargs.get('barrier') if option_type.lower() == 'barrier' else None
    state.option_params['averaging_type'] = kwargs.get('averaging_type') if option_type.lower() == 'asian' else None
    state.option_params['averaging_freq'] = kwargs.get('averaging_freq') if option_type.lower() == 'asian' else None
    state.option_params['strike'] = strike
    state.option_params['time'] = time_to_expiry
    state.option_params['vol'] = vol
    state.option_params['risk_free_rate'] = risk_free_rate
    
    call_params = state.option_params.copy()
    call_params['call'] = True
    
    if method == 'mc':
        state.option_params['call_value'] = calculate_option_prices(call_params, method = state.option_params['method'])
    else:
        state.option_params['call_value'] = cached_calculate_option_prices(call_params, method = state.option_params['method'])
    
    put_params = state.option_params.copy()
    put_params['call'] = False
    
    if method == 'mc':
        state.option_params['put_value'] = calculate_option_prices(put_params, method = state.option_params['method'])
    else:
        state.option_params['put_value'] = cached_calculate_option_prices(put_params, method = state.option_params['method'])

    state.heatmap_params = {'min_spot': max(state.option_params['spot'] - 20.00, 0.00),
                            'max_spot': state.option_params['spot'] + 20.00,
                            'min_vol': max(state.option_params['vol'] - 0.20, 0.01),
                            'max_vol': 1}
    
    return error

def process_heatmap_form(min_spot = DEFAULT_MIN_SPOT,
                         max_spot = DEFAULT_MAX_SPOT,
                         min_vol = DEFAULT_MIN_VOL,
                         max_vol = DEFAULT_MAX_VOL):
    """
    Process the heatmap form by saving the parameters in state.

    Parameters
    ----------
    min_spot, max_spot : float, optional
        The min/max of the spot price for the heatmaps. The defaults are DEFAULT_MIN_SPOT, DEFAULT_MAX_SPOT.
    min_vol, max_vol : float, optional
        The min/max of the volatility for the heatmaps. The defaults are DEFAULT_MIN_VOL, DEFAULT_MAX_VOL.
    
    Returns
    -------
    None.

    """
    
    state.heatmap_params = {'min_spot': min_spot, 'max_spot': max_spot, 'min_vol': min_vol, 'max_vol': max_vol}
    
    call_params = state.option_params.copy()
    call_params['price'] = state.option_params['call_value']
    call_params['call'] = True
    put_params = state.option_params.copy()
    put_params['price'] = state.option_params['put_value']
    put_params['call'] = False
    
    if state.option_params['method'] == 'mc':
        state['call_option_pnl'], state['heatmap_spots'], state['heatmap_vols'] = calculate_option_pnl(call_params, state.heatmap_params, state.option_params['method'])
        state['put_option_pnl'], state['heatmap_spots'], state['heatmap_vols'] = calculate_option_pnl(put_params, state.heatmap_params, state.option_params['method'])
    else:
        state['call_option_pnl'], state['heatmap_spots'], state['heatmap_vols'] = cached_calculate_option_pnl(call_params, state.heatmap_params, state.option_params['method'])
        state['put_option_pnl'], state['heatmap_spots'], state['heatmap_vols'] = cached_calculate_option_pnl(put_params, state.heatmap_params, state.option_params['method'])