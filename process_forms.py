""" Module for processing the streamlit forms and computing the relevant options prices."""

import numpy as np
import streamlit as st 

from data_loader import download_data
from Options import OptionFactory, BlackScholesModel, BinomialTreeModel

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
def calculate_option_prices(option_params, method = 'bs', num_steps = 1_000):
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
    
    # Create the option object
    S = option_params['spot']
    K = option_params['strike']
    T = option_params['time']
    vol = option_params['vol']
    r = option_params['risk_free_rate']
    call = option_params['call']
    american = option_params['american']
    option_type = option_params['type'].lower()
    
    if option_type =='barrier':
        barrier = option_params['barrier']
        barrier_type = option_params['barrier_type'].lower()
        option = OptionFactory.create_option(option_type, S, K, T, vol, r, barrier, barrier_type, call, american)
    else:
        option = OptionFactory.create_option(option_type, S, K, T, vol, r, call, american)
        
    if method == 'bs':
        return BlackScholesModel.price(option) # supports vectorised computation on vectorised inputs
    else: # method == 'tree'
        model = BinomialTreeModel(num_steps = num_steps)
        if isinstance(S, (np.ndarray, list)): # uses loops for tree computation which does not support vectorised input
            return np.array([model.price(OptionFactory.create_option(option_type, s, K, T, v, r, call, american))
                             for s, v in zip(S, vol)])
        
        else:
            return model.price(option)

@st.cache_data
def calculate_option_pnl(option_params, heatmap_params, method = 'bs', num_steps = 1_000):
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

    vols = np.linspace(heatmap_params['min_vol'], heatmap_params['max_vol'], 10)
    spots = np.linspace(heatmap_params['min_spot'], heatmap_params['max_spot'], 10)
    
    spot_grid, vol_grid = np.meshgrid(spots, vols)

    if method == 'bs':
        # Vectorized calculation for Black-Scholes
        model = BlackScholesModel()
        if option_type == 'barrier':
            option = OptionFactory.create_option(option_type, spot_grid, K, T, vol_grid, r, 
                                                 kwargs['barrier'], kwargs['barrier_type'], call, american)
        else:
            option = OptionFactory.create_option(option_type, spot_grid, K, T, vol_grid, r, call, american)
        option_prices = model.price(option)
        option_pnl = option_prices - current_price
    else:  # method == 'tree'
        # Iterative calculation for Binomial Tree
        model = BinomialTreeModel(num_steps=1000)  # You can adjust the number of steps
        option_pnl = np.zeros_like(spot_grid)
        for i in range(10):
            for j in range(10):
                if option_type == 'barrier':
                    option = OptionFactory.create_option(option_type, spot_grid[i, j], K, T, vol_grid[i, j], r, 
                                                         kwargs['barrier'], kwargs['barrier_type'], call, american)
                else:
                    option = OptionFactory.create_option(option_type, spot_grid[i, j], K, T, vol_grid[i, j], r, call, american)
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
                         barrier_type = None,
                         barrier = None,
                         method = 'bs'):
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
    
    state.option_params['asset'] = asset.upper()
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
    
    state.option_params['method'] = method
    state.option_params['american'] = american
    state.option_params['type'] = option_type
    state.option_params['barrier_type'] = barrier_type if option_type.lower() == 'barrier' else None
    state.option_params['barrier'] = barrier if option_type.lower() == 'barrier' else None
    state.option_params['strike'] = strike
    state.option_params['time'] = time_to_expiry
    state.option_params['vol'] = vol
    state.option_params['risk_free_rate'] = risk_free_rate
    
    # Additional keyword arguments for barrier options
    kwargs = {}
    if option_type.lower() == 'barrier':
        kwargs['barrier'] = barrier
        kwargs['barrier_type'] = barrier_type.lower()
    
    call_params = state.option_params.copy()
    call_params['call'] = True
    
    state.option_params['call_value'] = calculate_option_prices(call_params, method = state.option_params['method'])
    
    put_params = state.option_params.copy()
    put_params['call'] = False
    
    state.option_params['put_value'] = calculate_option_prices(put_params, method = state.option_params['method'])

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
    
    state['call_option_pnl'], state['heatmap_spots'], state['heatmap_vols'] = calculate_option_pnl(call_params, state.heatmap_params, state.option_params['method'])
    state['put_option_pnl'], state['heatmap_spots'], state['heatmap_vols'] = calculate_option_pnl(put_params, state.heatmap_params, state.option_params['method'])