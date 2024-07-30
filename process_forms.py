from options_pricing import calculate_price
from data_loader import download_data
import streamlit as st 

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
def process_options_form(asset = None, 
                         spot = DEFAULT_SPOT,
                         strike = DEFAULT_STRIKE,
                         time_to_maturity = DEFAULT_TIME,
                         vol = DEFAULT_VOL,
                         risk_free_rate = DEFAULT_RISK_FREE_RATE,
                         american = False,
                         option_type = DEFAULT_TYPE,
                         barrier_type = None,
                         barrier = None,
                         method = 'bs'):
    
    error = ""
    
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
    
    state.option_params['method'] = method
    state.option_params['american'] = american
    state.option_params['type'] = option_type
    state.option_params['barrier_type'] = barrier_type if option_type.lower() == 'barrier' else None
    state.option_params['barrier'] = barrier if option_type.lower() == 'barrier' else None
    state.option_params['strike'] = strike
    state.option_params['time'] = time_to_maturity
    state.option_params['vol'] = vol
    state.option_params['risk_free_rate'] = risk_free_rate
    
    # Additional keyword arguments for barrier options
    kwargs = {}
    if option_type.lower() == 'barrier':
        kwargs['barrier'] = barrier
        kwargs['barrier_type'] = barrier_type.lower()
    
    state.option_params['call_value'] = calculate_price(
        state.option_params['spot'], 
        strike,
        time_to_maturity, 
        vol, 
        risk_free_rate, 
        call = True, 
        american = american, 
        option_type = option_type.lower(),
        method = method,
        **kwargs
    )
    
    state.option_params['put_value'] = calculate_price(
        state.option_params['spot'], 
        strike, 
        time_to_maturity, 
        vol, 
        risk_free_rate, 
        call = False, 
        american = american, 
        option_type = option_type.lower(),
        method = method,
        **kwargs
    )

    state.heatmap_params = {'min_spot': max(state.option_params['spot'] - 20.00, 0.00),
                            'max_spot': state.option_params['spot'] + 20.00,
                            'min_vol': max(state.option_params['vol'] - 0.20, 0.01),
                            'max_vol': 1}
    
    return error

def process_heatmap_form(min_spot = DEFAULT_MIN_SPOT,
                         max_spot = DEFAULT_MAX_SPOT,
                         min_vol = DEFAULT_MIN_VOL,
                         max_vol = DEFAULT_MAX_VOL):
    
    state.heatmap_params = {'min_spot': min_spot, 'max_spot': max_spot, 'min_vol': min_vol, 'max_vol': max_vol}
    
    return