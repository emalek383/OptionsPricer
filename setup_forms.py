from process_forms import process_options_form, process_heatmap_form
from data_loader import download_data
import streamlit as st

state = st.session_state

def setup_options_form(form):
    asset = form.text_input("Optional: Asset (ticker)",
                            value = None,
                            help = "Optional: enter a ticker to automatically get the spot price.")
    
    default_value = None
    if asset:
        default_value = download_data(asset)
    if not default_value:
        default_value = 100.00
    
    spot = form.number_input("Spot Price",
                             value = default_value,
                             min_value = 0.00,
                             help = "Will be ignored if ticker entered.")
    
    strike = form.number_input("Strike Price",
                               value = 100.00,
                               min_value = 0.00)
    
    time_to_maturity = form.number_input("Time to Maturity (Years)",
                                         1.00)
    
    vol = form.number_input("Volatility (%)",
                            value = 20.00,
                            min_value = 0.00)
    
    risk_free_rate = form.number_input("Risk-free Rate (%)",
                                       value = 5.00,
                                       min_value = 0.00)
    
    exercise_type = form.radio("Exercise Type",
                               options = ["European", "American"])
    
    if exercise_type == 'American':
        american = True
    else:
        american = False
    
    option_type = form.selectbox("Options Type",
                                  options = ["Vanilla", "Digital", "Barrier"])
    
    barrier_type = None
    barrier = None
    if option_type == "Barrier":
        if american:
            barrier_type = form.selectbox("Barrier Type", ["Up-and-out", "Down-and-out"])
        else:
            barrier_type = form.selectbox("Barrier Type", ["Up-and-in", "Up-and-out", "Down-and-in", "Down-and-out"])
        barrier = form.number_input("Barrier", value = 100.00)
        
    def format_method(option):
        option_map = {'bs': 'Black-Scholes', 'tree': 'Binomial Tree'}
        return option_map[option]
        
    method = 'tree'
    method = form.radio("Calculation Method",
                        options = ['bs', 'tree'],
                        format_func = format_method,
                        disabled = True if exercise_type == "American" else False)
    
    vol /= 100
    risk_free_rate /= 100
    
    if asset:
        asset = asset.strip()
    
    submit_button = form.button(label = "Calculate")
    
    if submit_button:
        error = process_options_form(asset, spot, strike, time_to_maturity, vol, risk_free_rate, american, option_type, barrier_type, barrier, method)
        if error:
            form.error(error)
    return

def setup_heatmap_form(form):
    min_spot = form.number_input("Min Spot Price",
                                 min_value = 0.00,
                                 max_value = float(state.option_params['spot']),
                                 value = max(state.option_params['spot'] - 20.00, 0.00))
    max_spot = form.number_input("Max Spot Price",
                                 min_value = float(state.option_params['spot']),
                                 value = state.option_params['spot'] + 20.00)
    
    min_vol, max_vol = form.slider("Volatility Range (%)",
                                   value = [1, 100],
                                   min_value = 1,
                                   max_value = 100)
    
    min_vol_decimal = min_vol / 100
    max_vol_decimal = max_vol / 100
    
    process_heatmap_form(min_spot, max_spot, min_vol_decimal, max_vol_decimal)
    
    # form.button(label = "Compute Heatmap", on_click = process_heatmap_form, args = (min_spot, max_spot, min_vol / 100, max_vol / 100))
    
    return