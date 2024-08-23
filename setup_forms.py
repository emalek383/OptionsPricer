""" Module for setting up the streamlit forms. """

import streamlit as st

from process_forms import process_options_form, process_heatmap_form
from data_loader import download_data

state = st.session_state

def setup_options_form(form):
    """
    Setup the form for entering the options parameters.

    Parameters
    ----------
    form : st.container
        Container that will be used as the form.

    Returns
    -------
    None.

    """

    basic_expander = form.expander("Basic Option Parameters", expanded = True)

    asset = basic_expander.text_input("Optional: Asset (ticker)",
                            value = None,
                            help = "Optional: enter a ticker to automatically get the spot price.")
    
    default_value = None
    if asset:
        asset = asset.upper()
        default_value = download_data(asset)
    if not default_value:
        default_value = 100.00
    
    spot = basic_expander.number_input("Spot Price",
                             value = default_value,
                             min_value = 0.00,
                             help = "Will be ignored if ticker entered.")
    
    strike = basic_expander.number_input("Strike Price",
                               value = 100.00,
                               min_value = 0.00)
    
    time_to_maturity = basic_expander.number_input("Time to Expiry (Years)",
                                         1.00)
    
    vol = basic_expander.number_input("Volatility (%)",
                            value = 20.00,
                            min_value = 0.00)
    
    risk_free_rate = basic_expander.number_input("Risk-free Rate (%)",
                                       value = 5.00,
                                       min_value = 0.00)
    
    option_type = basic_expander.selectbox("Options Type",
                                  options = ["Vanilla", "Digital", "Barrier", "Asian"])
    
    exercise_type = basic_expander.radio("Exercise Type",
                               options = ["European", "American"],
                               disabled = True if option_type == "Asian" else False)
    
    american = (exercise_type == 'American')
    
    advanced_options_expander = form.expander("Advanced Option Parameters", expanded = False)
    
    kwargs = {}
    
    method = 'tree'
    kwargs['num_steps'] = 1_000
    
    if option_type == "Barrier":
        if american:
            kwargs['barrier_type'] = advanced_options_expander.selectbox("Barrier Type", ["Up-and-out", "Down-and-out"])
        else:
            kwargs['barrier_type'] = advanced_options_expander.selectbox("Barrier Type", ["Up-and-in", "Up-and-out", "Down-and-in", "Down-and-out"])
        kwargs['barrier'] = advanced_options_expander.number_input("Barrier", value = 100.00)
        
    if option_type == "Asian":
        american = False
        method = 'mc'
        if 'num_simulations' not in kwargs:
            kwargs['num_simulations'] = 10_000
            kwargs['antithetic'] = True
        kwargs['averaging_type'] = advanced_options_expander.selectbox("Averaging Type", ["Arithmetic", "Geometric"])
        kwargs['averaging_freq'] = advanced_options_expander.selectbox("Averaging Frequency", ["Monthly", "Weekly", "Daily"])
     
    implementation_expander = form.expander("Implementation Parameters", expanded = True)   
     
    def format_method(option):
        option_map = {'bs': 'Black-Scholes', 'tree': 'Binomial Tree', 'mc': 'Monte Carlo'}
        return option_map[option]
        
    
    method_options = ['bs', 'tree', 'mc']
    
    if option_type == "Asian":
        if kwargs["averaging_type"] == "Geometric":
            method_options = ['bs', 'mc']
        else:
            method_options = ['mc']
    
    if exercise_type == "American":
        method_options = "tree"
        
    
    method = implementation_expander.radio("Calculation Method",
                        options = method_options,
                        format_func = format_method,
                        disabled = True if exercise_type == "American" or (option_type == "Asian" and kwargs["averaging_type"] != "Geometric") else False)
    
    if method == 'tree':
        kwargs['num_steps'] = implementation_expander.select_slider("Number of steps", options = (10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000), value = 1_000)
    elif method == 'mc':
        kwargs['num_simulations'] = implementation_expander.select_slider("Number of simulations", options = (100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000), value = 10_000)
        mc_form = implementation_expander.radio("Variance Reduction Method",
                             options = ["None", "Antithetic", "Control Variate"])
        
        var_red_format_map = {'None': None, 'Antithetic': 'antithetic', 'Control Variate': 'control_variate'}
        
        kwargs['variance_reduction'] = var_red_format_map[mc_form]
    
    vol /= 100
    risk_free_rate /= 100
    
    if asset:
        asset = asset.strip()
    
    submit_button = form.button(label = "Calculate")
    
    if submit_button:
        error = process_options_form(asset, spot, strike, time_to_maturity, vol, risk_free_rate, american, option_type, method, **kwargs)
        if error:
            form.error(error)

def setup_heatmap_form(form):
    """
    Setup the form for choosing the heatmap parameters (min/max vol and spot).

    Parameters
    ----------
    form : st.container
        Container that will house the form.

    Returns
    -------
    None.

    """
    
    col1, col2 = form.columns(2)#[0.25, 0.25, 0.5])
    
    with col1:
        st.markdown("##### Spot Price Range")
        current_spot = state.option_params['spot']
        min_spot, max_spot = st.slider(label = "Spot Price Range",
            min_value = 0.00,
            max_value = max(2 * float(current_spot), state.option_params['strike'] + 50.00),
            value = (max(0.00, current_spot - 20.00), current_spot + 20.00),
            step = 0.01,
            label_visibility = "collapsed"
        )

    with col2:
        st.markdown("##### Volatility Range (%)")
        min_vol, max_vol = st.slider(label = "Range (%)",
                                       value = [1, 100],
                                       min_value = 1,
                                       max_value = 100,
                                       label_visibility = "collapsed")
        
    min_vol_decimal = min_vol / 100
    max_vol_decimal = max_vol / 100
    
    process_heatmap_form(min_spot, max_spot, min_vol_decimal, max_vol_decimal)
    
    # with col3:
    #     if state.is_session_pc:
    #         st.markdown("&nbsp;")
    
    #     if st.button("Update Heatmaps", use_container_width = True):
    #         process_heatmap_form(min_spot, max_spot, min_vol_decimal, max_vol_decimal)
    
    