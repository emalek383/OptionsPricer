""" Setup the streamlit displays. """

import streamlit as st

from setup_forms import setup_heatmap_form
from plot_functions import plot_heatmap

state = st.session_state


def setup_options_params_display(display):
    """
    Setup the display of the chosen options parameters. Display these in two 'cards: one for Option Style (Features: Vanilla / Exotic type, Exercise Type: European / American);
    one for the option parmaeters (Spot, Strike, Vol, Time to Expiry, Risk Free Rate, Barrier if applicable).
    Make these cards friendly for mobile devices.

    Parameters
    ----------
    display : st.container
        Streamlit container where output will be displayed.

    Returns
    -------
    None.

    """
    
    basic_params = {
        'Type': state.option_params['type'],
        'Exercise': 'American' if state.option_params['american'] else 'European',
        'Strike': state.option_params['strike'],
        'Time to Expiry': f"{state.option_params['time']} years",
    }
    
    advanced_params = {}
    if state.option_params['type'] == 'Barrier':
        advanced_params['Barrier Type'] = state.option_params['barrier_type']
        advanced_params['Barrier'] = state.option_params['barrier']
    elif state.option_params['type'] == 'Asian':
        advanced_params['Averaging'] = state.option_params['averaging_type']
        advanced_params['Frequency'] = state.option_params['averaging_freq']

    asset_params = {'Spot Price': state.option_params['spot'],
                    'Volatility': state.option_params['vol'],
                    'Risk-free Rate': state.option_params['risk_free_rate'],
                    }
    
    if state.option_params['asset']:
        asset_params['Asset'] = state.option_params['asset']
    
    if not state.is_session_pc:
        mobile_mapping  = {'Spot Price': 'Spot', 'Strike Price': 'Strike', 'Volatility': 'Vol', 'Time to Expiry': 'T',
                   'Exercise Type': 'Exercise'}
        
        basic_params = {mobile_mapping.get(k, k): v for k, v in basic_params.items()}
        advanced_params = {mobile_mapping.get(k, k): v for k, v in advanced_params.items()}
        asset_params = {mobile_mapping.get(k, k): v for k, v in asset_params.items()}
        basic_params['T'] = basic_params['T'].replace('years', '')
    
    # Formatting functions
    def format_percentage(value):
        return f'{value:.2%}'
    
    def format_float(value):
        return f'{value:.2f}'
    

    for params in [basic_params, advanced_params, asset_params]:
        for key, value in params.items():
            if key in ['Volatility', 'Vol', 'Risk-free Rate', 'R_F']:
                params[key] = format_percentage(value)
            elif isinstance(value, (float, int)):
                params[key] = format_float(value)
    
    # Create cards
    basic_card = create_card("Basic Option Parameters", basic_params, list(basic_params.keys()))
    asset_card = create_card("Asset Parameters", asset_params, list(asset_params.keys()))
    
    if advanced_params:
        advanced_card = create_card("Advanced Option Parameters", advanced_params, list(advanced_params.keys()))
        display.markdown(
            f"""
                <div class='card-container'>
                    {basic_card}
                    {advanced_card}
                    {asset_card}
                </div>
            """,
            unsafe_allow_html=True)
    else:
        display.markdown(
            f"""
                <div class='card-container'>
                    {basic_card}
                    {asset_card}
                </div>
            """,
            unsafe_allow_html=True)

def create_card(title, data, columns, subclass = ""):
    """
    Create a card for nicely displaying a table with a header.

    Parameters
    ----------
    title : str
        Title of the card.
    data : dict
        Dictionary containing the parameters to be displayed in the table.
            Key: Name of parameter (str)
            Value: Value of parameter
    columns : list(str)
        List containing the parameter names in order in which they will be displayed.
    subclass : str, optional
        Can be used to define a subclass for CSS styling (e.g. 'narrow' for a narrow table). The default is "".

    Returns
    -------
    markdown : str
        The markdown to be displayed.

    """
    rows = "".join([
        f"<tr>{''.join([f'<td>{data[col]}</td>' for col in columns])}</tr>"
    ])
    
    markdown = f"""<div class='card {subclass}'>
                    <div class='card-header'>
                        {title}
                    </div>
                    <div class = 'card-table-container'>
                        <table class='card-table'>
                            <tr>{''.join([f'<th>{col}</th>' for col in columns])}</tr>
                            {rows}
                        </table>
                    </div>
                </div>"""
    
    return markdown

def setup_options_price_display(display):
    """
    Setup the display of the Call of and Put option prices.

    Parameters
    ----------
    display : st.container
        Container where the output will be displayed.

    Returns
    -------
    None.

    """

    # HTML for the computed values
    display.markdown("### Computed Option Values")
    computed_values_html = f"""
    <div class="computed-values">
        <div class="value-container">
            <div class="value-box">
                <div class="value-label">Call Value</div>
                <div class="value">${state.option_params['call_value']:.2f}</div>
            </div>
            <div class="value-box">
                <div class="value-label">Put Value</div>
                <div class="value">${state.option_params['put_value']:.2f}</div>
            </div>
        </div>
    </div>
    """

    # Display the computed values
    display.markdown(computed_values_html, unsafe_allow_html=True)

def setup_heatmap_display(display):
    """
    Setup the display for the Call and Put heatmaps.

    Parameters
    ----------
    display : st.container
        Container where the output will be displayed.

    Returns
    -------
    None.

    """
    
    display.header("PnL Heatmaps for varying spot price and volatility")
    
    heatmap_parameters_form = display.expander("Adjust Heatmap Parameters", expanded = False)
    setup_heatmap_form(heatmap_parameters_form)
    
    display.write("See the PnL of the Put and Call options for varying spot price and volatility.")
    
    call_col, put_col = display.columns(2)
 
    # Call PnL
    call_col.markdown("<div class='centered-title'>Call PnL</div>", unsafe_allow_html=True)
    call_fig = plot_heatmap(state['call_option_pnl'], state['heatmap_spots'], state['heatmap_vols'])
    call_col.pyplot(call_fig)
    
    # Put PnL
    put_col.markdown("<div class='centered-title'>Put PnL</div>", unsafe_allow_html=True)
    put_fig = plot_heatmap(state['put_option_pnl'], state['heatmap_spots'], state['heatmap_vols'])
    put_col.pyplot(put_fig)