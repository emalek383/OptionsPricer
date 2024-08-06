""" Setup the streamlit displays. """

import streamlit as st

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
    
    
    # Option style data
    option_style = {
        'Option Features': state.option_params['type'],
        'Exercise Type': 'American' if state.option_params['american'] else 'European'
    }
    
    option_style_columns_order = ['Option Features', 'Exercise Type']
    if state.option_params['type'] == 'Barrier':
        option_style['Option Features'] = f"Barrier ({state.option_params['barrier_type']})"
    
    # Option parameters data
    option_params = {
        'Spot Price': state.option_params['spot'],
        'Strike Price': state.option_params['strike'],
        'Volatility': state.option_params['vol'],
        'Time to Expiry': f"{state.option_params['time']} years",
        'Risk Free Rate': state.option_params['risk_free_rate'],
    }
    
    params_columns_order = ['Spot Price', 'Strike Price', 'Volatility', 'Time to Expiry', 'Risk Free Rate']
    
    if state.option_params['type'] == 'Barrier':
        option_params['Barrier'] = state.option_params['barrier']
        params_columns_order.append('Barrier')
    
    if state.option_params['asset']:
        option_params['Asset'] = state.option_params['asset']
        params_columns_order.insert(0, 'Asset')
        
    subclass = 'narrow-card'
    if not state.is_session_pc:
        mapping = {'Spot Price': 'Spot', 'Strike Price': 'Strike', 'Volatility': 'Vol', 'Time to Expiry': 'T', 'Risk Free Rate': 'R_F'}
        for key, value in option_params.items():
            if key not in mapping:
                mapping[key] = key
                
        option_params = {mapping.get(key, key): value for key, value in option_params.items()}
        params_columns_order = [mapping.get(col, col) for col in params_columns_order]
        option_params['T'] = option_params['T'].replace('years', '')
        subclass = ''
    
    # Formatting functions
    def format_percentage(value):
        return f'{value:.2%}'
    
    def format_float(value):
        return f'{value:.2f}'
    
    # Apply formatting
    for key, value in option_params.items():
        if key in ['Volatility', 'Risk Free Rate']:
            option_params[key] = format_percentage(value)
        elif isinstance(value, (float, int)):
            option_params[key] = format_float(value)
    
    style_card = create_card("Option Style", option_style, option_style_columns_order, subclass = subclass)
    params_card = create_card("Option Parameters", option_params, params_columns_order)
    display.markdown(
        f"""
            <div class='card-container'>
                {style_card}
                {params_card}
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