import pandas as pd
import streamlit as st
from plot_functions import plot_heatmap
#from load_css import local_css
#local_css("style.css")

state = st.session_state

def setup_price_display(display):
    display.header("Options Pricing Tool")
        
    # Display parameters
    options_params = {'Spot Price': [state.option_params['spot']],
                      'Strike Price': [state.option_params['strike']],
                      'Volatility': [state.option_params['vol']],
                      'Time to Maturity': [f"{state.option_params['time']} years"],
                      'Risk Free Rate': [state.option_params['risk_free_rate']],
                      'Option Type': [state.option_params['type']],
                      'Exercise Type': ['American' if state.option_params['american'] else 'European']}
    columns = ['Spot Price', 'Strike Price', 'Volatility', 'Time to Maturity', 'Risk Free Rate', 'Exercise Type', 'Option Type']
    if state.option_params['type'] == 'Barrier':
        options_params['Option Type'] = f"Barrier ({state.option_params['barrier_type']})"
        options_params['Barrier'] = state.option_params['barrier']
        columns = columns + ['Barrier']
        
    if state.option_params['asset']:
        options_params['Asset'] = [state.option_params['asset']]
        columns = ['Asset'] + columns
        
    parameters_df = pd.DataFrame(options_params, columns = columns)
    
    # Format the DataFrame
    def format_percentage(value):
        return f'{value:.2%}'

    def format_float(value):
        return f'{value:.2f}'

    formatted_df = parameters_df.copy()

    # Apply formatting
    formatted_df['Volatility'] = formatted_df['Volatility'].apply(format_percentage)
    formatted_df['Risk Free Rate'] = formatted_df['Risk Free Rate'].apply(format_percentage)

    # Format other numeric columns to 2 decimal places
    numeric_columns = formatted_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        if col not in ['Volatility', 'Risk-Free-Rate']:
            formatted_df[col] = formatted_df[col].apply(format_float)
        
    html_table = formatted_df.to_html(index=False)
    
    # Define custom CSS
    css = """
    <style>
        .header {
        color: #1E90FF;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f0f7ff;
            color: black;
            font-weight: bold;
        }
    </style>
    """
    display.markdown(css, unsafe_allow_html=True)
    display.markdown("### Options Parameters ###")
    display.markdown(html_table, unsafe_allow_html = True)
    
    css_values = """
    <style>
        .computed-values {
            margin-top: -15px;
            padding: 10px 15px;
            }
        .value-container {
            display: flex;
            justify-content: space-around;
            margin-top: 10px;
            }
        .value-box {
            text-align: center;
            padding: 5px 15px;
            background-color: #f0f7ff;
            border-radius: 5px;
            width: 48%;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
        .value-label {
            font-weight: bold;
            font-size: 20px;
            }
        .value {
            font-weight: bold;
            font-size: 28px;
            colour: #333;
            }
        </style>
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

    # Combine the CSS and HTML
    styled_values = f"{css_values}{computed_values_html}"

    # Display the computed values
    display.markdown(styled_values, unsafe_allow_html=True)
    return

def setup_heatmap_display(display):
    display.header("PnL Heatmaps for varying spot price and volatility")
    
    display.write("See the PnL of the Put and Call options for varying spot price and volatility.")
    
    call_col, put_col = display.columns(2)
    
    # Custom CSS for centering the titles
    centered_title_css = """
    <style>
        .centered-title {
            text-align: center;
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 10px;
            }
        </style>
        """

    # Inject the CSS
    st.markdown(centered_title_css, unsafe_allow_html=True)

    # Call PnL
    call_col.markdown("<div class='centered-title'>Call PnL</div>", unsafe_allow_html=True)
    call_fig = plot_heatmap(state.option_params, state.heatmap_params, call = True)
    call_col.pyplot(call_fig)
    
    # Put PnL
    put_col.markdown("<div class='centered-title'>Put PnL</div>", unsafe_allow_html=True)
    put_fig = plot_heatmap(state.option_params, state.heatmap_params, call = False)
    put_col.pyplot(put_fig)
    
    return
