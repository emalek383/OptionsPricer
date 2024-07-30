import streamlit as st
from setup_forms import setup_options_form, setup_heatmap_form
from setup_displays import setup_price_display, setup_heatmap_display
from process_forms import process_options_form, process_heatmap_form

st.set_page_config(layout="wide")

state = st.session_state

if 'loaded' not in state:
    state.loaded = False
    state.option_params = {}
    state.heatmap_params = {}

with st.sidebar:    
    st.header("Select options parameters")
    options_parameters_form = st.container(border = True)#st.form(border = True, key = "options_parameters_form")
    
    st.header("Adjust your heatmap parameters")
    heatmap_parameters_form = st.container(border = False)
    
price_display = st.container(border = False)
heatmap_display = st.container(border = False)

if not state.loaded:
    process_options_form()
    process_heatmap_form()
    state.loaded = True

setup_options_form(options_parameters_form)
setup_heatmap_form(heatmap_parameters_form)
setup_price_display(price_display)
setup_heatmap_display(heatmap_display)