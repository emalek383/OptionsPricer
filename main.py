import streamlit as st
from streamlit_javascript import st_javascript
from user_agents import parse

from load_css import load_css
from setup_forms import setup_options_form, setup_heatmap_form
from setup_displays import setup_options_price_display, setup_heatmap_display, setup_options_params_display
from process_forms import process_options_form, process_heatmap_form


def detect_device():
    """
    Detect whether the page is being viewed on PC.

    Returns
    -------
    bool
        True if being viewed on PC, false otherwise.

    """
    
    if 'is_session_pc' not in st.session_state:
        st.session_state.is_session_pc = True
    try:
        ua_string = st_javascript("navigator.userAgent")
        if ua_string is not None:
            user_agent = parse(ua_string)
            st.session_state.is_session_pc = user_agent.is_pc
        else:
            st.session_state.is_session_pc = True
    except Exception:
        st.session_state.is_session_pc = True
    return st.session_state.is_session_pc


st.set_page_config(layout="wide")

detect_device()

state = st.session_state

if 'loaded' not in state:
    state.loaded = False
    state.option_params = {}
    state.heatmap_params = {}

load_css('styles/style.css')

with st.sidebar:
    option_parameters_form = st.container(border = False)
    
    #option_parameters_expander = st.expander("Select option parameters", expanded = True)
    #option_parameters_form = option_parameters_expander.container(border = False)
    
    # heatmap_parameters_expander = st.expander("Adjust heatmap parameters", expanded = True)
    # heatmap_parameters_form = heatmap_parameters_expander.container(border = False)
    
st.header("Options Pricing Tool")
parameters_display = st.container(border = False)
price_display = st.container(border = False)
heatmap_display = st.container(border = False)

if not state.loaded:
    process_options_form()
    process_heatmap_form()
    state.loaded = True

setup_options_form(option_parameters_form)

setup_options_params_display(parameters_display)
setup_options_price_display(price_display)
setup_heatmap_display(heatmap_display)