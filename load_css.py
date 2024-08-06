""" Load CSS file into streamlit. """

import streamlit as st

def load_css(file_name):
    """
    Load CSS file into streamlit.

    Parameters
    ----------
    file_name : str
        CSS file filename.

    Returns
    -------
    None.

    """
    
    with open(file_name) as f:
        css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)