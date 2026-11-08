"""
CardioGuard AI — Predictive Sepsis & Hemodynamic Intelligence
Streamlit Cloud Deployment Entrypoint for DataNova 2026 Global Data-Thon
"""
import pathlib
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="CardioGuard AI — ICU Sepsis Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

html_path = pathlib.Path(__file__).parent / "index.html"
if html_path.exists():
    components.html(html_path.read_text(encoding="utf-8"), height=1080, scrolling=True)
else:
    st.error("index.html not found in repository root.")
