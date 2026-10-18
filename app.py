from pathlib import Path

import streamlit as st

import analysis
from model import train
from views import dots, input_view, result_view, ticker

st.set_page_config(
    page_title="Package Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS_PATH = Path(__file__).parent / "assets" / "style.css"
st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    df, lr, x_train, x_test, y_train, y_test = train()
    return df, lr, analysis.build(df, lr, x_train, x_test, y_train, y_test)


df, lr, ana = get_model()

st.session_state.setdefault("page", "input")
st.session_state.setdefault("cgpa", 7.0)
st.session_state.setdefault("visit", 0)
st.session_state.setdefault("last_ts", None)

dots()
ticker(lr, ana["test"])

if st.session_state.page == "result":
    result_view(df, lr, ana)
else:
    input_view(ana["test"])
