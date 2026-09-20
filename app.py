"""Run with: python -m streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="Road Damage Inspector", page_icon="🛣️", layout="wide")

page = st.navigation([
    st.Page("pages/0_Detection.py", title="Road Damage Inspector", icon="🛣️", default=True),
    st.Page("pages/1_Model_Statistics.py", title="Model Statistics", icon="📊",
            url_path="Model_Statistics"),
])
page.run()
