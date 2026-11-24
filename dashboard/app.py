"""Run from repository root: python -m streamlit run app.py."""
import streamlit as st

st.set_page_config(page_title="OptiEnergy AI", page_icon="⚡", layout="wide")
st.markdown("""<style>
.block-container {padding-top:2rem; max-width:1500px;}
[data-testid="stMetric"] {background:#151d31;border:1px solid #29334a;border-radius:14px;padding:18px;}
h1 {letter-spacing:-1.2px;} .eyebrow {color:#43dfb0;font-size:12px;letter-spacing:3px;}
</style>""", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("## ⚡ OptiEnergy AI")
    st.caption("OPTIMIZATION RESEARCH LAB")
    st.selectbox("Experiment results", ["demo", "full"], key="results_mode")
    st.caption("Local CPU · UCI 235 · next-hour kWh")
navigation = st.navigation([
    st.Page("pages/1_Overview.py", title="Overview", icon="📊"),
    st.Page("pages/2_Dataset_Analysis.py", title="Dataset Analysis", icon="🔎"),
    st.Page("pages/3_Model_Training.py", title="Model Training", icon="⚙️"),
    st.Page("pages/4_Optimizer_Comparison.py", title="Optimizer Comparison", icon="📈"),
    st.Page("pages/5_Optimization_Playground.py", title="Optimization Playground", icon="🧭"),
    st.Page("pages/6_Energy_Prediction.py", title="Energy Prediction", icon="⚡"),
    st.Page("pages/7_Reports_and_Results.py", title="Reports and Results", icon="📄"),
])
navigation.run()
