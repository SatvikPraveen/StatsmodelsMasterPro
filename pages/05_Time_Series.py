# 05_Time_Series.py

import streamlit as st
import pandas as pd
from pathlib import Path
from streamlit_app.utils import st_helpers as sth

# -----------------------------------------------
# ⚙️ Page Config
# -----------------------------------------------
st.set_page_config(page_title="Time Series Analysis", layout="wide")
st.title("⏱ Time Series Analysis")

# -----------------------------------------------
# 📥 Load Data
# -----------------------------------------------
DATA_PATH = Path(__file__).parent.parent / "synthetic_data" / "arima_series.csv"
df = pd.read_csv(DATA_PATH)

# Ensure time index
df['t'] = pd.to_datetime(df['t'])
df.set_index('t', inplace=True)
series = df['value']

# -----------------------------------------------
# 🔍 Data Preview & Summary
# -----------------------------------------------
st.subheader("🔍 Data Preview")
st.dataframe(df.head(), use_container_width=True)

with st.expander("📈 Summary Statistics"):
    st.dataframe(df.describe(), use_container_width=True)

# -----------------------------------------------
# 📉 Time Series Plot
# -----------------------------------------------
st.subheader("📉 Time Series Plot")
sth.plot_time_series(df, 'value')

# -----------------------------------------------
# 📊 ACF & PACF
# -----------------------------------------------
st.subheader("📊 ACF & PACF Diagnostics")
sth.plot_acf_pacf(series)

# -----------------------------------------------
# 🧪 Stationarity Test (ADF)
# -----------------------------------------------
st.subheader("🧪 Augmented Dickey-Fuller Test")
stat, p_val, stationary = sth.check_stationarity(series)

col1, col2, col3 = st.columns(3)
col1.metric("ADF Statistic", f"{stat:.4f}")
col2.metric("p-value", f"{p_val:.4f}")
col3.metric("Stationary?", "✅ Yes" if stationary else "❌ No")

# -----------------------------------------------
# 🔮 Fit ARIMA Model
# -----------------------------------------------
st.subheader("🔮 Fit ARIMA Model")

with st.expander("⚙️ Configure ARIMA Parameters", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    p = col1.number_input("AR (p)", min_value=0, max_value=5, value=1)
    d = col2.number_input("I (d)", min_value=0, max_value=2, value=0)
    q = col3.number_input("MA (q)", min_value=0, max_value=5, value=0)
    steps = col4.slider("Forecast Steps", 5, 30, 10)

    if st.button("🚀 Fit Model & Forecast"):
        model = sth.fit_arima_model(series, order=(p, d, q))
        st.text(model.summary())
        sth.plot_forecast(series, model, steps=steps)

# -----------------------------------------------
# 🧾 Footer
# -----------------------------------------------
st.markdown("---")
st.caption("StatsmodelsMasterPro • Streamlit App • Time Series Module")
