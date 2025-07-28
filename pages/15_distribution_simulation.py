# 15_distribution_simulation.py

import streamlit as st
import pandas as pd
from streamlit_app.utils import st_helpers as sth

# ---------------------------------------
# 🔧 Page Setup
# ---------------------------------------
st.set_page_config(page_title="Distribution Simulation", layout="wide")
st.title("🎲 Simulate and Visualize Distributions — Statsmodels + SciPy")

# ---------------------------------------
# 📌 Distribution Selection
# ---------------------------------------
st.sidebar.header("🧪 Distribution Settings")

dist_type = st.sidebar.selectbox(
    "Choose Distribution Type",
    ["Normal", "Exponential", "Poisson", "Binomial", "Gamma", "Beta", "Lognormal", "Uniform"]
)

sample_size = st.sidebar.slider("Number of Samples", min_value=100, max_value=5000, value=1000, step=100)

# ---------------------------------------
# ⚙️ Additional Configuration (Insert here)
# ---------------------------------------
# --- Sidebar Configuration ---
st.sidebar.header("🎛 Configuration")

# Reproducibility
seed = st.sidebar.number_input("🔢 Random Seed", min_value=0, value=42, step=1)

# Overlay options
show_pdf = st.sidebar.checkbox("📈 Show PDF/PMF Curve", value=True)
compare_two = st.sidebar.checkbox("🔁 Compare Two Distributions Side-by-Side")


# ---------------------------------------
# 🔧 Distribution Parameter Input
# ---------------------------------------
st.markdown("### 🔧 Configure Distribution Parameters")
params = None

if dist_type == "Normal":
    mu = st.number_input("Mean (μ)", value=0.0)
    sigma = st.number_input("Standard Deviation (σ)", min_value=0.01, value=1.0)
    params = (mu, sigma)

elif dist_type == "Exponential":
    scale = st.number_input("Scale (1/λ)", min_value=0.01, value=1.0)
    params = (0, scale)  # scipy uses loc=0 by default

elif dist_type == "Poisson":
    lam = st.number_input("λ (Rate)", min_value=0.0, value=5.0)
    params = (lam,)

elif dist_type == "Binomial":
    n = st.number_input("Number of Trials (n)", min_value=1, value=10)
    p = st.slider("Probability of Success (p)", 0.0, 1.0, 0.5)
    params = (int(n), p)

elif dist_type == "Gamma":
    shape = st.number_input("Shape (α)", min_value=0.01, value=2.0)
    scale = st.number_input("Scale (θ)", min_value=0.01, value=2.0)
    params = (shape, 0, scale)

elif dist_type == "Beta":
    a = st.number_input("Alpha (α)", min_value=0.01, value=2.0)
    b = st.number_input("Beta (β)", min_value=0.01, value=5.0)
    params = (a, b)

elif dist_type == "Lognormal":
    mean = st.number_input("Mean of Log (μ)", value=0.0)
    sigma = st.number_input("Std Dev of Log (σ)", min_value=0.01, value=0.5)
    params = (mean, sigma)

elif dist_type == "Uniform":
    low = st.number_input("Lower Bound", value=0.0)
    high = st.number_input("Upper Bound", value=1.0)
    if high <= low:
        st.warning("⚠️ Upper bound must be greater than lower bound.")
    params = (low, high)


# ---------------------------------------
# ▶️ Simulation Trigger
# ---------------------------------------
import pandas as pd
import numpy as np
import plotly.graph_objects as go

if st.button("🎲 Simulate Distribution") and params:
    try:
        # Simulate first distribution
        data, dist = sth.simulate_distribution(dist_type, params, size=sample_size, random_state=seed)
        df = pd.DataFrame({f"{dist_type}_sample": data})

        st.success("✅ Simulation completed successfully!")

        # --- Preview & Stats ---
        with st.expander("📄 Dataset Preview"):
            st.dataframe(df.head())
            st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} column")

        st.markdown("### 📊 Summary Statistics")
        st.dataframe(df.describe().T.style.format(precision=3))

        # --- Visualization ---
        st.markdown("### 📉 Histogram with Optional PDF/PMF")

        fig = go.Figure()
        x_range = np.linspace(min(data), max(data), 200)

        fig.add_trace(go.Histogram(
            x=data,
            name=f"{dist_type} Histogram",
            histnorm="probability density" if dist_type not in ["Poisson", "Binomial"] else "probability",
            opacity=0.6
        ))

        if show_pdf:
            if dist_type in ["Poisson", "Binomial"]:
                x_discrete = np.arange(np.min(data), np.max(data) + 1)
                fig.add_trace(go.Scatter(
                    x=x_discrete,
                    y=dist.pmf(x_discrete),
                    mode="lines+markers",
                    name=f"{dist_type} PMF",
                    line=dict(color="crimson")
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=dist.pdf(x_range),
                    mode="lines",
                    name=f"{dist_type} PDF",
                    line=dict(color="crimson")
                ))

        # --- Optional Second Distribution ---
        if compare_two:
            st.markdown("## 🔁 Second Distribution for Comparison")
            dist2_type = st.selectbox("📦 Second Distribution", ["Normal", "Exponential", "Poisson", "Binomial", "Gamma", "Beta", "Lognormal", "Uniform"], index=0)
            st.caption("Enter matching parameters in the sidebar or use reasonable guesses.")

            if dist2_type == "Normal":
                p2 = (0, 1)
            elif dist2_type == "Poisson":
                p2 = (5,)
            elif dist2_type == "Exponential":
                p2 = (0, 1)
            elif dist2_type == "Binomial":
                p2 = (10, 0.5)
            elif dist2_type == "Gamma":
                p2 = (2, 0, 2)
            elif dist2_type == "Beta":
                p2 = (2, 5)
            elif dist2_type == "Lognormal":
                p2 = (0, 1)
            elif dist2_type == "Uniform":
                p2 = (0, 1)

            data2, dist2 = sth.simulate_distribution(dist2_type, p2, size=sample_size, random_state=seed + 100)
            x_range2 = np.linspace(min(data2), max(data2), 200)

            fig.add_trace(go.Histogram(
                x=data2,
                name=f"{dist2_type} Histogram",
                histnorm="probability density" if dist2_type not in ["Poisson", "Binomial"] else "probability",
                opacity=0.5
            ))

            if show_pdf:
                if dist2_type in ["Poisson", "Binomial"]:
                    x_disc2 = np.arange(np.min(data2), np.max(data2) + 1)
                    fig.add_trace(go.Scatter(
                        x=x_disc2,
                        y=dist2.pmf(x_disc2),
                        mode="lines+markers",
                        name=f"{dist2_type} PMF",
                        line=dict(dash="dash", color="green")
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=x_range2,
                        y=dist2.pdf(x_range2),
                        mode="lines",
                        name=f"{dist2_type} PDF",
                        line=dict(dash="dash", color="green")
                    ))

        # --- Final Figure Config ---
        fig.update_layout(
            title="Histogram + PDF/PMF Overlay",
            xaxis_title="Value",
            yaxis_title="Density / Probability",
            bargap=0.1,
            template="simple_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Download ---
        with st.expander("📥 Download Simulated Data"):
            st.download_button(
                label="📥 Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"{dist_type.lower()}_simulation.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ Error during simulation: {e}")


# ---------------------------------------
# 📘 Footer
# ---------------------------------------
st.markdown("---")
st.caption("StatsmodelsMasterPro • Streamlit App • Distribution Simulation Module")
