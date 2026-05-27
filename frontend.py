import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from backend import *

st.set_page_config(layout="wide")

st.title("🪁 Break-Even Point Analysis")
st.markdown("### AI + Real Data + Decision Intelligence System")

# ================= FILE UPLOAD =================
st.sidebar.header("📂 Data Input")

uploaded_file = st.sidebar.file_uploader("Upload CSV (Day, Demand)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    data = generate_data()

# ================= INPUT =================
st.sidebar.header("🕙 Business Inputs")

FC = st.sidebar.slider("Fixed Cost", 10000, 20000, 50000)
SP = st.sidebar.slider("Selling Price", 100, 1000, 500)
VC = st.sidebar.slider("Variable Cost", 50, 900, 300)

if SP <= VC:
    st.error("Selling Price must be greater than Variable Cost")
    st.stop()

# ================= DATA =================
X = data["Day"].values
y = data["Demand"].values

# ================= ML =================
m, c = linear_regression(X, y)
future_days, predicted = predict_demand(m, c)

# ================= SEASONAL EFFECT =================
seasonality = 20 * np.sin(future_days / 3)
predicted = predicted + seasonality

# ================= GRAPH =================
st.subheader("🏄‍♀️ Demand Forecast (Real + Predicted + Seasonal)")

fig = go.Figure()

fig.add_trace(go.Scatter(x=data["Day"], y=data["Demand"], name="Actual Data"))
fig.add_trace(go.Scatter(x=future_days, y=predicted, name="Predicted + Seasonal"))

st.plotly_chart(fig, use_container_width=True)

# ================= ANALYSIS =================
avg_demand = np.mean(predicted)

bep = break_even(FC, SP, VC)
profit = calculate_profit(SP, VC, FC, avg_demand)

col1, col2, col3 = st.columns(3)

col1.metric("Break-even Units", f"{bep:.2f}")
col2.metric("Predicted Demand", f"{avg_demand:.2f}")
col3.metric("Expected Profit", f"{profit:.2f}")

# ================= SMART INSIGHT =================
st.subheader("🧠 Smart Insight Engine")

if avg_demand > bep:
    st.success("🟢 Strong Business: Demand exceeds break-even")
elif avg_demand > 0.8*bep:
    st.warning("🟡 Risk Zone: Close to break-even")
else:
    st.error("🔴 Not Profitable: Needs pricing adjustment")

# ================= AI PRICING =================
st.subheader("🪄 Pricing Optimizer")

target = st.slider("Target Profit", 10000, 200000, 50000)

opt_price = optimal_price(FC, VC, target, avg_demand)

st.write(f"🩺 Recommended Selling Price: ₹ {opt_price:.2f}")

# ================= PROFIT CURVE =================
st.subheader("📊 Profit Curve")
Q = np.arange(0, 1000, 10)
profit_curve = (SP - VC)*Q - FC

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=Q, y=profit_curve, name="Profit"))
fig2.add_hline(y=0)
fig2.add_vline(x=bep)

st.plotly_chart(fig2, use_container_width=True)

# ================= MINI REPORT =================
st.subheader("📄 Auto Business Report")

st.write(f"""
- Predicted Demand: {avg_demand:.2f}
- Break-even Point: {bep:.2f}
- Expected Profit: {profit:.2f}

Conclusion:
""")

if avg_demand > bep:
    st.write("✔ Business is financially viable under current conditions.")
else:
    st.write("❌ Business needs strategic changes in pricing or cost.")