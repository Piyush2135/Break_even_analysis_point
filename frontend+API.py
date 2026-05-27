import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from backend import *
import io

# Import autorefresh with error handling
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ================= PAGE CONFIG (BEFORE AUTOREFRESH) =================
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# ================= SESSION STATE INITIALIZATION =================
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()

st.title("🪁 Break-Even Point Analysis")
st.markdown("### AI + Real Data + Decision Intelligence System")

# ================= CSV VALIDATION & SANITIZATION =================
def validate_csv_format(df):
    """
    VALUES EXPLAINED:
    - The code expects a DataFrame with at least two columns: "Day" (time periods) and "Demand" (units/revenue)
    - WHY: Linear regression needs X (independent variable) and Y (dependent variable)
      * Day = X axis (time progression: 1, 2, 3... or Week1, Week2...)
      * Demand = Y axis (actual sales/quantity: varies over time)
    - The algorithm: Y = mX + c (slope-intercept form)
      * m = slope (how much demand changes per day)
      * c = y-intercept (baseline demand when day=0)
    """
    errors = []
    
    # Check for minimum rows
    if len(df) < 2:
        errors.append("CSV must have at least 2 rows of data")
        return df, errors
    
    # Check for required columns (flexible naming)
    day_cols = [col.lower() for col in df.columns]
    demand_cols = [col.lower() for col in df.columns]
    
    has_day = any(col in ['day', 'date', 'time', 'period', 'week', 'month'] for col in day_cols)
    has_demand = any(col in ['demand', 'sales', 'quantity', 'units', 'revenue', 'quantity_sold'] for col in demand_cols)
    
    if not has_day:
        errors.append("❌ Missing time column. Expected: 'Day', 'Date', 'Week', 'Month', or 'Period'")
    if not has_demand:
        errors.append("❌ Missing demand column. Expected: 'Demand', 'Sales', 'Units', 'Quantity', or 'Revenue'")
    
    if errors:
        return df, errors
    
    # Standardize column names
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['day', 'date', 'time', 'period', 'week', 'month']:
            col_mapping[col] = 'Day'
        elif col_lower in ['demand', 'sales', 'quantity', 'units', 'revenue', 'quantity_sold']:
            col_mapping[col] = 'Demand'
    
    df_standardized = df.rename(columns=col_mapping)
    
    # Validate data types
    try:
        df_standardized['Day'] = pd.to_numeric(df_standardized['Day'], errors='coerce')
        df_standardized['Demand'] = pd.to_numeric(df_standardized['Demand'], errors='coerce')
    except Exception as e:
        errors.append(f"❌ Data type error: Columns must be numeric. Error: {str(e)}")
        return df, errors
    
    # Remove NaN values
    df_standardized = df_standardized.dropna()
    
    # Check for negative values
    if (df_standardized['Demand'] < 0).any():
        errors.append("⚠️ Warning: Negative demand values detected and ignored")
        df_standardized = df_standardized[df_standardized['Demand'] >= 0]
    
    return df_standardized[['Day', 'Demand']], errors

def generate_sample_csv():
    """Generate a sample CSV for download"""
    sample_data = {
        'Day': list(range(1, 21)),
        'Demand': [100 + np.random.randint(-20, 50) for _ in range(20)]
    }
    df = pd.DataFrame(sample_data)
    return df.to_csv(index=False).encode('utf-8')

# ================= LIVE DATA FUNCTIONS =================
def fetch_coindesk_data():
    """Fetch crypto price data from CoinDesk"""
    try:
        url = "https://api.coindesk.com/v1/bpi/currentprice.json"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        data = response.json()
        price = data["bpi"]["USD"]["rate_float"]
        days = np.arange(1, 21)
        demand = price + np.random.normal(0, 20, len(days))
        return pd.DataFrame({"Day": days, "Demand": demand}), "CoinDesk (Crypto)"
    except Exception as e:
        st.warning(f"CoinDesk API failed: {str(e)}. Falling back to simulation.")
        return None, "CoinDesk (Crypto) - Failed"

def fetch_random_market_data():
    """Fetch simulated market data"""
    try:
        base_price = np.random.uniform(50, 200)
        days = np.arange(1, 21)
        demand = base_price + np.random.normal(0, 15, len(days))
        return pd.DataFrame({"Day": days, "Demand": demand}), "Market Simulation"
    except Exception as e:
        st.warning(f"Market simulation failed: {str(e)}")
        return None, "Market Simulation - Failed"

def fetch_multiple_sources():
    """Try multiple data sources"""
    sources = [fetch_coindesk_data, fetch_random_market_data]
    for source_func in sources:
        data, source_name = source_func()
        if data is not None:
            return data, source_name
    return generate_data(), "Default"

# ================= DATA SOURCE =================
st.sidebar.header("📂 Data Source")

# Select data source
data_source_options = ["Simulation Mode", "Upload CSV", "Live Data (Multiple Sources)"]
data_source = st.sidebar.radio("Select Data Source", data_source_options)

# Get data based on selection
if data_source == "Upload CSV":
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 CSV Format Guide")
    with st.sidebar.expander("📋 Click to see CSV requirements"):
        st.markdown("""
        **Your CSV must have 2 columns:**
        
        | Day | Demand |
        |-----|--------|
        | 1   | 150    |
        | 2   | 165    |
        | 3   | 142    |
        | ... | ...    |
        
        **Column Names (flexible):**
        - Time: Day, Date, Week, Month, Period
        - Demand: Demand, Sales, Units, Quantity, Revenue
        
        **Rules:**
        - Min 2 rows of data required
        - Values must be numeric
        - No negative demand
        """)
    
    # Download sample CSV
    st.sidebar.download_button(
        label="📥 Download Sample CSV",
        data=generate_sample_csv(),
        file_name="sample_demand_data.csv",
        mime="text/csv",
        help="Download a template to understand the required format"
    )
    
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"], help="Select your CSV file with demand data")
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            data, errors = validate_csv_format(data)
            
            if errors:
                st.error("⚠️ CSV Validation Issues:")
                for error in errors:
                    st.warning(error)
                st.info("💡 Download the sample CSV template to see the correct format")
                st.stop()
            
            source_label = "📤 CSV Upload"
            st.sidebar.success(f"✅ CSV loaded: {len(data)} rows, {len(data.columns)} columns")
        except Exception as e:
            st.error(f"❌ Error reading CSV: {str(e)}")
            st.info("Make sure your file is a valid CSV with numeric columns")
            st.stop()
    else:
        st.info("Please upload a CSV file to proceed.")
        st.stop()

elif data_source == "Live Data (Multiple Sources)":
    data, source_label = fetch_multiple_sources()
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=5000, key="live_refresh")
    st.caption(f"🔴 Live Mode - {source_label} - Refreshing every 5 seconds")

else:  # Simulation Mode
    data = generate_data()
    source_label = "🟢 Simulation Mode"
    st.caption(source_label)

# ================= INPUT =================
st.sidebar.header("🕙 Business Inputs")

# Create columns for better layout
col_fc, col_sp, col_vc = st.sidebar.columns(3)

with col_fc:
    FC = st.slider("Fixed Cost (₹)", 1000, 1000000, 50000, step=1000)

with col_sp:
    SP = st.slider("Selling Price (₹)", 10, 5000, 500, step=10)

with col_vc:
    VC = st.slider("Variable Cost (₹)", 5, 4000, 300, step=5)

if SP <= VC:
    st.error("❌ Selling Price must be greater than Variable Cost")
    st.stop()

# Display current input values
with st.sidebar.expander("📊 Current Inputs"):
    st.write(f"Fixed Cost: ₹{FC:,.2f}")
    st.write(f"Selling Price: ₹{SP:,.2f}")
    st.write(f"Variable Cost: ₹{VC:,.2f}")
    st.write(f"Contribution Margin: ₹{SP - VC:,.2f}")
   

# ================= DATA PROCESSING =================
X = data["Day"].values
y = data["Demand"].values

# ================= ML REGRESSION =================
m, c = linear_regression(X, y)
future_days, predicted = predict_demand(m, c)

# ================= SEASONAL EFFECT =================
seasonality = 20 * np.sin(future_days / 3)
predicted = predicted + seasonality
predicted = np.clip(predicted, 0, None)  # Ensure non-negative predictions

# ================= GRAPH =================
st.subheader("📈 Demand Forecast & Analysis")

# Create two columns for graphs
col_graph1, col_graph2 = st.columns(2)

with col_graph1:
    st.write("**Demand Prediction**")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data["Day"], 
        y=data["Demand"], 
        mode='lines+markers',
        name="Actual Data",
        line=dict(color="#b41f71", width=2),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=future_days, 
        y=predicted, 
        mode='lines+markers',
        name="Predicted Demand",
        line=dict(color="#0ebfff", width=2, dash='dash'),
        marker=dict(size=6)
    ))

    fig.update_layout(
        xaxis_title="Days",
        yaxis_title="Demand Units",
        hovermode="x unified",
        template="plotly_white",
        height=400,
        showlegend=True,
        legend=dict(x=0.7, y=0.95)
    )
    
    st.plotly_chart(fig, width='stretch', key="fig1")

# ================= ANALYSIS =================
avg_demand = np.mean(predicted)

bep = break_even(FC, SP, VC)
profit = calculate_profit(SP, VC, FC, avg_demand)

# Display KPIs
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Break-even Units",
        f"{bep:.0f}",
        delta=f"{(avg_demand - bep):.0f}" if avg_demand > bep else f"{(avg_demand - bep):.0f}",
        delta_color="normal" if avg_demand > bep else "inverse"
    )

with col2:
    st.metric(
        "Avg Predicted Demand",
        f"{avg_demand:.0f}",
        delta=f"{((avg_demand/bep - 1)*100):.1f}% of BEP" if bep > 0 else "N/A"
    )

with col3:
    st.metric(
        "Expected Profit",
        f"₹{profit:,.0f}",
        delta="Positive" if profit > 0 else "Negative",
        delta_color="normal" if profit > 0 else "inverse"
    )

with col_graph2:
    st.write("**Profit Curve Analysis**")
    Q = np.arange(0, 1000, 10)
    profit_curve = (SP - VC)*Q - FC

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=Q, y=profit_curve, name="Profit", fill='tozeroy',
                              line=dict(color="#58a02c", width=2)))

    fig2.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
    fig2.add_vline(x=bep, line_dash="dash", line_color="red")

    fig2.update_layout(
        xaxis_title="Quantity Sold",
        yaxis_title="Profit (₹)",
        template="plotly_white",
        height=400,
        hovermode="x unified"
    )
    st.plotly_chart(fig2, width='stretch', key="fig2")

# ================= INSIGHT ENGINE =================
st.subheader("🧠 Smart Insight Engine")

with st.container(border=True):
    if avg_demand > bep:
        st.success("🟢 **Strong Business**: Demand ({:.0f}) exceeds break-even ({:.0f})".format(avg_demand, bep))
        st.write(f"You need to sell **{bep:.0f}** units to break even. Predicted demand is **{avg_demand:.0f}** units.")
        surplus = ((avg_demand - bep) / bep) * 100
        st.write(f"Your predicted demand exceeds break-even by **{surplus:.1f}%** - Excellent opportunity! ✨")
    elif avg_demand > 0.8*bep:
        st.warning("🟡 **Risk Zone**: Demand is close to break-even. Tighten costs or increase prices.")
        st.write(f"Current demand ({avg_demand:.0f}) is {((avg_demand/bep)*100):.1f}% of break-even point.")
    else:
        st.error("🔴 **Not Viable**: Demand is too low. Business needs adjustment.")
        st.write(f"Target demand: {bep:.0f} units | Predicted: {avg_demand:.0f} units | Gap: {(bep - avg_demand):.0f} units")

# ================= PRICING OPTIMIZER =================
st.subheader("🪄 Pricing Optimizer")

col_target, col_margin = st.columns(2)
with col_target:
    target = st.number_input("Target Profit (₹)", min_value=0, value=50000, step=5000)
with col_margin:
    st.metric("Current Margin %", f"{((SP-VC)/SP * 100):.1f}%")

opt_price = optimal_price(FC, VC, target, avg_demand)

if opt_price > 0:
    price_diff = opt_price - SP
    color = "🟢" if price_diff > 0 else "🔴"
    with st.container(border=True):
        st.write(f"**{color} Recommended Selling Price: ₹{opt_price:.2f}**")
        st.write(f"Current Price: ₹{SP:.2f} | Suggested Change: {'+' if price_diff > 0 else ''}{price_diff:.2f}")
        if price_diff > 0:
            st.write(f"💡 Increase price by ₹{price_diff:.2f} to achieve target profit")
        else:
            st.write(f"💡 Decrease price by ₹{abs(price_diff):.2f} (careful with margins!)")
else:
    st.warning("⚠️ Pricing optimization not possible with current parameters")

# ================= REPORT =================
st.subheader("📄 Auto Business Report")

report_data = {
    "Metric": [
        "Average Predicted Demand",
        "Break-even Point",
        "Expected Profit",
        "Sales Price",
        "Variable Cost",
        "Fixed Cost",
        "Profit Margin %",
        "Viability Status"
    ],
    "Value": [
        f"{avg_demand:.0f} units",
        f"{bep:.0f} units",
        f"₹{profit:,.2f}",
        f"₹{SP}",
        f"₹{VC}",
        f"₹{FC:,}",
        f"{((SP-VC)/SP * 100):.1f}%",
        "✅ Viable" if avg_demand > bep else "❌ Review Needed"
    ]
}

report_df = pd.DataFrame(report_data)
st.dataframe(report_df, width='stretch', hide_index=True, key="report_df")

# ================= EXPORT REPORT =================
st.subheader("💾 Export & Actions")
col_export1, col_export2 = st.columns(2)

with col_export1:
    csv = report_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Report (CSV)",
        data=csv,
        file_name=f"break_even_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

with col_export2:
    if st.button("🔄 Refresh Analysis Now"):
        st.rerun()