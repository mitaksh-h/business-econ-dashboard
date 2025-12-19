import streamlit as st
import pandas as pd

# ----------------------------
# Load Dataso gove me 
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("./data/transactions.csv")

    # FIX: ensure Arrow-safe types for Streamlit performance
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)

    return df

df = load_data()
# ----------------------------
# Preprocess once (CRITICAL for speed)
# ----------------------------
@st.cache_data
def preprocess_df(df):
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    df["date_only"] = df["date_dt"].dt.date
    return df

# Preprocess once (CRITICAL for speed)
df = preprocess_df(df)
# ----------------------------
# Performance defaults (defined BEFORE use)
# ----------------------------
fast_mode = True
enable_advanced = False
enable_ml = False
# ---------- Luxury Display Name Layer (Safe, No Monkey‑Patching) ----------

DISPLAY_NAME_MAP = {
    "revenue": "Revenue",
    "total_revenue": "Total Revenue",
    "gross_margin": "Gross Margin",
    "total_gross_margin": "Total Gross Margin",
    "marketing_cost": "Marketing Cost",
    "cac": "Customer Acquisition Cost",
    "customer_acquisition_cost": "Customer Acquisition Cost",
    "opex_allocated": "Operating Expense Allocated",
    "opex": "Operating Expense",
    "profit_per_order": "Profit Per Order",
    "net_profit": "Net Profit",
    "total_opex": "Total Operating Expense",
    "cogs": "Cost Of Goods Sold",
    "aov": "Average Order Value",
    "avg_basket_size": "Average Basket Size",
    "items_count": "Items Count",
    "order_id": "Order ID",
    "user_id": "User ID",
    "order_date": "Order Date",
    "date": "Date",
    "date_dt": "Date",
    "city": "City",
    "marketing_source": "Marketing Source",
    "promo_used": "Promo Used",
    "cohort_week": "Cohort Week",
    "y": "Actual",
    "yhat": "Predicted",
    "yhat_lower": "Predicted Lower Bound",
    "yhat_upper": "Predicted Upper Bound",
    # ---- Business‑friendly ML & Ops labels ----
    "transaction_id": "Transaction ID",
    "customer_id": "Customer ID",
    "total_orders": "Total Orders",
    "avg_basket": "Average Basket",
    "promo_rate": "Promo Rate",
    "days_since_first": "Days Since First Order",
    "days_since_last": "Days Since Last Order",
    "R_score": "Recency Score",
    "F_score": "Frequency Score",
    "M_score": "Monetary Score",
    "RFM": "RFM Composite Score",
    "rfm_churn": "RFM Churn Flag",
    "behavior_churn": "Behavioral Churn Flag",
    "churned": "Churned",
    "churn_probability": "Churn Probability",
    "ltv_actual": "Actual LTV",
    "ltv_pred": "Predicted LTV",
    "profit": "Profit",
    "delivery_time": "Delivery Time",
    "cancel_reason": "Cancel Reason",
    "first_order": "First Order",
    "last_order": "Last Order",
    "date_only": "Date Only",
    "synthetic_user_id": "Synthetic User ID",
    "orders": "Orders",
    "total_spend": "Total Spend",
    "roi": "ROI",
    "flagIso": "Isolation Forest Flag",
    "flagZRevenue": "Z-Score Revenue Flag",
    "flagZProfit": "Z-Score Profit Flag",
    "flagZCac": "Z-Score CAC Flag",
    "flagMadRevenue": "MAD Revenue Flag",
    "flagMadProfit": "MAD Profit Flag",
    "flagMadCac": "MAD CAC Flag",
    "flagProphet": "Prophet Flag",
    "any_flag": "Any Flag",
}

def pretty_label(name: str) -> str:
    try:
        key = str(name).strip()
    except:
        return name
    if key in DISPLAY_NAME_MAP:
        return DISPLAY_NAME_MAP[key]
    return key.replace("_", " ").title()

def apply_display_names(df):
    df = df.copy()
    df.columns = [pretty_label(c) for c in df.columns]
    return df

# End of luxury display layer

# -------- Global Display Monkey-Patch (Applies to ALL UI except KPI cards) --------
_original_dataframe = st.dataframe
_original_table = getattr(st, "table", None)
_original_line_chart = getattr(st, "line_chart", None)

def _patched_dataframe(df, *args, **kwargs):
    try:
        if isinstance(df, pd.DataFrame):
            return _original_dataframe(apply_display_names(df), *args, **kwargs)
    except Exception:
        pass
    return _original_dataframe(df, *args, **kwargs)

def _patched_table(df, *args, **kwargs):
    try:
        if isinstance(df, pd.DataFrame):
            return _original_table(apply_display_names(df), *args, **kwargs)
    except Exception:
        pass
    return _original_table(df, *args, **kwargs)

def _patched_line_chart(data=None, *args, **kwargs):
    try:
        if isinstance(data, pd.DataFrame):
            data = apply_display_names(data)
    except Exception:
        pass
    return _original_line_chart(data, *args, **kwargs)

if not fast_mode:
    st.dataframe = _patched_dataframe
    if _original_table is not None:
        st.table = _patched_table
    if _original_line_chart is not None:
        st.line_chart = _patched_line_chart
else:
    st.dataframe = _original_dataframe
    if _original_table is not None:
        st.table = _original_table
    if _original_line_chart is not None:
        st.line_chart = _original_line_chart

# --------------------------------------------
# Synthetic user_id generation (non-destructive)
# --------------------------------------------
import numpy as np

if not any(col.lower() in ["user_id","userid","customer_id","customer","user"] for col in df.columns):
    num_users = max(50, len(df) // 5)  # approx one user per 5 orders
    df["synthetic_user_id"] = np.random.randint(1, num_users + 1, size=len(df))
    inferred_user_col = "synthetic_user_id"
else:
    # Use the real one
    for c in df.columns:
        if str(c).lower() in ["user_id","userid","customer_id","customer","user"]:
            inferred_user_col = c
            break

# Try to infer a user identifier column (user-level metrics depend on this)
# Use the inferred synthetic or real user column
user_col = inferred_user_col

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Business Econ Dashboard",
    layout="wide"
)

# ----------------------------
# Global Styles (Dark, Full-width, Accent #1A362A)
# ----------------------------
st.markdown("""
<style>
/* -------- GLOBAL -------- */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", system-ui, sans-serif;
    background: radial-gradient(circle at 20% 20%, #0d0d0d 0%, #000000 80%);
    color: #F2F2F2;
    overflow-x: hidden;
}


/* -------- LAYOUT -------- */
main .block-container {
    padding-top: 2.5rem;
    padding-bottom: 4rem;
    padding-left: 4rem;
    padding-right: 4rem;
    max-width: 1200px;
    margin-left: auto;
    margin-right: auto;
}

.hero {
    display: inline-block;
    font-size: 80px !important;
    font-weight: 1000;
    text-align: center;
    letter-spacing: -1.4px;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", system-ui, sans-serif;

    background-image: linear-gradient(90deg, #6fbfa2, #1a362a, #78c9ad, #1a362a, #6fbfa2);
    background-size: 200% auto;
    background-position: 0% 50%;
    -webkit-background-clip: text;
    color: transparent;

    animation: heroFlow 32s linear infinite;
    margin-bottom: 1.2rem;
}

@keyframes heroFlow {
    0%   { background-position: 0% 50%; }
    100% { background-position: -200% 50%; }
}
.subhero {
    font-size: 22px;
    font-weight: 400;
    text-align: center;
    color: #D6D6D6;
    letter-spacing: -0.2px;
    margin-bottom: 3rem;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", system-ui, sans-serif;
    
}

/* -------- GLASS EFFECT -------- */
.glass {
    background: rgba(255,255,255,0.05);
    border-radius: 22px;
    backdrop-filte
r: blur(30px) saturate(180%);
    -webkit-backdrop-filter: blur(30px) saturate(180%);
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 0 45px rgba(0,0,0,0.55);
    padding: 22px 26px;
    transition: all 0.25s ease-out;
}
.glass:hover {
    box-shadow: 0 0 65px rgba(0,255,200,0.25);
    transform: translateY(-3px);
}

.kpi-row {
    display: grid !important;
    grid-template-columns: repeat(4, 260px) !important;  /* 4 cards per row */
    grid-auto-rows: 160px !important;                   /* 2 rows of same height */
    gap: 28px !important;

    width: 100% !important;
    max-width: 1200px !important;

    margin-left: auto !important;
    margin-right: auto !important;

    justify-content: center !important;
    justify-items: center !important;
}

.kpi-card {
    width: 260px !important;
    height: 160px !important;

    background: rgba(255,255,255,0.05);
    padding: 26px;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,0.10);
    backdrop-filter: blur(30px) saturate(180%);
    -webkit-backdrop-filter: blur(30px) saturate(180%);
    box-shadow: 0 0 22px rgba(0,0,0,0.45);
    transition: transform 0.25s ease, box-shadow 0.25s ease;

    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
}
.kpi-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 0 25px rgba(182,255,221,0.22);
}
.kpi-label {
    font-size: 12px;
    color: #E4E4E4;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 800;
    font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 26px;
    font-weight: 900;
    color: #FFFFFF;
    font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    letter-spacing: -0.5px;
}

/* Soft neon accent bar */
.kpi-accent {
    height: 3px;
    width: 40px;
    background: linear-gradient(90deg, #b6ffdd, #1a362a);
    border-radius: 999px;
    margin-bottom: 12px;
}

/* -------- PANELS -------- */
.section-panel {
    background: rgba(255,255,255,0.03);
    border-radius: 26px;
    padding: 28px 28px 26px 28px;
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 0 50px rgba(0,0,0,0.7);
    backdrop-filter: blur(30px) saturate(180%);
    -webkit-backdrop-filter: blur(30px) saturate(180%);
    margin-bottom: 36px;
}

/* -------- TITLES -------- */
.section-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 8px;
}
.section-caption {
    font-size: 14px;
    color: #A0A0A0;
    margin-bottom: 20px;
}

/* -------- TABS -------- */
button[role="tab"] {
    border-radius: 999px !important;
    padding: 8px 20px !important;
    background-color: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #EAEAEA !important;
    transition: all 0.2s ease-out;
}
button[role="tab"]:hover {
    background-color: rgba(182,255,221,0.12) !important;
    box-shadow: 0 0 22px rgba(182,255,221,0.35);
}
button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #b6ffdd, #1a362a) !important;
    border-color: rgba(255,255,255,0.4) !important;
    color: black !important;
    font-weight: 700 !important;
}

/* Remove ugly underline */
.stTabs [data-baseweb="tab-highlight"] {
    border-bottom: none !important;
}

/* -------- SIDEBAR -------- */
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.65);
    backdrop-filter: blur(18px);
    border-right: 1px solid rgba(255,255,255,0.08);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("### ⚡ Performance")
    enable_advanced = st.checkbox("Enable Advanced KPIs", value=False)
    enable_ml = st.checkbox("Enable ML Lab", value=False)
    fast_mode = st.checkbox("Fast Mode (Disable display mapping)", value=True)
    st.markdown("---")
    st.markdown("### 📊 Dashboard")
    st.write("Business Economics & Insights")
    st.markdown("---")
    st.markdown("**Sections**")
    st.write("• KPIs & Overview")
    st.write("• Cities & Marketing")
    st.write("• Profit & Simulator")
    st.write("• Auto Insights")
    st.markdown("---")
    st.caption("Accent: #1A362A · Dark hybrid theme")

# ----------------------------
# Hero
# ----------------------------
st.markdown('<p class="hero">Business Intelligence, Done Right.</p>', unsafe_allow_html=True)
st.markdown('<div class="fade-in"><p class="subhero">Dark, focused, and built to show you what actually makes money.</p></div>', unsafe_allow_html=True)

# ----------------------------
# KPI Calculations
# ----------------------------
total_revenue = df["revenue"].sum()
total_cogs = df["cogs"].sum()
total_gross_margin = df["gross_margin"].sum()
total_opex = df["opex_allocated"].sum()
net_profit = total_gross_margin - total_opex - df["marketing_cost"].sum()

# New KPIs
best_city = df.groupby("city")["gross_margin"].mean().idxmax()
top_marketing_source = df.groupby("marketing_source")["revenue"].sum().idxmax()
avg_basket_size = df["revenue"].mean()

# ----------------------------
# KPI Row (Glass-style)
# ----------------------------
kpis_html = f"""
<div class="kpi-row">
    <div class="kpi-card">
        <div class="kpi-accent"></div>
        <div class="kpi-label">Revenue</div>
        <div class="kpi-value">₹{total_revenue:,.0f}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-accent"></div>
        <div class="kpi-label">COGS</div>
        <div class="kpi-value">₹{total_cogs:,.0f}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-accent"></div>
        <div class="kpi-label">Gross Margin</div>
        <div class="kpi-value">₹{total_gross_margin:,.0f}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-accent"></div>
        <div class="kpi-label">OPEX</div>
        <div class="kpi-value">₹{total_opex:,.0f}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-accent"></div>
        <div class="kpi-label">Net Profit</div>
        <div class="kpi-value">₹{net_profit:,.0f}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-accent"></div>
        <div class="kpi-label">Best City</div>
        <div class="kpi-value">{best_city}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-accent"></div>
        <div class="kpi-label">Top Marketing Source</div>
        <div class="kpi-value">{top_marketing_source}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-accent"></div>
        <div class="kpi-label">Avg Basket Size</div>
        <div class="kpi-value">₹{avg_basket_size:,.0f}</div>
    </div>
</div>
"""



st.markdown(kpis_html, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")  # small spacing

# ----------------------------
# Tabs Layout
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Cities", "Marketing", "Profit + Simulator"])

# ----------------------------
# TAB 1 — Overview
# ----------------------------
with tab1:
    st.markdown('<div class="section-panel fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Revenue & Gross Margin Trend</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Daily view of top-line performance and contribution strength.</div>', unsafe_allow_html=True)

    df["date_dt"] = pd.to_datetime(df["date"])
    daily = df.groupby(df["date_dt"].dt.date).agg(
        revenue=("revenue", "sum"),
        gross_margin=("gross_margin", "sum")
    ).reset_index()

    st.line_chart(daily.set_index("date_dt")[["revenue", "gross_margin"]])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-panel fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sample Data (Head)</div>', unsafe_allow_html=True)
    st.dataframe(df.head(50))
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# TAB 2 — Cities
# ----------------------------
with tab2:
    st.markdown('<div class="section-panel fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">City-wise Revenue Breakdown</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Which markets actually drive your top-line?</div>', unsafe_allow_html=True)

    city_rev = df.groupby("city")["revenue"].sum().reset_index()
    st.bar_chart(city_rev, x="city", y="revenue")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# TAB 3 — Marketing
# ----------------------------
with tab3:
    st.markdown('<div class="section-panel fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Marketing Source Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">CAC and ROI by acquisition channel.</div>', unsafe_allow_html=True)

    marketing_stats = df.groupby("marketing_source").agg(
        total_spend=("marketing_cost", "sum"),
        orders=("marketing_cost", "count"),
        revenue=("revenue", "sum"),
        gross_margin=("gross_margin", "sum")
    ).reset_index()

    marketing_stats["CAC"] = marketing_stats["total_spend"] / marketing_stats["orders"]
    marketing_stats["ROI"] = (marketing_stats["gross_margin"] - marketing_stats["total_spend"]) / marketing_stats["total_spend"]

    st.dataframe(marketing_stats)
    st.bar_chart(marketing_stats, x="marketing_source", y="CAC")

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# TAB 4 — Profit + Simulator
# ----------------------------
with tab4:
    st.markdown('<div class="section-panel fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Profit Per Order Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">How many orders are actually making you money?</div>', unsafe_allow_html=True)

    df["profit_per_order"] = df["gross_margin"] - df["marketing_cost"] - df["opex_allocated"]

    # ---------- FAST PROFIT DISTRIBUTION (BUCKETED) ----------
    import numpy as np

    # Limit data size for UI responsiveness
    sample = df["profit_per_order"].dropna()
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)

    # Create buckets instead of raw value counts
    hist, bin_edges = np.histogram(sample, bins=30)
    profit_bins = pd.Series(hist, index=[f"{int(bin_edges[i])} to {int(bin_edges[i+1])}" for i in range(len(hist))])

    st.bar_chart(profit_bins)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-panel fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Scenario Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Tweak AOV, CAC, and OPEX to see per-order profitability.</div>', unsafe_allow_html=True)

    col_sim1, col_sim2, col_sim3 = st.columns(3)
    with col_sim1:
        aov = st.slider("Average Order Value (₹)", 50, 1000, 250)
    with col_sim2:
        cac = st.slider("Customer Acquisition Cost (₹)", 0, 500, 100)
    with col_sim3:
        opex = st.slider("OPEX per order (₹)", 0, 200, 30)

    new_gross_margin = aov * 0.25
    new_profit = new_gross_margin - cac - opex

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Gross Margin per Order", f"₹{new_gross_margin:.2f}")
    with c2:
        st.metric("Profit per Order", f"₹{new_profit:.2f}")

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Auto Insights (Under Tabs)
# ----------------------------
st.markdown('<div class="section-panel fade-in">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Auto Insights</div>', unsafe_allow_html=True)
st.markdown('<div class="section-caption">Quick read on whether this business is healthy or burning cash.</div>', unsafe_allow_html=True)

insights = []

avg_cac = df["marketing_cost"].mean()
if avg_cac > 120:
    insights.append(f"🔴 Average CAC is high at ₹{avg_cac:.2f}. Paid channels need tightening.")
else:
    insights.append(f"🟢 CAC is healthy at ₹{avg_cac:.2f} for this mix.")

negative_orders = (df["profit_per_order"] < 0).mean() * 100
if negative_orders > 25:
    insights.append(f"🔴 {negative_orders:.1f}% of orders are unprofitable. Either pricing, fees, or OPEX is off.")
else:
    insights.append(f"🟢 Only {negative_orders:.1f}% of orders lose money. Unit economics are reasonably under control.")

city_margin = df.groupby("city")["gross_margin"].mean().idxmax()
insights.append(f"🏙️ {city_margin} is your strongest margin city. Consider concentrating marketing there.")

promo_margin = df[df["promo_used"] == True]["gross_margin"].mean()
nonpromo_margin = df[df["promo_used"] == False]["gross_margin"].mean()
if promo_margin < nonpromo_margin * 0.8:
    insights.append("🔴 Promo-driven orders are dragging margins hard. Discounts are too aggressive.")
else:
    insights.append("🟢 Promo impact on gross margin looks acceptable at current levels.")

for tip in insights:
    st.write(tip)

st.markdown('</div>', unsafe_allow_html=True)

 # ===================== DATA DICTIONARY (Standalone Section) =====================
st.markdown('<div class="section-panel">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Data Dictionary (Standalone)</div>', unsafe_allow_html=True)
st.markdown('<div class="section-caption">All dataset fields with meanings and example values.</div>', unsafe_allow_html=True)

dict_df_standalone = pd.DataFrame({
    "Column": df.columns,
    "Meaning": ["_" for _ in df.columns],
    "Example": [df[col].iloc[0] for col in df.columns]
})

st.dataframe(dict_df_standalone)

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# NEW TAB: Advanced KPIs
# ------------------------------------------------------------
if enable_advanced:
    adv_tab1, adv_tab2, adv_tab3, adv_tab4, adv_tab5, adv_tab6 = st.tabs([
        "Pro KPIs", "Cohorts", "Operations", "What-If Lab 2.0", "User Segments", "Data Dictionary"
    ])


if enable_advanced:
    # ===================== PRO KPIs SECTION =====================
    with adv_tab1:
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Advanced KPIs</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Enterprise-level metrics layered on top of your existing KPIs.</div>', unsafe_allow_html=True)

        cac = df["marketing_cost"].mean()

        if user_col is not None:
            repeat_rate = df[user_col].value_counts().mean()
            ltv_simple = df["revenue"].mean() * repeat_rate
            mau = df[user_col].nunique()
            arpu = df["revenue"].sum() / max(mau, 1)
        else:
            repeat_rate = 0
            ltv_simple = df["revenue"].mean()
            mau = df["date_only"].nunique()
            arpu = df["revenue"].mean()

        ltv_cac_ratio = ltv_simple / cac if cac > 0 else 0

        dau = df["date_only"].nunique()
        stickiness = dau / mau if mau > 0 else 0

        st.write("**CAC:**", round(cac, 2))
        st.write("**LTV:**", round(ltv_simple, 2))
        st.write("**LTV/CAC Ratio:**", round(ltv_cac_ratio, 2))
        st.write("**Repeat Purchase Rate:**", round(repeat_rate, 2))
        st.write("**DAU:**", dau)
        st.write("**MAU:**", mau)
        st.write("**Stickiness Ratio:**", round(stickiness, 3))
        st.write("**ARPU:**", round(arpu, 2))

        st.markdown('</div>', unsafe_allow_html=True)

    # ===================== COHORTS TAB =====================
    with adv_tab2:
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Cohort Retention Analysis</div>', unsafe_allow_html=True)

        df["order_date"] = pd.to_datetime(df["date"])

        if user_col is not None:
            df["cohort_week"] = df.groupby(user_col)["order_date"].transform("min").dt.isocalendar().week
            cohort = df.pivot_table(
                index="cohort_week",
                columns=df["order_date"].dt.isocalendar().week,
                values=user_col,
                aggfunc="nunique"
            ).fillna(0)
        else:
            df["cohort_week"] = df["order_date"].dt.isocalendar().week
            cohort = df.pivot_table(
                index="cohort_week",
                columns=df["order_date"].dt.isocalendar().week,
                values="revenue",
                aggfunc="count"
            ).fillna(0)

        st.dataframe(cohort)
        st.line_chart(cohort.T)

        st.markdown('</div>', unsafe_allow_html=True)

    # ===================== OPERATIONS TAB =====================
    with adv_tab3:
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Operational KPIs</div>', unsafe_allow_html=True)

        st.write("Delivery time distribution:")
        if "delivery_time" in df:
            st.bar_chart(df["delivery_time"].value_counts().sort_index())

        st.write("Cancellation reasons:")
        if "cancel_reason" in df:
            st.bar_chart(df["cancel_reason"].value_counts())

        st.markdown('</div>', unsafe_allow_html=True)

    # ===================== WHAT IF LAB =====================
    with adv_tab4:
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">What-If Simulator</div>', unsafe_allow_html=True)

        colA, colB, colC = st.columns(3)
        with colA:
            new_marketing = st.slider("Marketing Spend Increase (%)", 0, 300, 20)
        with colB:
            churn_reduction = st.slider("Churn Reduction (%)", 0, 50, 10)
        with colC:
            conv_rate_increase = st.slider("Conversion Rate Increase (%)", 0, 100, 10)

        baseline_orders = len(df)
        baseline_aov = df["revenue"].mean()
        baseline_gm_rate = 0.25

        new_orders = baseline_orders * (1 + conv_rate_increase/100)
        new_revenue = baseline_aov * new_orders
        new_gross_margin = new_revenue * baseline_gm_rate

        st.write("Projected Revenue:", round(new_revenue, 2))
        st.write("Projected Gross Margin:", round(new_gross_margin, 2))

        st.markdown('</div>', unsafe_allow_html=True)

    # ===================== USER SEGMENTS TAB =====================
    with adv_tab5:
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">User Segments</div>', unsafe_allow_html=True)

        if user_col is not None:
            user_freq = df.groupby(user_col).size()
            st.write("One-time users:", int((user_freq == 1).sum()))
            st.write("Repeat users:", int((user_freq >= 2).sum()))

        st.markdown('</div>', unsafe_allow_html=True)

    # ===================== DATA DICTIONARY TAB =====================
    with adv_tab6:
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Data Dictionary</div>', unsafe_allow_html=True)

        dict_df = pd.DataFrame({
            "Column": df.columns,
            "Example": [df[col].iloc[0] for col in df.columns]
        })

        st.dataframe(dict_df)
        st.markdown('</div>', unsafe_allow_html=True)


if enable_ml:
    # ===================== ML LAB TAB =====================
    ml_tab, = st.tabs(["ML Lab"])

    with ml_tab:
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestRegressor, IsolationForest
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            mean_absolute_error, mean_squared_error
        )

        # Build user-level dataset
        if user_col is None:
            df["_ml_user"] = np.arange(len(df))
            ml_user = "_ml_user"
        else:
            ml_user = user_col

        df["date_dt"] = pd.to_datetime(df["date"])
        cust = df.groupby(ml_user).agg(
            total_orders=("revenue","count"),
            total_revenue=("revenue","sum"),
            avg_basket=("revenue","mean"),
            first_order=("date_dt","min"),
            last_order=("date_dt","max"),
            promo_rate=("promo_used","mean")
        ).reset_index()

        cust["days_since_first"] = (cust["last_order"] - cust["first_order"]).dt.days
        cust["days_since_last"] = (df["date_dt"].max() - cust["last_order"]).dt.days

        # --------------------------------------------------------
        # 1️⃣ CHURN MODEL — PREMIUM CLEAN VERSION
        # --------------------------------------------------------
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Churn Prediction</div>', unsafe_allow_html=True)

        # -------- FIXED: Robust dynamic churn logic --------

        # -------- RFM SCORING (ADDED CLEAN PREMIUM SECTION) --------
        # Recency: days since last order
        cust["R_score"] = pd.qcut(cust["days_since_last"], 4, labels=[4,3,2,1]).astype(int)

        # Frequency: total orders
        cust["F_score"] = pd.qcut(cust["total_orders"].rank(method="first"), 4, labels=[1,2,3,4]).astype(int)

        # Monetary: total revenue
        cust["M_score"] = pd.qcut(cust["total_revenue"].rank(method="first"), 4, labels=[1,2,3,4]).astype(int)

        # Weighted RFM composite
        cust["RFM"] = (cust["R_score"]*0.5) + (cust["F_score"]*0.25) + (cust["M_score"]*0.25)

        # RFM-based churn label before adaptive logic
        cust["rfm_churn"] = (cust["RFM"] <= cust["RFM"].median()).astype(int)

        # -------- HYBRID CHURN MODEL (RFM + Behavioral) --------
        CHURN_DAYS = max(7, int(cust["days_since_last"].quantile(0.70)))

        cust["behavior_churn"] = (
            (cust["days_since_last"] >= CHURN_DAYS) &
            (cust["total_orders"] <= cust["total_orders"].median())
        ).astype(int)

        # Combine RFM churn + behavioral churn
        cust["churned"] = (
            (cust["rfm_churn"] + cust["behavior_churn"]) >= 1
        ).astype(int)

        # Safety fallback
        if cust["churned"].nunique() < 2:
            cust["churned"] = cust["behavior_churn"]

        # Safety check
        if cust["churned"].nunique() < 2:
            st.warning("⚠️ Not enough churn variation in the dataset to train a meaningful model.")
        else:
            churn_features = ["total_orders","total_revenue","avg_basket","promo_rate","days_since_first","days_since_last"]
            X = cust[churn_features].fillna(0)
            y = cust["churned"]

            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
            churn_model = LogisticRegression(max_iter=400)
            churn_model.fit(Xtr, ytr)

            pred = churn_model.predict(Xte)
            st.write(f"Accuracy: {accuracy_score(yte,pred):.3f} | Precision: {precision_score(yte,pred):.3f} | Recall: {recall_score(yte,pred):.3f}")

            cust["churn_probability"] = churn_model.predict_proba(X)[:,1]
            top = cust.sort_values("churn_probability", ascending=False).head(5)[[ml_user,"churn_probability","total_orders","total_revenue"]]
            st.markdown("### 🔥 Highest-Risk Users (Hybrid RFM + Behavior)")
            top_display = top.copy()
            top_display["churn_probability"] = top_display["churn_probability"].round(2)
            st.dataframe(top_display)
        st.markdown('</div>', unsafe_allow_html=True)

        # --------------------------------------------------------
        # 2️⃣ LTV MODEL — PREMIUM CLEAN VERSION
        # --------------------------------------------------------
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">LTV Prediction</div>', unsafe_allow_html=True)

        cust["ltv_actual"] = cust["total_revenue"]
        ltv_features = ["total_orders","avg_basket","promo_rate","days_since_first"]

        Xl = cust[ltv_features].fillna(0)
        yl = cust["ltv_actual"]

        if len(cust) > 5:
            Xtr, Xte, ytr, yte = train_test_split(Xl, yl, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=80, random_state=42)
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)

            st.write(f"MAE: {mean_absolute_error(yte,pred):.2f} | RMSE: {(mean_squared_error(yte,pred)**0.5):.2f}")

            cust["ltv_pred"] = model.predict(Xl)
            st.write("Top Predicted LTV (5)")
            st.dataframe(cust.sort_values("ltv_pred", ascending=False).head(5)[[ml_user,"ltv_actual","ltv_pred"]])
        else:
            st.info("Not enough customers for LTV modelling.")
        st.markdown('</div>', unsafe_allow_html=True)

        # --------------------------------------------------------
        # 3️⃣ REVENUE FORECAST — PROPHET CORPORATE-GRADE VERSION
        # --------------------------------------------------------

        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Revenue Forecast</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Prophet-based corporate forecasting with trend + seasonality.</div>', unsafe_allow_html=True)

        from prophet import Prophet

        # Prepare data
        dailyRevenue = df.groupby(df["date_dt"].dt.date)["revenue"].sum().reset_index()
        dailyRevenue.columns = ["ds", "y"]
        dailyRevenue["ds"] = pd.to_datetime(dailyRevenue["ds"])

        # Forecast horizon
        forecastHorizon = st.slider("Forecast Days", 7, 60, 21)

        # Build Prophet model
        prophetModel = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.3,
            seasonality_prior_scale=10
        )

        prophetModel.fit(dailyRevenue)

        # Future DF
        futureDF = prophetModel.make_future_dataframe(periods=forecastHorizon)

        # Forecast
        forecastDF = prophetModel.predict(futureDF)

        # Display chart (Streamlit native)
        forecastPlot = forecastDF.set_index("ds")[["yhat"]]
        historyPlot = dailyRevenue.set_index("ds")[["y"]]

        mergedForecast = historyPlot.join(forecastPlot, how="outer")

        st.line_chart(mergedForecast)

        st.markdown('</div>', unsafe_allow_html=True)

        # --------------------------------------------------------
        # 4️⃣ ANOMALY DETECTION — CORPORATE-GRADE ENGINE
        # --------------------------------------------------------
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Anomaly Detection</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Multiple detectors (IsolationForest, Z-score, MAD and Prophet interval). Flags are combined for robust alerts.</div>', unsafe_allow_html=True)

        # ensure profit per order exists
        df["profit_per_order"] = df.get("profit_per_order", df.get("gross_margin", 0) - df.get("marketing_cost", 0) - df.get("opex_allocated", 0))

        # build daily operations frame
        daily_ops = df.groupby(df["date_dt"].dt.date).agg(
            revenue=("revenue", "sum"),
            profit=("profit_per_order", "mean"),
            cac=("marketing_cost", "mean")
        ).rename_axis("date_dt").reset_index()

        # Defensive: require at least 10 days for meaningful stats
        if len(daily_ops) < 10:
            st.info("Not enough daily history for corporate-grade anomaly detection (need >= 10 days).")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            import numpy as _np
            from sklearn.ensemble import IsolationForest as _IF
            from prophet import Prophet as _Prophet

            # ---------------- Z-SCORE detector (per metric)
            def zscore_flag(series, thresh=3.0):
                mu = series.mean()
                sigma = series.std(ddof=0)
                if sigma == 0 or _np.isnan(sigma):
                    return _np.zeros(len(series), dtype=bool)
                z = (series - mu) / sigma
                return _np.abs(z) > thresh

            # ---------------- MAD (robust) detector
            def mad_flag(series, thresh=3.5):
                med = series.median()
                dev = _np.abs(series - med)
                mad = _np.median(dev)
                if mad == 0 or _np.isnan(mad):
                    return _np.zeros(len(series), dtype=bool)
                mod_z = 0.6745 * (series - med) / mad
                return _np.abs(mod_z) > thresh

            # ---------------- Isolation Forest
            iso = _IF(contamination=0.06, random_state=42)
            iso_input = daily_ops[["revenue", "profit", "cac"]].fillna(0)
            try:
                iso_pred = iso.fit_predict(iso_input)
                daily_ops["flagIso"] = iso_pred == -1
            except Exception:
                daily_ops["flagIso"] = False

            # ---------------- Z-score & MAD flags
            daily_ops["flagZRevenue"] = zscore_flag(daily_ops["revenue"], thresh=3.0)
            daily_ops["flagZProfit"] = zscore_flag(daily_ops["profit"], thresh=3.0)
            daily_ops["flagZCac"] = zscore_flag(daily_ops["cac"], thresh=3.0)

            daily_ops["flagMadRevenue"] = mad_flag(daily_ops["revenue"], thresh=3.5)
            daily_ops["flagMadProfit"] = mad_flag(daily_ops["profit"], thresh=3.5)
            daily_ops["flagMadCac"] = mad_flag(daily_ops["cac"], thresh=3.5)

            # ---------------- Prophet-based interval breach (revenue only)
            try:
                prophet_df = daily_ops[["date_dt", "revenue"]].rename(columns={"date_dt":"ds","revenue":"y"})
                prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])  # ensure datetime
                m = _Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
                m.fit(prophet_df)
                future = m.make_future_dataframe(periods=0)
                fc = m.predict(future)
                fc = fc.set_index("ds")[ ["yhat","yhat_lower","yhat_upper"] ].rename_axis("date_dt").reset_index()
                # merge on date
                merged = daily_ops.merge(fc, on="date_dt", how="left")
                merged["flagProphet"] = (merged["revenue"] < merged["yhat_lower"]) | (merged["revenue"] > merged["yhat_upper"])
                daily_ops = merged
            except Exception:
                # if prophet fails, mark no prophet flags but continue
                daily_ops["flagProphet"] = False

            # ---------------- Combined alert
            flag_cols = [c for c in daily_ops.columns if str(c).startswith("flag")]

            daily_ops["any_flag"] = daily_ops[flag_cols].any(axis=1)

            flagged = daily_ops[daily_ops["any_flag"]].copy()

            st.write(f"Anomalies: {len(flagged)}")
            if len(flagged) > 0:
                # show the most important columns + which detectors fired
                display_cols = ["date_dt","revenue","profit","cac"] + flag_cols
                st.dataframe(flagged[display_cols].sort_values("date_dt", ascending=False).reset_index(drop=True))

            # Visuals: scaled charts and flagged overlay
            # plot revenue separately (dominant scale) and profit/cac on separate small chart
            st.markdown("**Revenue (with Prophet yhat if available)**")
            try:
                plot_df = daily_ops.set_index("date_dt")[ ["revenue"] ].join(daily_ops.set_index("date_dt")[ ["yhat"] ], how="left")
                st.line_chart(plot_df)
            except Exception:
                st.line_chart(daily_ops.set_index("date_dt")[ ["revenue"] ])

            st.markdown("**Profit & CAC (separate scale)**")
            st.line_chart(daily_ops.set_index("date_dt")[ ["profit","cac"] ])

            st.markdown("**Detector summary**")
            summary = {col: int(daily_ops[col].sum()) for col in flag_cols}
            st.table(pd.DataFrame(list(summary.items()), columns=["Detector","Count"]))

            # provide suggested next steps
            st.markdown("**Suggested next steps (automated)**")
            st.write("• Investigate days where `flagProphet` is true — likely structural change or campaign.")
            st.write("• Investigate `flagIso` days — multivariate oddity (ops/marketing combined).")
            st.write("• Use `flagMad*` for robust outlier filtering when data is heavy-tailed.")

            st.markdown('</div>', unsafe_allow_html=True)

        # --------------------------------------------------------
        # 5️⃣ EXECUTIVE SUMMARY — PREMIUM CLEAN VERSION
        # --------------------------------------------------------
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)

        st.markdown("""
        - Churn model highlights the highest-risk users.
        - LTV model identifies top revenue drivers.
        - Revenue forecast shows stable short-term outlook.
        - Anomaly detection flags operational / marketing spikes.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><br><center><span style='font-size:12px;color:#777;'>Built by <b>Mitaksh</b> · Business Economics Dashboard</span></center>", unsafe_allow_html=True)