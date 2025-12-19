import streamlit as st
import pandas as pd

st.set_page_config(page_title="Business Economics Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/transactions.csv")

df = load_data()

st.title("📊 Business Economics & Insights Dashboard")

st.subheader("Basic Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Revenue", f"₹{df['revenue'].sum():,.0f}")
col2.metric("Total Gross Margin", f"₹{df['gross_margin'].sum():,.0f}")
col3.metric("Average Revenue per Order", f"₹{df['revenue'].mean():.2f}")
col4.metric("Total Transactions", f"{len(df):,}")

st.write("### Preview of Data")
st.dataframe(df.head(20))

