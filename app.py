import pandas as pd
import numpy as np
import pickle
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Churn Risk Dashboard", layout="wide")

st.markdown("""
<style>
    body, .stApp { background: #fff; color: #222; font-family: sans-serif; }
    h1 { border-bottom: 2px solid #000; padding-bottom: 6px; }
   .kpi-box {
    border-radius: 12px;
    padding: 20px;
    background: white;
    box-shadow: 0 4px 10px rgba(0,0,0,.08);
    text-align:center;
}
    .kpi-num { font-size: 2rem; font-weight: bold; }
    .kpi-label { font-size: 0.85rem; color: #555; }
    .risk-high { color: #c00; font-weight: bold; }
    .risk-med  { color: #b87000; font-weight: bold; }
    .risk-low  { color: #007a00; font-weight: bold; }
    hr { border: none; border-top: 1px solid #ddd; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model   = pickle.load(open("model.pkl",   "rb"))
    scaler  = pickle.load(open("scaler.pkl",  "rb"))
    encoder = pickle.load(open("encoder.pkl", "rb"))
    return model, scaler, encoder

@st.cache_data
def load_data():
    flight  = pd.read_csv("Customer Flight Activity.csv")
    loyalty = pd.read_csv("Customer Loyalty History.csv")

    df = pd.merge(flight, loyalty, on="Loyalty Number", how="left")
    df["Churn"] = df["Cancellation Year"].notnull().astype(int)
    df["Salary"] = df["Salary"].fillna(df["Salary"].median())

    cdf = df.groupby("Loyalty Number").agg(
        Total_Flights      = ("Total Flights",       "sum"),
        Distance           = ("Distance",            "sum"),
        Points_Accumulated = ("Points Accumulated",  "sum"),
        Points_Redeemed    = ("Points Redeemed",     "sum"),
        Salary             = ("Salary",              "first"),
        CLV                = ("CLV",                 "first"),
        Gender             = ("Gender",              "first"),
        Education          = ("Education",           "first"),
        Marital_Status     = ("Marital Status",      "first"),
        Loyalty_Card       = ("Loyalty Card",        "first"),
        Enrollment_Type    = ("Enrollment Type",     "first"),
        Province           = ("Province",            "first"),
        Churn              = ("Churn",               "max"),
    ).reset_index()

    cdf["Redemption_Rate"]       = cdf["Points_Redeemed"]    / (cdf["Points_Accumulated"] + 1)
    cdf["Avg_Distance_Per_Flight"]= cdf["Distance"]          / (cdf["Total_Flights"] + 1)
    cdf["Points_Per_Flight"]      = cdf["Points_Accumulated"] / (cdf["Total_Flights"] + 1)
    cdf["CLV_Per_Flight"]         = cdf["CLV"]               / (cdf["Total_Flights"] + 1)

    return cdf

@st.cache_data
def predict(_cdf, _model, _scaler, _encoder):
    X = pd.DataFrame({
        "Gender":                  _cdf["Gender"],
        "Education":               _cdf["Education"],
        "Marital Status":          _cdf["Marital_Status"],
        "Loyalty Card":            _cdf["Loyalty_Card"],
        "Enrollment Type":         _cdf["Enrollment_Type"],
        "Total Flights":           _cdf["Total_Flights"],
        "Distance":                _cdf["Distance"],
        "Points Accumulated":      _cdf["Points_Accumulated"],
        "Points Redeemed":         _cdf["Points_Redeemed"],
        "Salary":                  _cdf["Salary"],
        "CLV":                     _cdf["CLV"],
        "Redemption Rate":         _cdf["Redemption_Rate"],
        "Avg Distance Per Flight": _cdf["Avg_Distance_Per_Flight"],
        "Points Per Flight":       _cdf["Points_Per_Flight"],
        "CLV Per Flight":          _cdf["CLV_Per_Flight"],
    })
    Xe = _encoder.transform(X)
    Xs = _scaler.transform(Xe)

    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(Xs)[:, 1]
    else:
        df_score = _model.decision_function(Xs)
        proba = (df_score - df_score.min()) / (df_score.max() - df_score.min())

    return proba

def risk_label(p):
    if p >= 0.65: return "High"
    if p >= 0.35: return "Medium"
    return "Low"

def action(row):
    if row["Risk_Level"] == "High" and row["Loyalty_Card"] == "Aurora":
        return "Personal call — 48 hrs"
    if row["Risk_Level"] == "High":
        return "Win-back email now"
    if row["Risk_Level"] == "Medium":
        return "Promotional offer email"
    return "Monitor monthly"

st.title("✈ Customer Churn Risk Dashboard")
st.caption("Loyalty programme")
st.markdown("<hr>", unsafe_allow_html=True)

st.info(
    f"""
📊 Dashboard Summary

• {at_risk:,} customers are High Risk

• ${clv_stake:,.0f} CLV currently at risk

• Historical churn rate: {churn_rate:.1f}%
"""
)

st.subheader("Filter Customers")
try:
    model, scaler, encoder = load_artifacts()
    cdf = load_data()
    proba = predict(cdf, model, scaler, encoder)

    cdf["Churn_Probability"] = proba
    cdf["Risk_Level"]        = cdf["Churn_Probability"].apply(risk_label)
    cdf["Recommended_Action"]= cdf.apply(action, axis=1)

    total      = len(cdf)
    at_risk    = (cdf["Risk_Level"] == "High").sum()
    clv_stake  = cdf.loc[cdf["Risk_Level"] == "High", "CLV"].sum()
    churn_rate = cdf["Churn"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    for col, num, label in [
        (c1, f"{total:,}",        "Total Members"),
        (c2, f"{at_risk:,}",      "High Risk Members"),
        (c3, f"${clv_stake/1e6:.1f}M", "CLV at Stake (High Risk)"),
        (c4, f"{churn_rate:.1f}%", "Historical Churn Rate"),
    ]:
        col.markdown(
            f'<div class="kpi-box"><div class="kpi-num">{num}</div>'
            f'<div class="kpi-label">{label}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.sidebar.header("Filters")

card_options = ["All"] + sorted(
    cdf["Loyalty_Card"].dropna().unique().tolist()
)

risk_options = ["All", "High", "Medium", "Low"]

card_filter = st.sidebar.selectbox(
    "Card Tier",
    card_options
)

risk_filter = st.sidebar.selectbox(
    "Risk Level",
    risk_options
)

min_clv = st.sidebar.number_input(
    "Minimum CLV ($)",
    min_value=0,
    value=0,
    step=500
)

search = st.sidebar.text_input(
    "Search Loyalty Number"
)

   if search:
    filtered = filtered[
        filtered["Loyalty Number"]
        .astype(str)
        .str.contains(search)
    ]
    if card_filter != "All":
        filtered = filtered[filtered["Loyalty_Card"] == card_filter]
    if risk_filter != "All":
        filtered = filtered[filtered["Risk_Level"] == risk_filter]
    filtered = filtered[filtered["CLV"] >= min_clv]

    st.markdown(f"**{len(filtered):,} customers match your filters**")

risk_counts = (
    filtered["Risk_Level"]
    .value_counts()
    .reset_index()
)

risk_counts.columns = [
    "Risk Level",
    "Customers"
]

fig = px.pie(
    risk_counts,
    names="Risk Level",
    values="Customers",
    title="Customer Risk Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("Customer Details")

selected_customer = st.selectbox(
    "Select Customer",
    filtered["Loyalty Number"]
)

customer = filtered[
    filtered["Loyalty Number"]
    == selected_customer
].iloc[0]

d1, d2, d3 = st.columns(3)

d1.metric(
    "Churn Probability",
    f"{customer['Churn_Probability']:.1%}"
)

d2.metric(
    "Customer Lifetime Value",
    f"${customer['CLV']:,.0f}"
)

d3.metric(
    "Flights",
    int(customer["Total_Flights"])
)

st.success(
    f"Recommended Action: {customer['Recommended_Action']}"
)

st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Customer List")

    display = filtered[[
        "Loyalty Number", "Loyalty_Card", "Province",
        "CLV", "Total_Flights", "Churn_Probability", "Risk_Level", "Recommended_Action"
    ]].copy()

    display.columns = [
        "Loyalty #", "Card Tier", "Province",
        "CLV ($)", "Total Flights", "Churn Prob", "Risk Level", "Action"
    ]
    display["CLV ($)"]     = display["CLV ($)"].round(0).astype(int)
    display["Churn Prob"]  = (display["Churn Prob"] * 100).round(1).astype(str) + "%"
    display = display.sort_values("CLV ($)", ascending=False).reset_index(drop=True)

    def color_risk(val):
        if val == "High":   return "color: #c00; font-weight: bold"
        if val == "Medium": return "color: #b87000; font-weight: bold"
        return "color: #007a00; font-weight: bold"

    st.dataframe(
        display.style.map(color_risk, subset=["Risk Level"]),
        use_container_width=True,
        height=420
    )

    csv = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download filtered list as CSV",
        data=csv,
        file_name="churn_risk_list.csv",
        mime="text/csv"
    )

except FileNotFoundError as e:
    st.error(f"Missing file: {e}\n\nMake sure model.pkl, scaler.pkl, encoder.pkl and both CSV files are in the same folder as app.py.")