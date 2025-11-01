import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(
    page_title="Tracking Error Lab", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Montserrat font and better styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
        font-weight: 500;
    }
    
    h1, h2, h3 {
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        letter-spacing: 0.015em;
    }
    
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4a90e2;
    }
</style>
""", unsafe_allow_html=True)

# --- Header + links ---
col_title, col_links = st.columns([3, 1])
with col_title:
    st.title("📊 Tracking Error Lab")
    st.markdown("**Interactive tool for exploring tracking error across time horizons**")
with col_links:
    st.markdown(
        "<div style='text-align:right; padding-top: 1.5rem;'>"
        "📚 <a href='https://tejasviswa.github.io/tracking-error-lab/' target='_blank'>Documentation</a><br>"
        "🔗 <a href='https://github.com/TejasViswa/tracking-error-lab' target='_blank'>GitHub</a>"
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# --- Introduction ---
with st.expander("ℹ️ What is Tracking Error?", expanded=False):
    st.markdown("""
    **Tracking error (TE)** measures how much a portfolio's returns deviate from its benchmark over time.
    
    **Active return:** $a_t = r_{p,t} - r_{b,t}$ (portfolio return minus benchmark return)
    
    **Sample tracking error:**
    """)
    st.latex(r"\widehat{TE}=\sqrt{\frac{1}{T-1}\sum_{t=1}^{T}(a_t-\bar a)^2},\quad \bar a=\frac{1}{T}\sum_{t=1}^{T}a_t")
    
    st.markdown("""
    **Annualization:**
    - Daily TE × √252 = Annualized TE
    - Monthly TE × √12 = Annualized TE
    
    **Key insight:** When active returns are autocorrelated (positively or negatively), 
    monthly and daily annualized TE can differ significantly!
    
    📖 Read more in the [Overview](https://tejasviswa.github.io/tracking-error-lab/) | 
    [Intuitive Math](https://tejasviswa.github.io/tracking-error-lab/intuitive_math.html)
    """)

# --- Simulator ---
st.header("🎲 Simulate Daily Active Returns")

st.markdown("""
Simulate an AR(1) process to see how **autocorrelation** affects the relationship between 
daily and monthly tracking error. 

**Positive φ** = Persistent drift (momentum) | **Negative φ** = Mean reversion | **φ ≈ 0** = Random walk
""")

col_param1, col_param2, col_param3 = st.columns(3)
with col_param1:
    T_days = st.slider("Trading days", 100, 2000, 756, 5)
with col_param2:
    sigma_bps = st.slider("Daily active vol (bps)", 1, 50, 12)
with col_param3:
    seed = st.number_input("Random seed", 0, 10_000, 42)

phi = st.slider(
    "AR(1) coefficient φ (autocorrelation)", 
    -0.9, 0.9, 0.2, 0.05,
    help="φ > 0: Persistent drift (momentum) | φ < 0: Mean reversion | φ ≈ 0: No serial correlation"
)

# Simulate AR(1) process
rng = np.random.default_rng(int(seed))
sigma_eps = (sigma_bps/10000.0) * np.sqrt(1 - phi**2) if abs(phi) < 0.99 else (sigma_bps/10000.0)
eps = rng.normal(0, sigma_eps, T_days)
a = np.zeros(T_days)
for t in range(1, T_days):
    a[t] = phi * a[t-1] + eps[t]

dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=T_days)
df = pd.DataFrame({"date": dates, "a": a}).set_index("date")

# Calculate daily TE
te_d = df["a"].std(ddof=1)
te_d_ann = te_d * np.sqrt(252)

# Calculate monthly TE
df_m = df.resample("M").sum()
te_m = df_m["a"].std(ddof=1)
te_m_ann = te_m * np.sqrt(12)

# Calculate ratio
ratio = te_m_ann / te_d_ann if te_d_ann > 0 else 1.0

# Display metrics
col_metric1, col_metric2, col_metric3 = st.columns(3)
with col_metric1:
    st.metric("Daily TE (annualized)", f"{te_d_ann*100:.2f}%")
with col_metric2:
    st.metric("Monthly TE (annualized)", f"{te_m_ann*100:.2f}%")
with col_metric3:
    delta_text = "Higher" if ratio > 1.05 else ("Lower" if ratio < 0.95 else "Similar")
    st.metric(
        "Monthly / Daily Ratio", 
        f"{ratio:.3f}",
        delta=delta_text,
        delta_color="normal" if abs(ratio - 1) < 0.05 else ("off" if ratio < 1 else "normal")
    )

# Interpretation
if ratio > 1.1:
    st.info(f"📈 **Persistent drift detected** (φ={phi:.2f}): Monthly TE is {(ratio-1)*100:.1f}% higher than daily TE predicts. Active returns are positively autocorrelated.")
elif ratio < 0.9:
    st.info(f"📉 **Mean reversion detected** (φ={phi:.2f}): Monthly TE is {(1-ratio)*100:.1f}% lower than daily TE predicts. Active returns are negatively autocorrelated.")
else:
    st.success(f"✅ **Near random walk** (φ={phi:.2f}): Monthly and daily TE are consistent. Minimal serial correlation detected.")

# --- Visualizations ---
tab1, tab2, tab3 = st.tabs(["📈 Daily Active Returns", "📊 Cumulative Drift", "🔍 Autocorrelation"])

with tab1:
    st.markdown("**Daily active returns** show the day-to-day portfolio vs benchmark differences.")
    sim_chart = (
        alt.Chart(df.reset_index())
        .mark_line(strokeWidth=2, color="#4a90e2")
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("a:Q", title="Active Return", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"), 
                alt.Tooltip("a:Q", title="Active Return", format=".5f")
            ],
        )
        .properties(height=400)
    )
    
    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='red', strokeDash=[5, 5]).encode(y='y:Q')
    
    st.altair_chart((sim_chart + zero_line).interactive(), use_container_width=True)

with tab2:
    st.markdown("**Cumulative drift** shows how tracking error accumulates over time. Persistent drift causes larger swings.")
    df_cumulative = df.copy()
    df_cumulative['cumulative'] = df_cumulative['a'].cumsum()
    
    cum_chart = (
        alt.Chart(df_cumulative.reset_index())
        .mark_line(strokeWidth=2.5, color="#e74c3c")
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("cumulative:Q", title="Cumulative Active Return", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"), 
                alt.Tooltip("cumulative:Q", title="Cumulative Return", format=".5f")
            ],
        )
        .properties(height=400)
    )
    
    zero_line_cum = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='gray', strokeDash=[5, 5]).encode(y='y:Q')
    
    st.altair_chart((cum_chart + zero_line_cum).interactive(), use_container_width=True)
    
    st.caption(f"Final cumulative return: {df_cumulative['cumulative'].iloc[-1]:.4f}")

with tab3:
    st.markdown("**Sample autocorrelation function (ACF)** shows correlation at different lags.")
    
    # Calculate sample autocorrelation
    max_lag = min(40, len(df) // 4)
    acf_values = []
    a_centered = df['a'] - df['a'].mean()
    var_a = np.var(a_centered, ddof=1)
    
    for lag in range(max_lag + 1):
        if lag == 0:
            acf_values.append(1.0)
        else:
            cov = np.mean(a_centered[:-lag] * a_centered[lag:])
            acf_values.append(cov / var_a if var_a > 0 else 0)
    
    acf_df = pd.DataFrame({'lag': range(max_lag + 1), 'acf': acf_values})
    
    acf_chart = (
        alt.Chart(acf_df)
        .mark_bar(size=8, color="#27ae60")
        .encode(
            x=alt.X("lag:Q", title="Lag (days)", scale=alt.Scale(domain=[0, max_lag])),
            y=alt.Y("acf:Q", title="Autocorrelation", scale=alt.Scale(domain=[-1, 1])),
            tooltip=[alt.Tooltip("lag:Q", title="Lag"), alt.Tooltip("acf:Q", title="ACF", format=".4f")],
        )
        .properties(height=350)
    )
    
    # Add confidence bands (approximate 95% CI for white noise)
    ci = 1.96 / np.sqrt(len(df))
    ci_df = pd.DataFrame({'y': [ci, -ci, 0]})
    ci_lines = alt.Chart(ci_df).mark_rule(strokeDash=[3, 3], color='red', opacity=0.5).encode(y='y:Q')
    
    st.altair_chart((acf_chart + ci_lines).interactive(), use_container_width=True)
    
    st.caption(f"Lag-1 autocorrelation (estimate of φ): {acf_values[1]:.4f} | True φ: {phi:.4f}")

st.markdown("---")

# --- Upload section ---
st.header("📤 Upload Your Own Data")

st.markdown("""
Upload a CSV file with your portfolio and benchmark returns to calculate tracking error.  
**Required columns:** `date`, `rp` (portfolio return), `rb` (benchmark return)
""")

col_upload, col_freq = st.columns([2, 1])
with col_upload:
    f = st.file_uploader("Upload CSV file", type=["csv"], help="CSV must contain: date, rp, rb columns")
with col_freq:
    freq = st.radio("Data frequency", ["Daily", "Monthly"], horizontal=True, help="Specify the frequency of your returns data")

ann = 252 if freq == "Daily" else 12

if f is not None:
    raw = pd.read_csv(f)
    lower = {c.lower(): c for c in raw.columns}
    try:
        dcol, rpcol, rbcol = lower["date"], lower["rp"], lower["rb"]
    except KeyError:
        st.error("Missing required columns: date, rp, rb")
        st.stop()

    dfu = raw.rename(columns={dcol: "date", rpcol: "rp", rbcol: "rb"})
    dfu["date"] = pd.to_datetime(dfu["date"])
    dfu = dfu.sort_values("date").set_index("date")
    dfu["a"] = dfu["rp"].astype(float) - dfu["rb"].astype(float)

    te = (dfu["a"] - dfu["a"].mean()).std(ddof=1)
    te_ann = te * np.sqrt(ann)

    c3, c4 = st.columns(2)
    with c3:
        st.metric("Periodic TE", f"{te*100:.2f}%")
    with c4:
        st.metric("Annualized TE", f"{te_ann*100:.2f}%")

    # Visualization tabs for uploaded data
    st.subheader(f"📊 Your Data Analysis ({freq})")
    
    tab_upload1, tab_upload2 = st.tabs(["Time Series", "Statistics"])
    
    with tab_upload1:
        up_chart = (
            alt.Chart(dfu.reset_index())
            .mark_line(strokeWidth=2, color="#4a90e2")
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("a:Q", title="Active Return", scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"), 
                    alt.Tooltip("a:Q", title="Active Return", format=".5f")
                ],
            )
            .properties(height=400)
        )
        
        zero_line_upload = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='red', strokeDash=[5, 5]).encode(y='y:Q')
        st.altair_chart((up_chart + zero_line_upload).interactive(), use_container_width=True)
    
    with tab_upload2:
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Mean Active Return", f"{dfu['a'].mean()*100:.3f}%")
        with col_stat2:
            st.metric("Min Active Return", f"{dfu['a'].min()*100:.3f}%")
        with col_stat3:
            st.metric("Max Active Return", f"{dfu['a'].max()*100:.3f}%")
        
        st.markdown("**Distribution of Active Returns:**")
        hist_data = pd.DataFrame({'a': dfu['a']})
        hist_chart = (
            alt.Chart(hist_data)
            .mark_bar(color="#27ae60")
            .encode(
                alt.X("a:Q", bin=alt.Bin(maxbins=50), title="Active Return"),
                alt.Y("count()", title="Frequency"),
                tooltip=[alt.Tooltip("count()", title="Count")]
            )
            .properties(height=300)
        )
        st.altair_chart(hist_chart, use_container_width=True)

    if freq == "Daily":
        st.markdown("---")
        st.subheader("📅 Daily → Monthly Aggregation")
        
        # Aggregate daily to monthly (sum of arithmetic daily returns per month)
        dfm2 = dfu["a"].resample("M").sum().to_frame("a_m")
        te_m2 = (dfm2["a_m"] - dfm2["a_m"].mean()).std(ddof=1)
        te_m2_ann = te_m2 * np.sqrt(12)
        
        ratio_uploaded = te_m2_ann / te_ann if te_ann > 0 else 1.0

        col_agg1, col_agg2 = st.columns(2)
        with col_agg1:
            st.metric("Monthly TE (from daily)", f"{te_m2_ann*100:.2f}%")
        with col_agg2:
            st.metric("Ratio (Monthly/Daily)", f"{ratio_uploaded:.3f}")
        
        if ratio_uploaded > 1.1:
            st.info(f"📈 Monthly TE is {(ratio_uploaded-1)*100:.1f}% higher → suggests positive autocorrelation (persistent drift)")
        elif ratio_uploaded < 0.9:
            st.info(f"📉 Monthly TE is {(1-ratio_uploaded)*100:.1f}% lower → suggests negative autocorrelation (mean reversion)")
        else:
            st.success("✅ Monthly and daily TE are consistent → minimal serial correlation")

        # Altair bar chart for monthly aggregated active returns
        monthly_chart = (
            alt.Chart(dfm2.reset_index())
            .mark_bar(color="#9b59b6")
            .encode(
                x=alt.X("date:T", title="Month"),
                y=alt.Y("a_m:Q", title="Monthly Active Return"),
                tooltip=[
                    alt.Tooltip("date:T", title="Month", format="%Y-%m"), 
                    alt.Tooltip("a_m:Q", title="Active Return", format=".5f")
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(monthly_chart.interactive(), use_container_width=True)

# --- Sidebar ---
st.sidebar.title("📚 Learn More")

st.sidebar.markdown("""
### Documentation
- [Overview](https://tejasviswa.github.io/tracking-error-lab/)
- [Intuitive Math](https://tejasviswa.github.io/tracking-error-lab/intuitive_math.html)
- [Technical Math](https://tejasviswa.github.io/tracking-error-lab/technical_math.html)

### External Links
- [GitHub Repository](https://github.com/TejasViswa/tracking-error-lab)

---

### 💡 Quick Tips

**Positive φ (e.g., 0.3-0.5)**  
Momentum/growth portfolios - Tech, FAANG  
→ Monthly TE > Daily TE

**Negative φ (e.g., -0.2 to -0.4)**  
Mean-reverting - Energy, Value stocks  
→ Monthly TE < Daily TE

**φ ≈ 0**  
Random walk - ESG screens, broad index  
→ Monthly TE ≈ Daily TE

---

### 🎯 Preset Scenarios

Try these φ values:
- **+0.5** → Strong momentum
- **+0.2** → Mild persistence
- **0.0** → Random walk
- **-0.2** → Mild reversion
- **-0.5** → Strong reversion
""")
