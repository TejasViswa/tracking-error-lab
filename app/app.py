import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(
    page_title="Tracking Error Lab - Interactive Simulator", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple, clean CSS - only essentials
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
    
    /* Font */
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Clean metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>

<!-- Privacy-friendly analytics (no cookies, no personal data) -->
<script src="https://cdn.counter.dev/script.js" data-id="a1415d18-b90b-45df-ac84-45ddeb23fa8f" data-utcoffset="-6"></script>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Interactive Simulator")
    st.markdown("Configure parameters and data sources for tracking error analysis.")
    
    st.divider()
    
    st.markdown("##### Data Source")
    analysis_mode = st.radio(
        "Choose data source",
        ["Simulated Portfolios", "Upload Your Data", "Compare Both"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.divider()
    
    st.markdown("##### Documentation")
    st.page_link("https://tejasviswa.github.io/tracking-error-lab/", label="Overview")
    st.page_link("https://tejasviswa.github.io/tracking-error-lab/intuitive_math.html", label="Intuitive Math")
    st.page_link("https://tejasviswa.github.io/tracking-error-lab/technical_math.html", label="Technical Details")
    
    st.divider()
    
    st.markdown("##### Resources")
    st.page_link("https://github.com/TejasViswa/tracking-error-lab", label="Source Code")
    
    st.divider()
    
    st.markdown("##### Contact")
    st.page_link("https://www.linkedin.com/in/tejasviswa/", label="LinkedIn")
    st.page_link("https://github.com/TejasViswa/", label="GitHub")
    st.markdown("[Email](mailto:tejasviswa@gmail.com)")

# Main content
st.title("Tracking Error Lab: Interactive Simulator")
st.markdown("Explore how autocorrelation in active returns affects tracking error estimation across different time horizons.")

st.divider()

# Introduction
with st.expander("About This Tool"):
    st.markdown("""
    This interactive tool demonstrates how **autocorrelation** in active returns affects the relationship 
    between daily and monthly tracking error estimates.
    
    **Key Concepts:**
    - When active returns are **positively correlated** (momentum), monthly TE is higher than daily TE predicts
    - When active returns are **negatively correlated** (mean reversion), monthly TE is lower than daily TE predicts
    - When active returns are **uncorrelated** (random walk), monthly and daily TE estimates align
    
    Read the full mathematical treatment in the [documentation pages](https://tejasviswa.github.io/tracking-error-lab/).
    """)

# Educational content on portfolio behaviors
with st.expander("Understanding Portfolio Behaviors: Momentum, Random Walk, and Mean Reversion"):
    st.markdown("### Three Types of Portfolio Behavior")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Persistent Drift (Momentum)")
        st.markdown("""
        **What it means:** Yesterday's winners keep winning. Trends persist over time.
        
        **Why it happens:**
        - Momentum strategies betting on recent winners
        - Concentrated positions in trending sectors
        - Slow rebalancing allowing positions to drift
        
        **Real examples:**
        - Tech-heavy growth fund (2020-2021)
        - Momentum ETF (MTUM)
        - Concentrated sector bets
        
        **Impact:** Monthly TE can be **15-40% higher** than daily TE predicts
        """)
    
    with col2:
        st.markdown("#### Random Walk")
        st.markdown("""
        **What it means:** Past performance doesn't predict future performance. Each day is independent.
        
        **Why it happens:**
        - Well-diversified portfolios
        - Frequent algorithmic rebalancing
        - Many small, uncorrelated bets
        
        **Real examples:**
        - S&P 500 index fund
        - Enhanced index with small tilts
        - Balanced multi-factor portfolio
        
        **Impact:** Monthly TE **matches** daily TE × √21 (textbook case)
        """)
    
    with col3:
        st.markdown("#### Mean Reversion")
        st.markdown("""
        **What it means:** What goes up tends to come down. Returns oscillate around zero.
        
        **Why it happens:**
        - Contrarian/value strategies
        - Active rebalancing (trim winners, buy losers)
        - Pairs trading strategies
        
        **Real examples:**
        - Deep value fund
        - Energy sector (boom-bust cycles)
        - Rebalanced 60/40 portfolio
        
        **Impact:** Monthly TE can be **10-30% lower** than daily TE predicts
        """)
    
    st.divider()
    
    st.markdown("### Real-World Examples")
    
    st.markdown("""
    Most portfolios exhibit some combination of these behaviors:
    
    - **Tech fund (2021)**: Strong momentum (φ ≈ +0.4 to +0.6)
    - **Diversified equity fund**: Near random walk (φ ≈ -0.1 to +0.1)
    - **Value fund**: Mild mean reversion (φ ≈ -0.2 to -0.3)
    
    **Key insight:** Standard TE calculations (daily TE × √252) only work for random walks. 
    If your portfolio has momentum or mean reversion, you need different methods like AR(1) formulas or Newey-West.
    """)
    
    st.info("💡 Upload your portfolio data to discover which behavior type you have!")

# --- Portfolio Upload Section ---
uploaded_portfolio_data = None
uploaded_portfolio_name = None

if analysis_mode in ["Upload Your Data", "Compare Both"]:
    st.subheader("Upload Your Portfolio Data")
    
    st.info("""
    **Upload Active Returns Data**: Upload a CSV or Excel file containing your portfolio's active returns 
    (portfolio return minus benchmark return). The file should have two columns: Date and Active Return 
    in decimal format (e.g., 0.0015 for 15 basis points).
    """)
    
    col_upload, col_example = st.columns([2, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose CSV or Excel file",
            type=["csv", "xlsx", "xls"]
        )
    
    with col_example:
        st.markdown("**Example format:**")
        st.code("""Date,Active Return
2023-01-03,0.0012
2023-01-04,-0.0008
2023-01-05,0.0015""", language="csv")
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
            
            with st.expander("Preview uploaded data"):
                st.dataframe(df_upload.head(10), use_container_width=True)
            
            st.markdown("**Map your columns:**")
            col_date, col_return, col_name = st.columns(3)
            
            with col_date:
                date_column = st.selectbox("Date column", options=df_upload.columns.tolist(), index=0)
            
            with col_return:
                return_column = st.selectbox("Active Return column", options=df_upload.columns.tolist(), 
                                            index=min(1, len(df_upload.columns) - 1))
            
            with col_name:
                uploaded_portfolio_name = st.text_input("Portfolio name", value="My Portfolio")
            
            df_clean = df_upload[[date_column, return_column]].copy()
            df_clean.columns = ["date", "active_return"]
            df_clean["date"] = pd.to_datetime(df_clean["date"])
            df_clean = df_clean.sort_values("date").reset_index(drop=True)
            df_clean["active_return"] = pd.to_numeric(df_clean["active_return"], errors="coerce")
            df_clean = df_clean.dropna()
            
            if len(df_clean) < 50:
                st.error("Need at least 50 observations for meaningful analysis.")
            else:
                uploaded_portfolio_data = df_clean["active_return"].values
                
                st.success(f"Loaded {len(df_clean)} observations from {df_clean['date'].min().strftime('%Y-%m-%d')} to {df_clean['date'].max().strftime('%Y-%m-%d')}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Observations", f"{len(df_clean)}")
                col2.metric("Mean Return", f"{np.mean(uploaded_portfolio_data)*100:.2f}%")
                col3.metric("Std Dev", f"{np.std(uploaded_portfolio_data, ddof=1)*100:.2f}%")
                col4.metric("Range", f"{np.min(uploaded_portfolio_data)*100:.2f}% / {np.max(uploaded_portfolio_data)*100:.2f}%")
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    st.divider()

# --- Regime Configuration ---
regime_presets = {
    "Persistent Drift (φ=+0.5)": {"phi": 0.5, "color": "#e74c3c", "description": "Momentum/trending behavior"},
    "Random Walk (φ=0)": {"phi": 0.0, "color": "#3498db", "description": "No serial correlation (textbook case)"},
    "Mean Reversion (φ=-0.45)": {"phi": -0.45, "color": "#27ae60", "description": "Contrarian/value behavior"}
}

selected_regimes = []
if analysis_mode in ["Simulated Portfolios", "Compare Both"]:
    st.subheader("Configure Simulated Portfolios")
    
    st.markdown("""
    **Choose portfolio behaviors to simulate:**
    - **Persistent Drift**: Like a tech/growth fund that trends (momentum)
    - **Random Walk**: Like a diversified index tracker (no correlation)
    - **Mean Reversion**: Like a value fund that oscillates (contrarian)
    """)
    
    selected_regimes = st.multiselect(
        "Select regimes:",
        options=list(regime_presets.keys()),
        default=list(regime_presets.keys()),
        help="Choose one or more portfolio behavior types to analyze"
    )
    
    if not selected_regimes and analysis_mode != "Upload Your Data":
        st.warning("Please select at least one regime to continue.")
        st.stop()
    
    if selected_regimes:
        st.markdown("**Simulation parameters:**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            T_days = st.slider("Trading days", 252, 2520, 756, 252)
        with col2:
            target_annual_te_bps = st.slider("Target annual TE (bps)", 100, 1500, 500, 50)
        with col3:
            seed = st.number_input("Random seed", 0, 10_000, 42)
        
        with st.expander("Advanced: Customize φ (autocorrelation) values"):
            st.markdown("Override the default autocorrelation coefficients for each regime:")
            custom_phi = {}
            cols = st.columns(len(selected_regimes))
            for i, regime in enumerate(selected_regimes):
                with cols[i]:
                    default_phi = regime_presets[regime]["phi"]
                    custom_phi[regime] = st.slider(
                        f"{regime.split('(')[0].strip()}",
                        -0.7, 0.7, default_phi, 0.05,
                        key=f"phi_{regime}"
                    )
    else:
        custom_phi = {}
        T_days = 756
        target_annual_te_bps = 500
        seed = 42
else:
    T_days = 756
    target_annual_te_bps = 500
    seed = 42
    custom_phi = {}

st.divider()

# --- Helper Functions ---
def generate_realistic_ar1(phi, target_annual_te_bps, T_days, seed):
    """Generate realistic AR(1) active returns."""
    rng = np.random.default_rng(int(seed))
    target_annual_te = target_annual_te_bps / 10000.0
    target_daily_std = target_annual_te / np.sqrt(252)
    
    if abs(phi) < 0.99:
        sigma_eps = target_daily_std * np.sqrt(1 - phi**2)
    else:
        sigma_eps = target_daily_std
    
    eps = rng.normal(0, sigma_eps, T_days)
    a = np.zeros(T_days)
    a[0] = eps[0]
    
    for t in range(1, T_days):
        a[t] = phi * a[t-1] + eps[t]
    
    n_jumps = max(1, int(T_days / 252 * 2))
    jump_indices = rng.choice(T_days, size=n_jumps, replace=False)
    jump_sizes = rng.normal(0, target_daily_std * 3, n_jumps)
    a[jump_indices] += jump_sizes
    
    vol_regime = np.ones(T_days)
    regime_changes = sorted(rng.choice(T_days, size=max(1, T_days // 126), replace=False))
    for i, change_pt in enumerate(regime_changes):
        regime_factor = rng.uniform(0.7, 1.3)
        end_pt = regime_changes[i+1] if i+1 < len(regime_changes) else T_days
        vol_regime[change_pt:end_pt] = regime_factor
    a = a * vol_regime
    
    return a

def calculate_ar1_theoretical_te(phi, sigma_daily, D=21):
    """Calculate theoretical monthly TE using AR(1) closed-form formula."""
    if abs(phi) < 0.99:
        gamma_0 = sigma_daily**2
        if abs(phi) < 1e-6:
            te_m_theoretical = np.sqrt(D * gamma_0)
        else:
            sum_term = (1 - (1 - phi**D) / (D * (1 - phi)))
            factor = 1 + 2 * phi / (1 - phi) * sum_term
            te_m_theoretical = np.sqrt(D * gamma_0 * factor)
        return te_m_theoretical
    else:
        return np.sqrt(D) * sigma_daily

def newey_west_lrv(a, L=None):
    """Calculate Newey-West long-run variance estimator with Bartlett kernel."""
    T = len(a)
    a_demean = a - np.mean(a)
    
    if L is None:
        L = int(np.floor(4 * (T/100)**(2/9)))
    
    gamma_0 = np.dot(a_demean, a_demean) / T
    lrv = gamma_0
    
    for h in range(1, min(L+1, T)):
        gamma_h = np.dot(a_demean[:-h], a_demean[h:]) / T
        weight = 1 - h / (L + 1)
        lrv += 2 * weight * gamma_h
    
    return max(lrv, 0)

def estimate_ar1_parameters(a):
    """Estimate AR(1) parameters using OLS."""
    a_demean = a - np.mean(a)
    y = a_demean[1:]
    X = a_demean[:-1]
    
    phi_hat = np.dot(X, y) / np.dot(X, X)
    phi_hat = np.clip(phi_hat, -0.99, 0.99)
    
    residuals = y - phi_hat * X
    sigma_eps = np.std(residuals, ddof=1)
    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return phi_hat, sigma_eps, r_squared

# --- Analysis ---
st.subheader("Results & Analysis")

if not selected_regimes and uploaded_portfolio_data is None:
    st.info("Configure data source in the sidebar to begin analysis.")
    st.stop()

all_data = {}
all_metrics = []

# Analyze uploaded portfolio
if uploaded_portfolio_data is not None:
    st.markdown("#### Your Portfolio Analysis")
    
    a_uploaded = uploaded_portfolio_data
    T_uploaded = len(a_uploaded)
    
    te_daily_up = np.std(a_uploaded, ddof=1)
    te_daily_ann_up = te_daily_up * np.sqrt(252)
    
    days_per_month = 21
    n_months = T_uploaded // days_per_month
    if n_months >= 2:
        monthly_returns_up = [np.sum(a_uploaded[i*days_per_month:(i+1)*days_per_month]) 
                             for i in range(n_months)]
        te_monthly_up = np.std(monthly_returns_up, ddof=1)
        te_monthly_ann_up = te_monthly_up * np.sqrt(12)
    else:
        te_monthly_ann_up = te_daily_ann_up
    
    phi_estimated, sigma_eps_estimated, r_squared = estimate_ar1_parameters(a_uploaded)
    te_monthly_ar1_up = calculate_ar1_theoretical_te(phi_estimated, te_daily_up, D=21)
    te_monthly_ar1_ann_up = te_monthly_ar1_up * np.sqrt(12)
    
    lrv_nw_up = newey_west_lrv(a_uploaded, L=int(np.floor(4 * (T_uploaded/100)**(2/9))))
    te_annual_nw_up = np.sqrt(252 * lrv_nw_up)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated φ (AR(1))", f"{phi_estimated:+.3f}")
    col2.metric("AR(1) R²", f"{r_squared:.3f}")
    if phi_estimated > 0.1:
        behavior = "Persistent Drift"
    elif phi_estimated < -0.1:
        behavior = "Mean Reversion"
    else:
        behavior = "Random Walk"
    col3.metric("Detected Behavior", behavior)
    
    all_data[uploaded_portfolio_name] = a_uploaded
    all_metrics.append({
        "Portfolio": f"{uploaded_portfolio_name}",
        "φ": f"{phi_estimated:+.2f}",
        "Daily TE (ann)": f"{te_daily_ann_up*100:.2f}%",
        "Monthly TE (ann)": f"{te_monthly_ann_up*100:.2f}%",
        "AR(1) Formula": f"{te_monthly_ar1_ann_up*100:.2f}%",
        "Newey-West": f"{te_annual_nw_up*100:.2f}%",
        "Ratio": f"{te_monthly_ann_up/te_daily_ann_up:.3f}",
        "Effect": f"{(te_monthly_ann_up/te_daily_ann_up - 1)*100:+.1f}%"
    })
    
    st.divider()

# Simulated regimes
if selected_regimes:
    st.markdown("#### Simulated Regimes Analysis")

dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=T_days)

for regime in selected_regimes:
    phi = custom_phi.get(regime, regime_presets[regime]["phi"])
    regime_seed = seed + list(regime_presets.keys()).index(regime)
    a = generate_realistic_ar1(phi, target_annual_te_bps, T_days, regime_seed)
    all_data[regime] = a
    
    te_daily = np.std(a, ddof=1)
    te_daily_ann = te_daily * np.sqrt(252)
    
    df_temp = pd.DataFrame({"date": dates, "return": a}).set_index("date")
    monthly_returns = df_temp["return"].resample("M").sum()
    te_monthly = np.std(monthly_returns, ddof=1)
    te_monthly_ann = te_monthly * np.sqrt(12)
    
    te_monthly_ar1 = calculate_ar1_theoretical_te(phi, te_daily, D=21)
    te_monthly_ar1_ann = te_monthly_ar1 * np.sqrt(12)
    
    lrv_nw = newey_west_lrv(a, L=int(np.floor(4 * (T_days/100)**(2/9))))
    te_annual_nw = np.sqrt(252 * lrv_nw)
    
    all_metrics.append({
        "Portfolio": regime,
        "φ": f"{phi:+.2f}",
        "Daily TE (ann)": f"{te_daily_ann*100:.2f}%",
        "Monthly TE (ann)": f"{te_monthly_ann*100:.2f}%",
        "AR(1) Formula": f"{te_monthly_ar1_ann*100:.2f}%",
        "Newey-West": f"{te_annual_nw*100:.2f}%",
        "Ratio": f"{te_monthly_ann/te_daily_ann:.3f}",
        "Effect": f"{(te_monthly_ann/te_daily_ann - 1)*100:+.1f}%"
    })

# Display metrics
st.markdown("**Comparison of Estimation Methods**")

col_explain1, col_explain2 = st.columns(2)

with col_explain1:
    with st.expander("Understanding the metrics"):
        st.markdown("""
        - **Daily TE (ann)**: Empirical daily std × √252
        - **Monthly TE (ann)**: Empirical monthly std × √12
        - **AR(1) Formula**: Theoretical prediction using closed-form solution
        - **Newey-West**: Robust long-run variance estimator
        - **Ratio**: Monthly TE / Daily TE
        - **Effect**: Percentage difference
        """)

with col_explain2:
    with st.expander("Mathematical methods explained"):
        st.markdown("""
        **1. Square-Root-of-Time Rule**
        
        Assumes i.i.d. returns (no autocorrelation):
        
        TE_monthly,ann = TE_daily × √252
        
        Valid when φ = 0 (random walk)
        
        ---
        
        **2. AR(1) Closed-Form Solution**
        
        Assumes AR(1) process: a_t = φa_(t-1) + ε_t
        
        Exact formula accounts for autocorrelation structure.
        Valid when returns follow first-order autoregressive process.
        
        ---
        
        **3. Newey-West Long-Run Variance**
        
        Robust approach with no parametric assumptions.
        Uses Bartlett kernel weights to estimate long-run variance.
        Most reliable for real data with unknown autocorrelation.
        
        See [Technical Math](https://tejasviswa.github.io/tracking-error-lab/technical_math.html) for full derivations.
        """)

metrics_df = pd.DataFrame(all_metrics)
st.dataframe(metrics_df, use_container_width=True, hide_index=True)

st.markdown("**Key Insights:**")
for idx, row in metrics_df.iterrows():
    portfolio_name = row['Portfolio']
    effect = row['Effect']
    phi_val = float(row['φ'])
    
    # Add interpretation
    if phi_val > 0.15:
        interpretation = " → Momentum behavior: trends persist, risk compounds"
    elif phi_val < -0.15:
        interpretation = " → Mean reversion: returns oscillate, risk dampens over time"
    else:
        interpretation = " → Random walk: textbook case, square-root-of-time rule applies"
    
    st.markdown(f"• **{portfolio_name}**: Monthly TE is **{effect}** compared to daily TE prediction{interpretation}")

with st.expander("Interpreting your results"):
    st.markdown("""
    **If your Effect is positive (+10% to +40%):**
    - Momentum behavior: trends persist
    - Monthly risk is **higher** than daily risk suggests
    - Consider using Newey-West or monthly TE for more accurate estimates
    
    **If your Effect is near zero (-5% to +5%):**
    - Random walk behavior: no serial correlation
    - Monthly and daily TE are **consistent**
    - Standard calculations (daily TE × √252) work well
    
    **If your Effect is negative (-10% to -30%):**
    - Mean reversion behavior: returns oscillate
    - Monthly risk is **lower** than daily risk suggests
    - Daily TE may overstate your long-term tracking error
    
    **Bottom line:** Don't blindly trust daily TE × √252 if you have strong autocorrelation. 
    Use multiple methods (AR(1), Newey-West) to validate your risk estimates.
    """)


# --- Visualizations ---
st.divider()
st.subheader("Visualizations")

# Prepare data
viz_data = []
if uploaded_portfolio_data is not None:
    T_up = len(uploaded_portfolio_data)
    dates_up = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=T_up)
    for i, (date, value) in enumerate(zip(dates_up, uploaded_portfolio_data)):
        viz_data.append({"date": date, "regime": uploaded_portfolio_name, "value": value, "index": i})

for regime in selected_regimes:
    for i, (date, value) in enumerate(zip(dates, all_data[regime])):
        viz_data.append({"date": date, "regime": regime, "value": value, "index": i})

df_viz = pd.DataFrame(viz_data)

all_regime_names = list(regime_presets.keys())
all_colors = [regime_presets[r]["color"] for r in regime_presets.keys()]
if uploaded_portfolio_data is not None:
    all_regime_names.append(uploaded_portfolio_name)
    all_colors.append("#9b59b6")

color_scale = alt.Scale(domain=all_regime_names, range=all_colors)

tab1, tab2, tab3 = st.tabs(["Cumulative Drift", "Daily Returns", "Autocorrelation"])

with tab1:
    st.markdown("Cumulative active returns show how portfolio drift accumulates over time.")
    
    df_cum = df_viz.copy()
    df_cum = df_cum.sort_values(["regime", "index"])
    df_cum["cumulative"] = df_cum.groupby("regime")["value"].cumsum()
    
    chart_cum = (
        alt.Chart(df_cum)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("cumulative:Q", title="Cumulative Active Return"),
            color=alt.Color("regime:N", scale=color_scale, legend=alt.Legend(title="Portfolio")),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip("regime:N", title="Portfolio"),
                alt.Tooltip("cumulative:Q", title="Cumulative Return", format=".4f")
            ]
        )
        .properties(height=400)
        .interactive()
    )
    
    st.altair_chart(chart_cum, use_container_width=True)

with tab2:
    st.markdown("Daily active returns show raw volatility and patterns.")
    
    chart_daily = (
        alt.Chart(df_viz)
        .mark_line(strokeWidth=1.5, opacity=0.7)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Daily Active Return"),
            color=alt.Color("regime:N", scale=color_scale, legend=alt.Legend(title="Portfolio")),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip("regime:N", title="Portfolio"),
                alt.Tooltip("value:Q", title="Daily Return", format=".4f")
            ]
        )
        .properties(height=400)
        .interactive()
    )
    
    st.altair_chart(chart_daily, use_container_width=True)

with tab3:
    st.markdown("Autocorrelation function (ACF) shows serial correlation structure.")
    
    acf_data = []
    for regime_name, regime_data in all_data.items():
        a = regime_data
        max_lag = min(50, len(a) // 10)
        a_demean = a - np.mean(a)
        gamma_0 = np.dot(a_demean, a_demean) / len(a)
        
        for lag in range(max_lag + 1):
            if lag == 0:
                acf = 1.0
            else:
                gamma_h = np.dot(a_demean[:-lag], a_demean[lag:]) / len(a)
                acf = gamma_h / gamma_0
            
            acf_data.append({"Regime": regime_name, "Lag": lag, "ACF": acf})
    
    df_acf = pd.DataFrame(acf_data)
    
    chart_acf = (
        alt.Chart(df_acf)
        .mark_line(strokeWidth=2, point=True)
        .encode(
            x=alt.X("Lag:Q", title="Lag (days)"),
            y=alt.Y("ACF:Q", title="Autocorrelation", scale=alt.Scale(domain=[-0.5, 1])),
            color=alt.Color("Regime:N", scale=color_scale, legend=alt.Legend(title="Portfolio")),
            tooltip=[
                alt.Tooltip("Lag:Q", title="Lag"),
                alt.Tooltip("Regime:N", title="Portfolio"),
                alt.Tooltip("ACF:Q", title="ACF", format=".3f")
            ]
        )
        .properties(height=400)
    )
    
    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[5, 5], color="gray").encode(y="y:Q")
    
    st.altair_chart((chart_acf + zero_line), use_container_width=True)
    
    with st.expander("How to interpret the ACF"):
        st.markdown("""
        - **Positive ACF** at lags 1-5: Momentum/persistence
        - **Negative ACF** at lags 1-5: Mean reversion
        - **ACF ≈ 0** at all lags: Random walk
        """)

# Footer
st.divider()
st.markdown("© 2025 Tejas Viswanath • [LinkedIn](https://www.linkedin.com/in/tejasviswa/) • [GitHub](https://github.com/TejasViswa/) • [Email](mailto:tejasviswa@gmail.com)")
st.caption("Built with Streamlit • [View Source](https://github.com/TejasViswa/tracking-error-lab) • Privacy-friendly analytics (no cookies, no personal data)")
