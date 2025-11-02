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

# Custom CSS matching Quarto site
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
    
    /* Global font */
    html, body, [class*="css"], .stMarkdown, p, span, div {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 400;
        color: #2c3e50;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 600;
        letter-spacing: 0em;
        color: #2c3e50;
    }
    
    /* Top navigation bar */
    .top-nav {
        background-color: #f8f9fa;
        padding: 1rem 2rem;
        border-bottom: 1px solid #e8eaed;
        margin-bottom: 2rem;
    }
    
    .top-nav h1 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .top-nav-links {
        display: flex;
        gap: 1.5rem;
        margin-top: 0.5rem;
    }
    
    .top-nav-links a {
        color: #5a6c7d;
        text-decoration: none;
        font-size: 0.95rem;
        transition: color 0.2s;
    }
    
    .top-nav-links a:hover {
        color: #3498db;
    }
    
    /* Sidebar styling to match Quarto */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e8eaed;
    }
    
    [data-testid="stSidebar"] .sidebar-content {
        padding: 1.5rem 1rem;
    }
    
    /* Section headers in sidebar */
    .sidebar-section {
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #7f8c8d;
    }
    
    /* Sidebar links */
    .sidebar-link {
        display: block;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.25rem;
        color: #5a6c7d;
        text-decoration: none;
        border-radius: 0.25rem;
        transition: background-color 0.2s, color 0.2s;
    }
    
    .sidebar-link:hover {
        background-color: #e8eaed;
        color: #2c3e50;
    }
    
    /* Remove default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Metrics styling */
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.375rem;
        border-left: 3px solid #4a90e2;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #4a90e2;
        color: white;
        border: none;
        border-radius: 0.375rem;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    
    .stButton button:hover {
        background-color: #3498db;
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Footer */
    .custom-footer {
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 1px solid #e8eaed;
        text-align: center;
        color: #7f8c8d;
        font-size: 0.9rem;
    }
    
    .custom-footer a {
        color: #5a6c7d;
        text-decoration: none;
    }
    
    .custom-footer a:hover {
        color: #3498db;
    }
    
    /* Clean card styling */
    .info-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.375rem;
        border-left: 3px solid #e8eaed;
        margin-bottom: 1rem;
    }
    
    /* Remove emoji/icon clutter */
    .no-emoji {
        font-style: normal;
    }
</style>
""", unsafe_allow_html=True)

# Top Navigation Bar
st.markdown("""
<div class="top-nav">
    <h1>Tracking Error Lab</h1>
    <div class="top-nav-links">
        <a href="https://tejasviswa.github.io/tracking-error-lab/" target="_blank">Overview</a>
        <a href="https://tejasviswa.github.io/tracking-error-lab/intuitive_math.html" target="_blank">Intuitive Math</a>
        <a href="https://tejasviswa.github.io/tracking-error-lab/technical_math.html" target="_blank">Technical Details</a>
        <a href="https://tracking-error-lab-kgkju98o4pqxdjoevqtsay.streamlit.app/" target="_blank">Interactive Simulator</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
    st.markdown("""
    <a href="https://tejasviswa.github.io/tracking-error-lab/" target="_blank" class="sidebar-link">Overview</a>
    <a href="https://tejasviswa.github.io/tracking-error-lab/intuitive_math.html" target="_blank" class="sidebar-link">Intuitive Math</a>
    <a href="https://tejasviswa.github.io/tracking-error-lab/technical_math.html" target="_blank" class="sidebar-link">Technical Details</a>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">Resources</div>', unsafe_allow_html=True)
    st.markdown("""
    <a href="https://tracking-error-lab-kgkju98o4pqxdjoevqtsay.streamlit.app/" target="_blank" class="sidebar-link">Interactive Simulator</a>
    <a href="https://github.com/TejasViswa/tracking-error-lab" target="_blank" class="sidebar-link">Source Code</a>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">Contact</div>', unsafe_allow_html=True)
    st.markdown("""
    <a href="https://www.linkedin.com/in/tejasviswa/" target="_blank" class="sidebar-link">LinkedIn</a>
    <a href="https://github.com/TejasViswa/" target="_blank" class="sidebar-link">GitHub</a>
    <a href="mailto:tejasviswa@gmail.com" class="sidebar-link">Email</a>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Analysis mode in sidebar
    st.markdown("### Analysis Mode")
    analysis_mode = st.radio(
        "",
        ["Simulated Regimes", "Upload Portfolio", "Compare Both"],
        index=0,
        label_visibility="collapsed"
    )

# Main content
st.title("Interactive Simulator")
st.markdown("Explore how autocorrelation in active returns affects tracking error across different time horizons.")

st.markdown("---")

# Introduction
with st.expander("About This Tool"):
    st.markdown("""
    This interactive tool demonstrates how **autocorrelation** in active returns affects the relationship 
    between daily and monthly tracking error estimates.
    
    **Key Concepts:**
    - When active returns are **positively correlated** (momentum), monthly TE is higher than daily TE predicts
    - When active returns are **negatively correlated** (mean reversion), monthly TE is lower than daily TE predicts
    - When active returns are **uncorrelated** (random walk), monthly and daily TE estimates align
    
    Read the full mathematical treatment in the documentation pages linked above.
    """)

# --- Portfolio Upload Section ---
uploaded_portfolio_data = None
uploaded_portfolio_name = None

if analysis_mode in ["Upload Portfolio", "Compare Both"]:
    st.subheader("Upload Portfolio Data")
    
    st.markdown("""
    Upload a CSV or Excel file with your portfolio's **active returns** (portfolio return minus benchmark return).
    The file should contain a Date column and an Active Return column in decimal format (e.g., 0.0015 for 15 basis points).
    """)
    
    col_upload, col_example = st.columns([2, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose file",
            type=["csv", "xlsx", "xls"],
            help="CSV or Excel file with Date and Active Return columns"
        )
    
    with col_example:
        st.markdown("**Example format:**")
        st.code("""Date,Active Return
2023-01-03,0.0012
2023-01-04,-0.0008
2023-01-05,0.0015""", language="csv")
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
            
            # Display raw data preview
            with st.expander("Preview data"):
                st.dataframe(df_upload.head(10), use_container_width=True)
            
            # Column mapping
            st.markdown("**Configure columns:**")
            col_date_select, col_return_select = st.columns(2)
            
            with col_date_select:
                date_column = st.selectbox("Date column", options=df_upload.columns.tolist(), index=0)
            
            with col_return_select:
                return_column = st.selectbox("Active Return column", options=df_upload.columns.tolist(), 
                                            index=min(1, len(df_upload.columns) - 1))
            
            # Portfolio name
            uploaded_portfolio_name = st.text_input("Portfolio name", value="My Portfolio")
            
            # Parse and validate
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
                
                # Summary statistics
                st.success(f"Loaded {len(df_clean)} observations from {df_clean['date'].min().strftime('%Y-%m-%d')} to {df_clean['date'].max().strftime('%Y-%m-%d')}")
                
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    st.metric("Observations", f"{len(df_clean)}")
                with col_s2:
                    st.metric("Mean Return", f"{np.mean(uploaded_portfolio_data)*100:.2f}%")
                with col_s3:
                    st.metric("Std Dev", f"{np.std(uploaded_portfolio_data, ddof=1)*100:.2f}%")
                with col_s4:
                    st.metric("Range", f"{np.min(uploaded_portfolio_data)*100:.2f}% / {np.max(uploaded_portfolio_data)*100:.2f}%")
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    st.markdown("---")

# --- Regime Configuration ---
regime_presets = {
    "Persistent Drift (φ=+0.5)": {"phi": 0.5, "color": "#e74c3c"},
    "Random Walk (φ=0)": {"phi": 0.0, "color": "#3498db"},
    "Mean Reversion (φ=-0.45)": {"phi": -0.45, "color": "#27ae60"}
}

selected_regimes = []
if analysis_mode in ["Simulated Regimes", "Compare Both"]:
    st.subheader("Simulated Regimes")
    
    st.markdown("Select which autocorrelation regimes to simulate and compare:")
    
    selected_regimes = st.multiselect(
        "Select regimes",
        options=list(regime_presets.keys()),
        default=list(regime_presets.keys()),
        label_visibility="collapsed"
    )
    
    if not selected_regimes and analysis_mode != "Upload Portfolio":
        st.warning("Please select at least one regime.")
        st.stop()

if analysis_mode in ["Simulated Regimes", "Compare Both"]:
    st.markdown("**Simulation parameters:**")
    
    col_days, col_te, col_seed = st.columns(3)
    with col_days:
        T_days = st.slider("Trading days", 252, 2520, 756, 252)
    with col_te:
        target_annual_te_bps = st.slider("Target annual TE (bps)", 100, 1500, 500, 50)
    with col_seed:
        seed = st.number_input("Random seed", 0, 10_000, 42)
    
    # Advanced customization
    if selected_regimes:
        with st.expander("Advanced: Customize φ values"):
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
else:
    T_days = 756
    target_annual_te_bps = 500
    seed = 42
    custom_phi = {}

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
    
    # Add realistic features
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
st.markdown("---")
st.subheader("Results & Analysis")

if not selected_regimes and uploaded_portfolio_data is None:
    st.info("Select regimes or upload portfolio data to begin analysis.")
    st.stop()

all_data = {}
all_metrics = []

# Analyze uploaded portfolio
if uploaded_portfolio_data is not None:
    st.markdown("**Your Portfolio**")
    
    a_uploaded = uploaded_portfolio_data
    T_uploaded = len(a_uploaded)
    
    # Calculate metrics
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
    
    # Display metrics
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Estimated φ (AR(1))", f"{phi_estimated:+.3f}")
    with col_info2:
        st.metric("AR(1) R²", f"{r_squared:.3f}")
    with col_info3:
        if phi_estimated > 0.1:
            behavior = "Persistent Drift"
        elif phi_estimated < -0.1:
            behavior = "Mean Reversion"
        else:
            behavior = "Random Walk"
        st.metric("Behavior", behavior)
    
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
    
    st.markdown("---")

# Simulated regimes
if selected_regimes:
    st.markdown("**Simulated Regimes**")

dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=T_days)

for regime in selected_regimes:
    phi = custom_phi.get(regime, regime_presets[regime]["phi"])
    regime_seed = seed + list(regime_presets.keys()).index(regime)
    a = generate_realistic_ar1(phi, target_annual_te_bps, T_days, regime_seed)
    all_data[regime] = a
    
    # Calculate metrics
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

# Display metrics table
st.markdown("**Comparison of Estimation Methods**")
st.markdown("""
- **Daily TE (ann)**: Empirical daily std × √252
- **Monthly TE (ann)**: Empirical monthly std × √12
- **AR(1) Formula**: Theoretical closed-form solution
- **Newey-West**: Robust long-run variance estimator
""")

metrics_df = pd.DataFrame(all_metrics)
st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# Key insights
st.markdown("**Key Insights:**")
for idx, row in metrics_df.iterrows():
    portfolio_name = row["Portfolio"]
    effect = row["Effect"]
    st.markdown(f"• **{portfolio_name}**: Monthly TE is **{effect}** compared to daily TE prediction")

# --- Visualizations ---
st.markdown("---")
st.subheader("Visualizations")

# Prepare visualization data
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

# Color mapping
all_regime_names = list(regime_presets.keys())
all_colors = [regime_presets[r]["color"] for r in regime_presets.keys()]
if uploaded_portfolio_data is not None:
    all_regime_names.append(uploaded_portfolio_name)
    all_colors.append("#9b59b6")

color_scale = alt.Scale(domain=all_regime_names, range=all_colors)

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["Cumulative Drift", "Daily Returns", "Autocorrelation", "Methods Explained"])

with tab1:
    st.markdown("Cumulative active returns show how portfolio drift accumulates over time")
    
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
    st.markdown("Daily active returns show raw volatility and patterns")
    
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
    st.markdown("Autocorrelation function (ACF) shows serial correlation structure")
    
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
    
    st.markdown("""
    **Interpretation:**
    - Positive ACF at lags 1-5: Momentum/persistence in returns
    - Negative ACF at lags 1-5: Mean reversion in returns
    - ACF ≈ 0 at all lags: No serial correlation (random walk)
    """)

with tab4:
    st.markdown("### Mathematical Methods for TE Estimation")
    
    st.markdown("**1. Square-Root-of-Time Rule**")
    st.markdown("Assumes active returns are i.i.d. (no autocorrelation)")
    st.latex(r"TE_{\text{monthly,ann}} = TE_{\text{daily}} \times \sqrt{252} = TE_{\text{monthly}} \times \sqrt{12}")
    st.markdown("Valid when φ = 0 (random walk)")
    
    st.markdown("**2. AR(1) Closed-Form Solution**")
    st.markdown("Assumes active returns follow AR(1): $a_t = \\phi a_{t-1} + \\varepsilon_t$")
    st.latex(r"TE_m = \sqrt{D \cdot \gamma(0) \left[1 + \frac{2\phi}{1-\phi}\left(1 - \frac{1-\phi^D}{D(1-\phi)}\right)\right]}")
    st.markdown("Exact when autocorrelation structure is AR(1)")
    
    st.markdown("**3. Newey-West Long-Run Variance**")
    st.markdown("Robust approach with no parametric assumptions")
    st.latex(r"\widehat{\sigma}^2_{LR} = \hat{\gamma}(0) + 2\sum_{h=1}^{L}\left(1 - \frac{h}{L+1}\right)\hat{\gamma}(h)")
    st.latex(r"TE_{\text{annual,NW}} = \sqrt{252 \cdot \widehat{\sigma}^2_{LR}}")
    st.markdown("Most robust for real data with unknown autocorrelation")
    
    st.markdown("---")
    st.markdown("For detailed derivations, see the [Technical Math](https://tejasviswa.github.io/tracking-error-lab/technical_math.html) page.")

# Footer
st.markdown("""
<div class="custom-footer">
    <p>© 2025 Tejas Viswanath • 
    <a href="https://www.linkedin.com/in/tejasviswa/" target="_blank">LinkedIn</a> • 
    <a href="https://github.com/TejasViswa/" target="_blank">GitHub</a> • 
    <a href="mailto:tejasviswa@gmail.com">Email</a></p>
    <p>Built with Streamlit • <a href="https://github.com/TejasViswa/tracking-error-lab" target="_blank">View Source</a></p>
</div>
""", unsafe_allow_html=True)
