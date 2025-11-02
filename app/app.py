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

# Custom CSS for IBM Plex Sans font and better styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 400;
    }
    
    h1, h2, h3 {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
        letter-spacing: 0em;
    }
    
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4a90e2;
    }
    
    .regime-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Header + links ---
col_title, col_links = st.columns([3, 1])
with col_title:
    st.title("📊 Tracking Error Lab")
    st.markdown("**Interactive exploration of tracking error, autocorrelation, and annualization**")
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
with st.expander("ℹ️ What This Tool Does", expanded=False):
    st.markdown("""
    **Tracking error (TE)** measures how much a portfolio's returns deviate from its benchmark.
    
    **The Problem:** When we annualize tracking error from different frequencies (daily vs monthly), 
    we often get different answers! This happens because of **autocorrelation** in active returns.
    
    **This tool helps you:**
    - Visualize how autocorrelation affects tracking error
    - Compare different portfolio regimes (momentum, random walk, mean reversion)
    - See theoretical formulas (AR(1)) vs robust estimators (Newey-West)
    - Understand when monthly and daily TE estimates diverge
    
    📖 Full explanation: [Overview](https://tejasviswa.github.io/tracking-error-lab/) | 
    [Intuitive Math](https://tejasviswa.github.io/tracking-error-lab/intuitive_math.html) | 
    [Technical Details](https://tejasviswa.github.io/tracking-error-lab/technical_math.html)
    """)

# --- Mode Selection ---
st.header("🎯 Choose Analysis Mode")

analysis_mode = st.radio(
    "Select how you want to explore tracking error:",
    ["📊 Simulated Regimes", "📁 Upload Your Portfolio Data", "🔀 Both (Compare Uploaded vs Simulated)"],
    index=0,
    help="Choose to work with simulated data, your own portfolio data, or both"
)

st.markdown("---")

# --- Portfolio Upload Section ---
uploaded_portfolio_data = None
uploaded_portfolio_name = None

if analysis_mode in ["📁 Upload Your Portfolio Data", "🔀 Both (Compare Uploaded vs Simulated)"]:
    st.header("📁 Upload Portfolio Data")
    
    st.markdown("""
    Upload your portfolio's **active returns** (portfolio return - benchmark return) as a CSV or Excel file.
    The file should have two columns: **Date** and **Active Return** (in decimal form, e.g., 0.0015 for 15 bps).
    """)
    
    col_upload, col_example = st.columns([2, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose a file (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            help="File should contain Date and Active Return columns"
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
            with st.expander("📋 Preview uploaded data", expanded=False):
                st.dataframe(df_upload.head(10), use_container_width=True)
            
            # Parse columns
            st.subheader("🔧 Configure Data Mapping")
            col_date_select, col_return_select = st.columns(2)
            
            with col_date_select:
                date_column = st.selectbox(
                    "Select Date column:",
                    options=df_upload.columns.tolist(),
                    index=0
                )
            
            with col_return_select:
                return_column = st.selectbox(
                    "Select Active Return column:",
                    options=df_upload.columns.tolist(),
                    index=min(1, len(df_upload.columns) - 1)
                )
            
            # Portfolio name
            uploaded_portfolio_name = st.text_input(
                "Portfolio name (for charts):",
                value="My Portfolio",
                help="Give your portfolio a name for the analysis"
            )
            
            # Parse and validate
            df_clean = df_upload[[date_column, return_column]].copy()
            df_clean.columns = ["date", "active_return"]
            
            # Convert date
            df_clean["date"] = pd.to_datetime(df_clean["date"])
            df_clean = df_clean.sort_values("date").reset_index(drop=True)
            
            # Convert returns to float
            df_clean["active_return"] = pd.to_numeric(df_clean["active_return"], errors="coerce")
            
            # Drop NaN
            df_clean = df_clean.dropna()
            
            if len(df_clean) < 50:
                st.error("⚠️ Need at least 50 observations for meaningful analysis. Please upload more data.")
            else:
                uploaded_portfolio_data = df_clean["active_return"].values
                
                # Show summary statistics
                st.success(f"✅ Successfully loaded {len(df_clean)} observations from {df_clean['date'].min().strftime('%Y-%m-%d')} to {df_clean['date'].max().strftime('%Y-%m-%d')}")
                
                col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                with col_stats1:
                    st.metric("Observations", f"{len(df_clean)}")
                with col_stats2:
                    st.metric("Mean Active Return", f"{np.mean(uploaded_portfolio_data)*100:.2f}%")
                with col_stats3:
                    st.metric("Std Dev", f"{np.std(uploaded_portfolio_data, ddof=1)*100:.2f}%")
                with col_stats4:
                    st.metric("Min / Max", f"{np.min(uploaded_portfolio_data)*100:.2f}% / {np.max(uploaded_portfolio_data)*100:.2f}%")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please ensure your file has Date and Active Return columns.")
    
    st.markdown("---")

# --- Main Configuration ---
if analysis_mode in ["📊 Simulated Regimes", "🔀 Both (Compare Uploaded vs Simulated)"]:
    st.header("🎯 Select Simulated Regimes to Compare")
    
    st.markdown("""
    Choose which autocorrelation regimes to explore. By default, all three are shown so you can 
    see how different portfolio behaviors affect tracking error annualization.
    """)

# Define regime presets
regime_presets = {
    "Persistent Drift (φ=+0.5)": {
        "phi": 0.5,
        "color": "#e74c3c",
        "description": "Momentum/trend-following behavior (e.g., Tech/Growth funds)",
        "icon": "📈"
    },
    "Random Walk (φ=0)": {
        "phi": 0.0,
        "color": "#3498db",
        "description": "No serial correlation (e.g., well-diversified index trackers)",
        "icon": "🎲"
    },
    "Mean Reversion (φ=-0.45)": {
        "phi": -0.45,
        "color": "#27ae60",
        "description": "Contrarian/value behavior (e.g., frequent rebalancing)",
        "icon": "↩️"
    }
}

if analysis_mode in ["📊 Simulated Regimes", "🔀 Both (Compare Uploaded vs Simulated)"]:
    # Regime selector (multiselect)
    selected_regimes = st.multiselect(
        "Select regimes to compare:",
        options=list(regime_presets.keys()),
        default=list(regime_presets.keys()),  # All selected by default
        help="Choose one or more regimes to simulate and compare"
    )

    if not selected_regimes and analysis_mode != "📁 Upload Your Portfolio Data":
        st.warning("⚠️ Please select at least one regime to continue.")
        st.stop()
else:
    selected_regimes = []

if analysis_mode in ["📊 Simulated Regimes", "🔀 Both (Compare Uploaded vs Simulated)"]:
    # --- Simulation Parameters ---
    st.subheader("⚙️ Simulation Parameters")

    col_days, col_te, col_seed = st.columns(3)
    with col_days:
        T_days = st.slider("Trading days", 252, 2520, 756, 252, help="Number of trading days (~252 per year)")
    with col_te:
        target_annual_te_bps = st.slider("Target annual TE (bps)", 100, 1500, 500, 50, help="Target annualized tracking error in basis points")
    with col_seed:
        seed = st.number_input("Random seed", 0, 10_000, 42, help="Set seed for reproducibility")

    # Advanced regime customization
    if selected_regimes:
        with st.expander("🔧 Customize Regime Parameters", expanded=False):
            st.markdown("Override default φ (autocorrelation) values for each selected regime:")
            
            custom_phi = {}
            cols = st.columns(len(selected_regimes))
            for i, regime in enumerate(selected_regimes):
                with cols[i]:
                    st.markdown(f"**{regime}**")
                    default_phi = regime_presets[regime]["phi"]
                    custom_phi[regime] = st.slider(
                        f"φ for {regime.split('(')[0].strip()}",
                        -0.7, 0.7, default_phi, 0.05,
                        key=f"phi_{regime}",
                        help="AR(1) coefficient: positive = momentum, negative = mean reversion"
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
    """
    Generate realistic AR(1) active returns that look like real portfolio data.
    """
    rng = np.random.default_rng(int(seed))
    
    # Convert target annual TE from bps to decimal
    target_annual_te = target_annual_te_bps / 10000.0
    
    # Calculate daily volatility to achieve target annual TE
    target_daily_std = target_annual_te / np.sqrt(252)
    
    # Back out epsilon volatility from stationary variance
    if abs(phi) < 0.99:
        sigma_eps = target_daily_std * np.sqrt(1 - phi**2)
    else:
        sigma_eps = target_daily_std
    
    # Generate AR(1) process
    eps = rng.normal(0, sigma_eps, T_days)
    a = np.zeros(T_days)
    a[0] = eps[0]
    
    for t in range(1, T_days):
        a[t] = phi * a[t-1] + eps[t]
    
    # Add realistic features
    # 1. Occasional small jumps (market events)
    n_jumps = max(1, int(T_days / 252 * 2))
    jump_indices = rng.choice(T_days, size=n_jumps, replace=False)
    jump_sizes = rng.normal(0, target_daily_std * 3, n_jumps)
    a[jump_indices] += jump_sizes
    
    # 2. Slight heteroskedasticity (volatility clustering)
    vol_regime = np.ones(T_days)
    regime_changes = sorted(rng.choice(T_days, size=max(1, T_days // 126), replace=False))
    for i, change_pt in enumerate(regime_changes):
        regime_factor = rng.uniform(0.7, 1.3)
        end_pt = regime_changes[i+1] if i+1 < len(regime_changes) else T_days
        vol_regime[change_pt:end_pt] = regime_factor
    a = a * vol_regime
    
    return a

def calculate_ar1_theoretical_te(phi, sigma_daily, D=21):
    """
    Calculate theoretical monthly TE using AR(1) closed-form formula.
    
    From technical_math.qmd:
    TE_m = sqrt(D * gamma(0) * (1 + 2*phi/(1-phi) * (1 - (1-phi^D)/D/(1-phi))))
    """
    if abs(phi) < 0.99:
        gamma_0 = sigma_daily**2
        
        if abs(phi) < 1e-6:  # phi ≈ 0
            te_m_theoretical = np.sqrt(D * gamma_0)
        else:
            # Closed-form AR(1) formula
            sum_term = (1 - (1 - phi**D) / (D * (1 - phi)))
            factor = 1 + 2 * phi / (1 - phi) * sum_term
            te_m_theoretical = np.sqrt(D * gamma_0 * factor)
        
        return te_m_theoretical
    else:
        return np.sqrt(D) * sigma_daily

def newey_west_lrv(a, L=None):
    """
    Calculate Newey-West long-run variance estimator with Bartlett kernel.
    
    From technical_math.qmd:
    σ²_LR = γ(0) + 2*Σ(1 - h/(L+1))*γ(h)
    """
    T = len(a)
    a_demean = a - np.mean(a)
    
    # Default bandwidth: Andrews rule of thumb
    if L is None:
        L = int(np.floor(4 * (T/100)**(2/9)))
    
    # Compute autocovariances
    gamma_0 = np.dot(a_demean, a_demean) / T
    
    lrv = gamma_0
    for h in range(1, min(L+1, T)):
        gamma_h = np.dot(a_demean[:-h], a_demean[h:]) / T
        weight = 1 - h / (L + 1)
        lrv += 2 * weight * gamma_h
    
    return max(lrv, 0)  # Ensure non-negative

def estimate_ar1_parameters(a):
    """
    Estimate AR(1) parameters (φ, σ_ε) from observed active returns using OLS.
    
    AR(1): a_t = φ * a_{t-1} + ε_t
    
    Returns:
        phi: Estimated AR(1) coefficient
        sigma_eps: Estimated innovation std dev
        r_squared: R² of the AR(1) fit
    """
    # Demean the data
    a_demean = a - np.mean(a)
    
    # OLS regression: a_t on a_{t-1}
    y = a_demean[1:]  # a_t
    X = a_demean[:-1]  # a_{t-1}
    
    # Estimate φ
    phi_hat = np.dot(X, y) / np.dot(X, X)
    
    # Ensure stationarity
    phi_hat = np.clip(phi_hat, -0.99, 0.99)
    
    # Residuals
    residuals = y - phi_hat * X
    sigma_eps = np.std(residuals, ddof=1)
    
    # R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return phi_hat, sigma_eps, r_squared

# --- Generate Data for Selected Regimes ---
st.markdown("---")
st.header("📊 Results & Analysis")

# Check if we have any data to analyze
if not selected_regimes and uploaded_portfolio_data is None:
    st.warning("⚠️ Please select at least one regime or upload portfolio data to continue.")
    st.stop()

all_data = {}
all_metrics = []

# --- Analyze Uploaded Portfolio (if present) ---
if uploaded_portfolio_data is not None:
    st.subheader("📁 Your Portfolio Analysis")
    
    a_uploaded = uploaded_portfolio_data
    T_uploaded = len(a_uploaded)
    
    # Calculate metrics
    # 1. Daily TE (empirical)
    te_daily_up = np.std(a_uploaded, ddof=1)
    te_daily_ann_up = te_daily_up * np.sqrt(252)
    
    # 2. Monthly TE (empirical)
    # Estimate monthly periods (assuming ~21 trading days per month)
    days_per_month = 21
    n_months = T_uploaded // days_per_month
    if n_months >= 2:
        monthly_returns_up = [np.sum(a_uploaded[i*days_per_month:(i+1)*days_per_month]) 
                             for i in range(n_months)]
        te_monthly_up = np.std(monthly_returns_up, ddof=1)
        te_monthly_ann_up = te_monthly_up * np.sqrt(12)
    else:
        te_monthly_ann_up = te_daily_ann_up  # Fallback
    
    # 3. Estimate AR(1) parameters
    phi_estimated, sigma_eps_estimated, r_squared = estimate_ar1_parameters(a_uploaded)
    
    # 4. AR(1) Theoretical (using estimated φ)
    te_monthly_ar1_up = calculate_ar1_theoretical_te(phi_estimated, te_daily_up, D=21)
    te_monthly_ar1_ann_up = te_monthly_ar1_up * np.sqrt(12)
    
    # 5. Newey-West LRV
    lrv_nw_up = newey_west_lrv(a_uploaded, L=int(np.floor(4 * (T_uploaded/100)**(2/9))))
    te_annual_nw_up = np.sqrt(252 * lrv_nw_up)
    
    # Display uploaded portfolio metrics
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Estimated φ (AR(1))", f"{phi_estimated:+.3f}", 
                 help="Autocorrelation coefficient estimated from your data")
    with col_info2:
        st.metric("AR(1) R²", f"{r_squared:.3f}",
                 help="How well AR(1) model fits your data (higher is better)")
    with col_info3:
        if phi_estimated > 0.1:
            behavior = "📈 Persistent Drift"
        elif phi_estimated < -0.1:
            behavior = "↩️ Mean Reversion"
        else:
            behavior = "🎲 Random Walk"
        st.metric("Detected Behavior", behavior)
    
    # Store uploaded portfolio data and metrics
    all_data[uploaded_portfolio_name] = a_uploaded
    all_metrics.append({
        "Regime": f"{uploaded_portfolio_name} (Uploaded)",
        "φ (actual)": f"{phi_estimated:+.2f}",
        "Daily TE (ann)": f"{te_daily_ann_up*100:.2f}%",
        "Monthly TE (ann, empirical)": f"{te_monthly_ann_up*100:.2f}%",
        "Monthly TE (ann, AR(1) formula)": f"{te_monthly_ar1_ann_up*100:.2f}%",
        "Annual TE (Newey-West)": f"{te_annual_nw_up*100:.2f}%",
        "Ratio (Monthly/Daily)": f"{te_monthly_ann_up/te_daily_ann_up:.3f}",
        "Effect": f"{(te_monthly_ann_up/te_daily_ann_up - 1)*100:+.1f}%"
    })
    
    st.markdown("---")

# --- Simulated Regimes ---
if selected_regimes:
    st.subheader("🎲 Simulated Regimes Analysis")

dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=T_days)

for regime in selected_regimes:
    # Get phi (custom or default)
    if regime in custom_phi:
        phi = custom_phi[regime]
    else:
        phi = regime_presets[regime]["phi"]
    
    # Generate data
    regime_seed = seed + list(regime_presets.keys()).index(regime)
    a = generate_realistic_ar1(phi, target_annual_te_bps, T_days, regime_seed)
    all_data[regime] = a
    
    # Calculate metrics
    # 1. Daily TE (empirical)
    te_daily = np.std(a, ddof=1)
    te_daily_ann = te_daily * np.sqrt(252)
    
    # 2. Monthly TE (empirical)
    df_temp = pd.DataFrame({"date": dates, "return": a}).set_index("date")
    monthly_returns = df_temp["return"].resample("M").sum()
    te_monthly = np.std(monthly_returns, ddof=1)
    te_monthly_ann = te_monthly * np.sqrt(12)
    
    # 3. AR(1) Theoretical (closed-form)
    te_monthly_ar1 = calculate_ar1_theoretical_te(phi, te_daily, D=21)
    te_monthly_ar1_ann = te_monthly_ar1 * np.sqrt(12)
    
    # 4. Newey-West LRV
    lrv_nw = newey_west_lrv(a, L=int(np.floor(4 * (T_days/100)**(2/9))))
    te_annual_nw = np.sqrt(252 * lrv_nw)
    
    # Store metrics
    all_metrics.append({
        "Regime": regime,
        "φ (actual)": f"{phi:+.2f}",
        "Daily TE (ann)": f"{te_daily_ann*100:.2f}%",
        "Monthly TE (ann, empirical)": f"{te_monthly_ann*100:.2f}%",
        "Monthly TE (ann, AR(1) formula)": f"{te_monthly_ar1_ann*100:.2f}%",
        "Annual TE (Newey-West)": f"{te_annual_nw*100:.2f}%",
        "Ratio (Monthly/Daily)": f"{te_monthly_ann/te_daily_ann:.3f}",
        "Effect": f"{(te_monthly_ann/te_daily_ann - 1)*100:+.1f}%"
    })

# --- Display Metrics Table ---
st.subheader("📈 Tracking Error Comparison Across Methods")

st.markdown("""
This table shows **how different estimation methods compare** for each regime:
- **Daily TE (ann):** Empirical daily std × √252
- **Monthly TE (ann, empirical):** Empirical monthly std × √12
- **Monthly TE (ann, AR(1) formula):** Theoretical prediction using closed-form AR(1) solution
- **Annual TE (Newey-West):** Robust long-run variance estimator (handles unknown autocorrelation)
""")

metrics_df = pd.DataFrame(all_metrics)
st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# Key insights
st.markdown("**💡 Key Insights:**")
for idx, row in metrics_df.iterrows():
    regime_name = row["Regime"]
    effect = row["Effect"]
    
    # Check if it's uploaded or simulated
    if "(Uploaded)" in regime_name:
        color = "#9b59b6"  # Purple for uploaded
        icon = "📁"
        clean_name = regime_name.replace(" (Uploaded)", "")
    elif regime_name in regime_presets:
        color = regime_presets[regime_name]["color"]
        icon = regime_presets[regime_name]["icon"]
        clean_name = regime_name
    else:
        color = "#34495e"
        icon = "📊"
        clean_name = regime_name
    
    st.markdown(
        f"<div style='border-left: 4px solid {color}; padding-left: 1rem; margin-bottom: 0.5rem;'>"
        f"{icon} <strong>{clean_name}:</strong> Monthly TE is <strong>{effect}</strong> compared to daily TE prediction"
        f"</div>",
        unsafe_allow_html=True
    )

# --- Visualizations ---
st.markdown("---")
st.subheader("📉 Visual Comparison")

# Prepare data for visualization
# Create separate dataframes for uploaded and simulated data (they may have different lengths)
viz_data = []

# Add uploaded portfolio data
if uploaded_portfolio_data is not None:
    T_up = len(uploaded_portfolio_data)
    dates_up = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=T_up)
    for i, (date, value) in enumerate(zip(dates_up, uploaded_portfolio_data)):
        viz_data.append({
            "date": date,
            "regime": uploaded_portfolio_name,
            "value": value,
            "index": i,
            "is_uploaded": True
        })

# Add simulated regime data
for regime in selected_regimes:
    for i, (date, value) in enumerate(zip(dates, all_data[regime])):
        viz_data.append({
            "date": date,
            "regime": regime,
            "value": value,
            "index": i,
            "is_uploaded": False
        })

df_viz = pd.DataFrame(viz_data)

# Tab layout for different views
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Cumulative Drift", 
    "📈 Daily Returns", 
    "🔄 Autocorrelation", 
    "📐 Theoretical Formulas"
])

with tab1:
    st.markdown("**Cumulative active returns show how portfolio drift accumulates over time**")
    
    # Calculate cumulative returns by regime
    df_cum = df_viz.copy()
    df_cum = df_cum.sort_values(["regime", "index"])
    df_cum["cumulative"] = df_cum.groupby("regime")["value"].cumsum()
    
    # Color mapping (include uploaded portfolio)
    all_regime_names = list(regime_presets.keys())
    all_colors = [regime_presets[r]["color"] for r in regime_presets.keys()]
    if uploaded_portfolio_data is not None:
        all_regime_names.append(uploaded_portfolio_name)
        all_colors.append("#9b59b6")  # Purple for uploaded
    
    color_scale = alt.Scale(domain=all_regime_names, range=all_colors)
    
    chart_cum = (
        alt.Chart(df_cum)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("cumulative:Q", title="Cumulative Active Return"),
            color=alt.Color("regime:N", scale=color_scale, legend=alt.Legend(title="Regime", orient="bottom")),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip("regime:N", title="Regime"),
                alt.Tooltip("cumulative:Q", title="Cumulative Return", format=".4f")
            ]
        )
        .properties(height=400)
        .interactive()
    )
    
    st.altair_chart(chart_cum, use_container_width=True)
    
    st.markdown("""
    **What to look for:**
    - **Persistent drift** (red): Cumulative return trends away from zero → positive autocorrelation
    - **Random walk** (blue): Cumulative return meanders with no clear trend → zero autocorrelation
    - **Mean reversion** (green): Cumulative return oscillates around zero → negative autocorrelation
    - **Your portfolio** (purple): Compare against theoretical regimes
    """)

with tab2:
    st.markdown("**Daily active returns show the raw volatility and patterns**")
    
    chart_daily = (
        alt.Chart(df_viz)
        .mark_line(strokeWidth=1.5, opacity=0.7)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title="Daily Active Return"),
            color=alt.Color("regime:N", scale=color_scale, legend=alt.Legend(title="Regime", orient="bottom")),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip("regime:N", title="Regime"),
                alt.Tooltip("value:Q", title="Daily Return", format=".4f")
            ]
        )
        .properties(height=400)
        .interactive()
    )
    
    st.altair_chart(chart_daily, use_container_width=True)

with tab3:
    st.markdown("**Autocorrelation function (ACF) shows serial correlation structure**")
    
    # Calculate ACF for each regime (including uploaded)
    acf_data = []
    
    # Calculate for all regimes in all_data
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
            
            acf_data.append({
                "Regime": regime_name,
                "Lag": lag,
                "ACF": acf
            })
    
    df_acf = pd.DataFrame(acf_data)
    
    chart_acf = (
        alt.Chart(df_acf)
        .mark_line(strokeWidth=2, point=True)
        .encode(
            x=alt.X("Lag:Q", title="Lag (days)"),
            y=alt.Y("ACF:Q", title="Autocorrelation", scale=alt.Scale(domain=[-0.5, 1])),
            color=alt.Color("Regime:N", scale=color_scale, legend=alt.Legend(title="Regime", orient="bottom")),
            tooltip=[
                alt.Tooltip("Lag:Q", title="Lag"),
                alt.Tooltip("Regime:N", title="Regime"),
                alt.Tooltip("ACF:Q", title="ACF", format=".3f")
            ]
        )
        .properties(height=400)
    )
    
    # Add zero line
    zero_line = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(strokeDash=[5, 5], color="gray")
        .encode(y="y:Q")
    )
    
    st.altair_chart((chart_acf + zero_line), use_container_width=True)
    
    st.markdown("""
    **Interpretation:**
    - **Positive ACF** at lag 1-5: Returns are positively correlated → momentum/persistence
    - **Negative ACF** at lag 1-5: Returns are negatively correlated → mean reversion
    - **ACF ≈ 0** at all lags: No serial correlation → random walk
    """)

with tab4:
    st.markdown("**Mathematical formulas for tracking error under different assumptions**")
    
    st.markdown("### 1. Standard Square-Root-of-Time Rule")
    st.markdown("**Assumption:** Active returns are i.i.d. (no autocorrelation)")
    st.latex(r"TE_{\text{monthly,ann}} = TE_{\text{daily}} \times \sqrt{252} = TE_{\text{monthly}} \times \sqrt{12}")
    st.markdown("✅ Valid when φ = 0 (random walk)")
    
    st.markdown("### 2. AR(1) Closed-Form Solution")
    st.markdown("**Assumption:** Active returns follow AR(1) process: $a_t = \\phi a_{t-1} + \\varepsilon_t$")
    st.latex(r"TE_m = \sqrt{D \cdot \gamma(0) \left[1 + \frac{2\phi}{1-\phi}\left(1 - \frac{1-\phi^D}{D(1-\phi)}\right)\right]}")
    st.markdown("where $D$ = days per month (typically 21), $\\gamma(0)$ = variance of daily returns")
    st.markdown("✅ Exact when autocorrelation structure is truly AR(1)")
    
    st.markdown("### 3. Newey-West Long-Run Variance")
    st.markdown("**Assumption:** Unknown autocorrelation structure (robust approach)")
    st.latex(r"\widehat{\sigma}^2_{LR} = \hat{\gamma}(0) + 2\sum_{h=1}^{L}\left(1 - \frac{h}{L+1}\right)\hat{\gamma}(h)")
    st.latex(r"TE_{\text{annual,NW}} = \sqrt{252 \cdot \widehat{\sigma}^2_{LR}}")
    st.markdown("where $L$ = bandwidth (lag truncation), typically $L \\approx T^{1/4}$ or Andrews (1991) automatic selection")
    st.markdown("✅ Most robust; no parametric assumptions")
    
    st.markdown("---")
    st.markdown("**💡 Which method to use?**")
    st.markdown("""
    - **Square-root-of-time:** Quick estimate; assumes no autocorrelation
    - **AR(1) formula:** Best when you know autocorrelation is first-order
    - **Newey-West:** Most robust for real data with unknown autocorrelation structure
    
    📖 See [Technical Math](https://tejasviswa.github.io/tracking-error-lab/technical_math.html) for full derivations
    """)

# --- Educational Insights ---
st.markdown("---")
st.header("🎓 Key Takeaways")

col_edu1, col_edu2, col_edu3 = st.columns(3)

with col_edu1:
    st.markdown("### 📈 Positive Autocorrelation")
    st.markdown("""
    **What it means:** Yesterday's positive return → today's positive return (momentum)
    
    **Effect on TE:**
    - Monthly TE > Daily TE × √21
    - Drift accumulates
    - Simple annualization underestimates risk
    
    **Examples:** Tech/Growth funds, momentum strategies
    """)

with col_edu2:
    st.markdown("### 🎲 Zero Autocorrelation")
    st.markdown("""
    **What it means:** Past returns don't predict future returns
    
    **Effect on TE:**
    - Monthly TE ≈ Daily TE × √21
    - Square-root-of-time works!
    - Different frequencies agree
    
    **Examples:** Well-diversified trackers, random strategies
    """)

with col_edu3:
    st.markdown("### ↩️ Negative Autocorrelation")
    st.markdown("""
    **What it means:** Yesterday's positive return → today's negative return (mean reversion)
    
    **Effect on TE:**
    - Monthly TE < Daily TE × √21
    - Returns cancel out
    - Simple annualization overestimates risk
    
    **Examples:** Value/contrarian, frequent rebalancing
    """)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 0.9rem;'>
    <p>Built by <a href='https://www.linkedin.com/in/tejasviswa/' target='_blank'>Tejas Viswanath</a> • 
    <a href='https://github.com/TejasViswa/tracking-error-lab' target='_blank'>View Source</a> • 
    <a href='https://tejasviswa.github.io/tracking-error-lab/' target='_blank'>Read Documentation</a></p>
    <p>Based on rigorous time-series theory and portfolio risk management principles</p>
</div>
""", unsafe_allow_html=True)
