import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Tracking Error Lab", page_icon="📈", layout="wide")

# --- Header + links ---
left, right = st.columns([1, 1])
with left:
    st.title("Tracking Error Lab")
with right:
    st.markdown(
        "<div style='text-align:right'>"
        "🔗 <a href='https://github.com/TejasViswa/tracking-error-lab' target='_blank'>GitHub Repo</a>"
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("""
**Definition.** Let $a_t = r_{p,t} - r_{b,t}$ be active returns for period $t$.

Sample tracking error:
""")
st.latex(r"\widehat{TE}=\sqrt{\frac{1}{T-1}\sum_{t=1}^{T}(a_t-\bar a)^2},\quad \bar a=\frac{1}{T}\sum_{t=1}^{T}a_t")

st.markdown("Annualization: Daily × √252, Monthly × √12.")

# --- Simulator ---
st.header("Simulate Daily Active Returns")
T_days = st.slider("Trading days", 100, 2000, 756, 5)
sigma_bps = st.slider("Daily active vol (bps)", 1, 50, 12)
phi = st.slider("AR(1) coefficient (serial correlation)", -0.9, 0.9, 0.2, 0.1)
seed = st.number_input("Random seed", 0, 10_000, 42)

rng = np.random.default_rng(int(seed))
eps = rng.normal(0, 1, T_days)
a = np.zeros(T_days)
for t in range(1, T_days):
    a[t] = phi * a[t-1] + (sigma_bps/10000.0) * eps[t]

dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=T_days)
df = pd.DataFrame({"date": dates, "a": a}).set_index("date")

te_d = df["a"].std(ddof=1)
te_d_ann = te_d * np.sqrt(252)

df_m = df.resample("M").sum()
te_m = df_m["a"].std(ddof=1)
te_m_ann = te_m * np.sqrt(12)

c1, c2 = st.columns(2)
with c1:
    st.metric("Daily TE (annualized)", f"{te_d_ann*100:.2f}%")
with c2:
    st.metric("Monthly TE (annualized)", f"{te_m_ann*100:.2f}%")

# --- Altair: Daily active returns (interactive) ---
st.subheader("Active Return Over Time (Simulated)")
sim_chart = (
    alt.Chart(df.reset_index())
    .mark_line(point=True)
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("a:Q", title="Active Return"),
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("a:Q", title="Active Return", format=".5f")],
    )
    .properties(height=420)
    .interactive()
)
st.altair_chart(sim_chart, use_container_width=True)

# --- Upload section ---
st.header("Upload Your Returns")
st.caption("CSV with columns: date, rp, rb (same frequency per file).")
f = st.file_uploader("Upload CSV", type=["csv"])
freq = st.radio("Row frequency", ["Daily", "Monthly"], horizontal=True)
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

    # Altair chart for uploaded series
    st.subheader(f"Active Returns ({freq})")
    up_chart = (
        alt.Chart(dfu.reset_index())
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("a:Q", title="Active Return"),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("a:Q", title="Active Return", format=".5f")],
        )
        .properties(height=420)
        .interactive()
    )
    st.altair_chart(up_chart, use_container_width=True)

    if freq == "Daily":
        # Aggregate daily to monthly (sum of arithmetic daily returns per month)
        dfm2 = dfu["a"].resample("M").sum().to_frame("a_m")
        te_m2 = (dfm2["a_m"] - dfm2["a_m"].mean()).std(ddof=1)
        te_m2_ann = te_m2 * np.sqrt(12)

        st.subheader("From daily → monthly (aggregated)")
        st.write(f"Monthly TE (annualized): **{te_m2_ann*100:.2f}%**")

        # Altair bar chart for monthly aggregated active returns
        monthly_chart = (
            alt.Chart(dfm2.reset_index())
            .mark_bar()
            .encode(
                x=alt.X("date:T", title="Month"),
                y=alt.Y("a_m:Q", title="Monthly Active Return"),
                tooltip=[alt.Tooltip("date:T", title="Month"), alt.Tooltip("a_m:Q", title="Active Return", format=".5f")],
            )
            .properties(height=300)
            .interactive()
        )
        st.altair_chart(monthly_chart, use_container_width=True)

# --- Sidebar links ---
st.sidebar.markdown("**Docs:** [Read the article](https://tejasviswa.github.io/tracking-error-lab/)")
st.sidebar.markdown("**GitHub:** [TejasViswa/tracking-error-lab](https://github.com/TejasViswa/tracking-error-lab)")
