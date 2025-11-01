# 📊 Tracking Error Lab

**Interactive tools and comprehensive documentation** exploring why tracking error differs across time horizons — and what it means for portfolio management.

> **Key Insight:** When your portfolio is overweight in trending sectors (like Tech), monthly tracking error can be 15-30% higher than daily metrics suggest. When overweight in mean-reverting sectors (like Energy), it can be 10-20% lower. This isn't about market direction — it's about whether your active bets persist or reverse.

---

## 🔗 Live Resources

- 🧮 **Interactive App:** [tracking-error-lab.streamlit.app](https://tracking-error-lab-kgkju98o4pqxdjoevqtsay.streamlit.app/)
  - Simulate different portfolio behaviors
  - Upload your own data
  - Visualize autocorrelation effects
  
- 📖 **Documentation:** [tejasviswa.github.io/tracking-error-lab](https://tejasviswa.github.io/tracking-error-lab/)
  - **[Overview](https://tejasviswa.github.io/tracking-error-lab/)** — For advisors and portfolio managers
  - **[Intuitive Math](https://tejasviswa.github.io/tracking-error-lab/intuitive_math.html)** — Complete derivation with examples
  - **[Technical Math](https://tejasviswa.github.io/tracking-error-lab/technical_math.html)** — AR(1) models and Newey-West estimation

---

## 🎯 Who Should Use This?

### Portfolio Managers & Advisors
Understand why your risk reports might show different tracking error depending on the measurement period. Learn what portfolio characteristics drive these differences.

### Risk Managers
Calibrate TE budgets more accurately by accounting for autocorrelation in active returns. Stop being surprised by monthly TE that exceeds daily projections.

### Quantitative Analysts
Access complete mathematical derivations, closed-form AR(1) solutions, and Newey-West estimation code for implementation.

---

## 🧠 What You'll Learn

### The Problem
Traditional risk systems assume tracking error scales with √time. But this only works if returns are independent day-to-day. **Real portfolios don't work this way.**

### Three Portfolio Behaviors

| **Portfolio Type** | **Behavior** | **Effect on TE** | **Example** |
|-------------------|--------------|------------------|-------------|
| **Momentum/Growth** | Trending sectors persist | Monthly TE 15-30% **higher** | Tech-heavy, FAANG overweight |
| **Random Walk** | No serial correlation | Monthly TE matches daily | ESG screens, broad tilts |
| **Mean-Reverting** | Gains quickly reverse | Monthly TE 10-20% **lower** | Energy/Value overweights |

### Practical Implications
- Risk budgets may be too loose (momentum) or too conservative (mean reversion)
- Rebalancing frequency affects how much drift accumulates
- Client communication needs to account for horizon-dependent risk

---

## 🗂️ Repository Structure

| Component | Description | For Whom? |
|:----------|:------------|:----------|
| **`/app`** | Interactive Streamlit dashboard | Everyone — start here! |
| **`/site`** | Quarto documentation (3 levels) | Choose your depth |
| **`.github/workflows/`** | Auto-publishing for docs | Developers |

### Documentation Levels

```
Overview (index.qmd)
└─ For: Advisors, portfolio managers
   Focus: Practical implications, no math required
   
Intuitive Math (intuitive_math.qmd)
└─ For: Analysts who want to understand "why"
   Focus: Complete step-by-step derivation
   
Technical Math (technical_math.qmd)
└─ For: Quants implementing this in systems
   Focus: AR(1) models, Newey-West, Python code
```

---

## ⚙️ Local Setup

Clone and run locally:

```bash
git clone https://github.com/TejasViswa/tracking-error-lab.git
cd tracking-error-lab/app
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

## 🧩 Dependencies

- streamlit
- pandas
- numpy
- altair
- (optional) quarto CLI — if you want to build the documentation site locally

## 📜 License

Released under the MIT License.

## ✨ Author

Tejas Viswanath
- 📬 [LinkedIn](https://www.linkedin.com/in/tejasviswa/)
- 🧑‍💻 [Github](https://github.com/TejasViswa/)
- 📧 Email: tejasviswa@gmail.com